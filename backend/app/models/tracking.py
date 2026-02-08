import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import supervision as sv


class PlayerTracker:
    """Player tracking using ByteTrack algorithm"""
    
    def __init__(self):
        """Initialize ByteTrack tracker"""
        # Using supervision library's ByteTrack implementation
        self.tracker = sv.ByteTrack()
        
        # Store tracking history
        self.track_history = defaultdict(list)
        
        # Store player metadata
        self.player_info = {}
        
    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_id: int
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from detector
            frame_id: Current frame number
            
        Returns:
            List of tracked detections with track IDs
        """
        if not detections:
            return []
        
        # Convert detections to supervision format
        xyxy = np.array([det['bbox'] for det in detections])
        confidence = np.array([det['confidence'] for det in detections])
        class_id = np.array([det['class_id'] for det in detections])
        
        # Create Detections object
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )
        
        # Update tracker
        tracked = self.tracker.update_with_detections(sv_detections)
        
        # Convert back to our format with track IDs
        tracked_detections = []
        
        for i in range(len(tracked.xyxy)):
            track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            
            # Get original detection info
            bbox = tracked.xyxy[i].tolist()
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
            
            # Calculate center
            center = [
                (bbox[0] + bbox[2]) / 2,
                (bbox[1] + bbox[3]) / 2
            ]
            
            tracked_det = {
                'track_id': track_id,
                'bbox': bbox,
                'confidence': conf,
                'class_id': cls_id,
                'class_name': detections[i]['class_name'] if i < len(detections) else 'unknown',
                'center': center,
                'width': bbox[2] - bbox[0],
                'height': bbox[3] - bbox[1],
                'frame_id': frame_id
            }
            
            # Store in history
            if track_id >= 0:
                self.track_history[track_id].append({
                    'frame_id': frame_id,
                    'center': center,
                    'bbox': bbox
                })
                
                # Update player info
                if track_id not in self.player_info:
                    self.player_info[track_id] = {
                        'class_name': tracked_det['class_name'],
                        'first_seen': frame_id,
                        'last_seen': frame_id
                    }
                else:
                    self.player_info[track_id]['last_seen'] = frame_id
            
            tracked_detections.append(tracked_det)
        
        return tracked_detections
    
    def get_track_history(self, track_id: int, window: int = 10) -> List[Dict[str, Any]]:
        """
        Get tracking history for a specific track
        
        Args:
            track_id: Track ID
            window: Number of recent frames to return
            
        Returns:
            List of historical positions
        """
        if track_id not in self.track_history:
            return []
        
        return self.track_history[track_id][-window:]
    
    def calculate_velocity(
        self,
        track_id: int,
        fps: float = 30.0,
        window: int = 5
    ) -> Tuple[float, float]:
        """
        Calculate velocity for a tracked player
        
        Args:
            track_id: Track ID
            fps: Video frame rate
            window: Number of frames to use for calculation
            
        Returns:
            (vx, vy) velocity in pixels per second
        """
        history = self.get_track_history(track_id, window)
        
        if len(history) < 2:
            return (0.0, 0.0)
        
        # Calculate displacement
        start_pos = history[0]['center']
        end_pos = history[-1]['center']
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Calculate time difference
        dt = (history[-1]['frame_id'] - history[0]['frame_id']) / fps
        
        if dt == 0:
            return (0.0, 0.0)
        
        # Velocity in pixels per second
        vx = dx / dt
        vy = dy / dt
        
        return (vx, vy)
    
    def calculate_speed(
        self,
        track_id: int,
        fps: float = 30.0,
        window: int = 5
    ) -> float:
        """
        Calculate speed (magnitude of velocity)
        
        Args:
            track_id: Track ID
            fps: Video frame rate
            window: Number of frames to use
            
        Returns:
            Speed in pixels per second
        """
        vx, vy = self.calculate_velocity(track_id, fps, window)
        return np.sqrt(vx**2 + vy**2)
    
    def get_distance_between_tracks(
        self,
        track_id1: int,
        track_id2: int,
        frame_id: int = None
    ) -> float:
        """
        Calculate distance between two tracked players
        
        Args:
            track_id1: First track ID
            track_id2: Second track ID
            frame_id: Specific frame (None for latest)
            
        Returns:
            Euclidean distance in pixels
        """
        history1 = self.track_history.get(track_id1, [])
        history2 = self.track_history.get(track_id2, [])
        
        if not history1 or not history2:
            return float('inf')
        
        # Get positions at specific frame or latest
        if frame_id is not None:
            pos1 = next((h['center'] for h in history1 if h['frame_id'] == frame_id), None)
            pos2 = next((h['center'] for h in history2 if h['frame_id'] == frame_id), None)
        else:
            pos1 = history1[-1]['center']
            pos2 = history2[-1]['center']
        
        if pos1 is None or pos2 is None:
            return float('inf')
        
        # Calculate Euclidean distance
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracked_detections: List[Dict[str, Any]],
        show_trails: bool = True,
        team_roles: Optional[Dict[int, str]] = None
    ) -> np.ndarray:
        """
        Draw tracking visualization on frame
        
        Args:
            frame: Input frame
            tracked_detections: List of tracked detections
            show_trails: Whether to show tracking trails
            
        Returns:
            Annotated frame
        """
        import cv2
        
        annotated_frame = frame.copy()
        
        for det in tracked_detections:
            track_id = det['track_id']
            bbox = det['bbox']
            class_name = det['class_name']
            role = team_roles.get(track_id) if team_roles else None
            
            # Draw bounding box with track ID
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Team role colors take priority over class colors.
            if role == 'offense':
                color = (0, 0, 255)  # Red
            elif role == 'defense':
                color = (255, 0, 0)  # Blue
            else:
                colors = {
                    'quarterback': (255, 0, 0),  # Blue
                    'receiver': (0, 255, 255),   # Yellow
                    'defender': (0, 0, 255),     # Red
                    'player': (0, 255, 0),       # Green
                    'ball': (255, 255, 255)      # White
                }
                color = colors.get(class_name, (128, 128, 128))
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw role + track ID
            if role:
                label = f"{role.upper()} | ID: {track_id}"
            else:
                label = f"ID: {track_id}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Draw tracking trail
            if show_trails and track_id >= 0:
                history = self.get_track_history(track_id, window=30)
                if len(history) > 1:
                    points = np.array([h['center'] for h in history], dtype=np.int32)
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        False,
                        color,
                        2
                    )
        
        return annotated_frame
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get tracking summary statistics
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_tracks': len(self.player_info),
            'active_tracks': len([
                tid for tid, info in self.player_info.items()
                if len(self.track_history[tid]) > 0
            ]),
            'players_by_class': self._count_by_class(),
            'avg_track_length': np.mean([
                len(self.track_history[tid])
                for tid in self.player_info.keys()
            ]) if self.player_info else 0
        }
    
    def _count_by_class(self) -> Dict[str, int]:
        """Count tracks by class"""
        counts = defaultdict(int)
        for info in self.player_info.values():
            counts[info['class_name']] += 1
        return dict(counts)
