import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class OpenScoreCalculator:
    """Calculate openness scores for receivers based on defensive coverage"""
    
    def __init__(self, field_width: int = 1920, field_height: int = 1080):
        """
        Initialize OpenScore calculator
        
        Args:
            field_width: Video frame width (pixels)
            field_height: Video frame height (pixels)
        """
        self.field_width = field_width
        self.field_height = field_height
        
        # Scoring weights
        self.weights = {
            'distance': 0.4,      # Weight for defender distance
            'velocity': 0.25,     # Weight for defender velocity
            'separation': 0.25,   # Weight for route separation
            'coverage': 0.1       # Weight for coverage scheme
        }
    
    def calculate_openscore(
        self,
        receiver_data: Dict[str, Any],
        defenders_data: List[Dict[str, Any]],
        tracker,
        fps: float = 30.0
    ) -> float:
        """
        Calculate openness score for a receiver
        
        Args:
            receiver_data: Receiver tracking data
            defenders_data: List of defender tracking data
            tracker: PlayerTracker instance
            fps: Video frame rate
            
        Returns:
            OpenScore value (0-100, higher is more open)
        """
        if not defenders_data:
            return 100.0  # No defenders = completely open
        
        # Component scores
        distance_score = self._calculate_distance_score(receiver_data, defenders_data, tracker)
        velocity_score = self._calculate_velocity_score(receiver_data, defenders_data, tracker, fps)
        separation_score = self._calculate_separation_score(receiver_data, tracker)
        coverage_score = self._calculate_coverage_score(receiver_data, defenders_data)
        
        # Weighted combination
        openscore = (
            self.weights['distance'] * distance_score +
            self.weights['velocity'] * velocity_score +
            self.weights['separation'] * separation_score +
            self.weights['coverage'] * coverage_score
        )
        
        return float(np.clip(openscore, 0, 100))
    
    def _calculate_distance_score(
        self,
        receiver: Dict[str, Any],
        defenders: List[Dict[str, Any]],
        tracker
    ) -> float:
        """
        Calculate score based on distance to nearest defender
        Higher score = farther from defenders
        """
        receiver_pos = receiver['center']
        
        # Find minimum distance to any defender
        min_distance = float('inf')
        
        for defender in defenders:
            defender_pos = defender['center']
            distance = np.sqrt(
                (receiver_pos[0] - defender_pos[0])**2 +
                (receiver_pos[1] - defender_pos[1])**2
            )
            min_distance = min(min_distance, distance)
        
        # Normalize distance (assuming field diagonal as max)
        max_distance = np.sqrt(self.field_width**2 + self.field_height**2)
        normalized_distance = min_distance / max_distance
        
        # Convert to 0-100 scale with sigmoid-like curve
        return 100 * (1 - np.exp(-5 * normalized_distance))
    
    def _calculate_velocity_score(
        self,
        receiver: Dict[str, Any],
        defenders: List[Dict[str, Any]],
        tracker,
        fps: float
    ) -> float:
        """
        Calculate score based on defender approach velocity
        Higher score = defenders moving away or slower
        """
        receiver_track_id = receiver.get('track_id', -1)
        
        if receiver_track_id < 0:
            return 50.0  # Neutral score if no tracking
        
        # Get receiver position and velocity
        receiver_pos = receiver['center']
        receiver_vel = tracker.calculate_velocity(receiver_track_id, fps)
        
        # Calculate relative velocities of defenders
        threat_scores = []
        
        for defender in defenders:
            defender_track_id = defender.get('track_id', -1)
            
            if defender_track_id < 0:
                continue
            
            defender_pos = defender['center']
            defender_vel = tracker.calculate_velocity(defender_track_id, fps)
            
            # Calculate relative velocity (defender closing speed)
            # Vector from defender to receiver
            to_receiver = np.array([
                receiver_pos[0] - defender_pos[0],
                receiver_pos[1] - defender_pos[1]
            ])
            
            distance = np.linalg.norm(to_receiver)
            
            if distance > 0:
                direction = to_receiver / distance
                
                # Relative velocity in direction of receiver
                rel_vel = np.array([
                    defender_vel[0] - receiver_vel[0],
                    defender_vel[1] - receiver_vel[1]
                ])
                
                closing_speed = np.dot(rel_vel, direction)
                
                # Negative closing speed = defender moving away (good)
                # Positive closing speed = defender closing in (bad)
                threat_scores.append(-closing_speed)
        
        if not threat_scores:
            return 50.0
        
        # Use minimum (most threatening) score
        min_threat = min(threat_scores)
        
        # Normalize and convert to 0-100 scale
        # Assuming max closing speed of 500 pixels/second
        normalized = (min_threat + 500) / 1000
        return float(np.clip(normalized * 100, 0, 100))
    
    def _calculate_separation_score(
        self,
        receiver: Dict[str, Any],
        tracker
    ) -> float:
        """
        Calculate score based on route separation
        Higher score = better separation from route
        """
        receiver_track_id = receiver.get('track_id', -1)
        
        if receiver_track_id < 0:
            return 50.0
        
        # Get receiver's movement history
        history = tracker.get_track_history(receiver_track_id, window=15)
        
        if len(history) < 5:
            return 50.0
        
        # Calculate route smoothness (straighter = better separation)
        positions = np.array([h['center'] for h in history])
        
        # Calculate total distance vs straight-line distance
        total_dist = 0
        for i in range(1, len(positions)):
            total_dist += np.linalg.norm(positions[i] - positions[i-1])
        
        straight_dist = np.linalg.norm(positions[-1] - positions[0])
        
        if straight_dist == 0:
            return 50.0
        
        # Path efficiency (straight path = 1.0)
        efficiency = straight_dist / total_dist if total_dist > 0 else 1.0
        
        # Higher efficiency = better separation
        return float(efficiency * 100)
    
    def _calculate_coverage_score(
        self,
        receiver: Dict[str, Any],
        defenders: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate score based on coverage scheme detection
        Higher score = single/no coverage, lower = multiple defenders
        """
        receiver_pos = receiver['center']
        
        # Count defenders within certain radius
        coverage_radius = min(self.field_width, self.field_height) * 0.15
        nearby_defenders = 0
        
        for defender in defenders:
            defender_pos = defender['center']
            distance = np.sqrt(
                (receiver_pos[0] - defender_pos[0])**2 +
                (receiver_pos[1] - defender_pos[1])**2
            )
            
            if distance < coverage_radius:
                nearby_defenders += 1
        
        # Score based on number of nearby defenders
        if nearby_defenders == 0:
            return 100.0
        elif nearby_defenders == 1:
            return 70.0
        elif nearby_defenders == 2:
            return 40.0
        else:
            return 20.0
    
    def calculate_frame_openscores(
        self,
        tracked_detections: List[Dict[str, Any]],
        tracker,
        fps: float = 30.0
    ) -> Dict[int, float]:
        """
        Calculate openscores for all receivers in a frame
        
        Args:
            tracked_detections: All tracked detections in frame
            tracker: PlayerTracker instance
            fps: Video frame rate
            
        Returns:
            Dictionary mapping receiver track_id to openscore
        """
        # Separate receivers and defenders
        receivers = [d for d in tracked_detections if d['class_name'] in ['receiver', 'player']]
        defenders = [d for d in tracked_detections if d['class_name'] == 'defender']
        
        openscores = {}
        
        for receiver in receivers:
            track_id = receiver.get('track_id', -1)
            if track_id >= 0:
                score = self.calculate_openscore(receiver, defenders, tracker, fps)
                openscores[track_id] = score
        
        return openscores
    
    def get_best_option(
        self,
        openscores: Dict[int, float],
        min_threshold: float = 60.0
    ) -> Tuple[int, float]:
        """
        Get the best passing option based on openscores
        
        Args:
            openscores: Dictionary of receiver track_id to openscore
            min_threshold: Minimum acceptable openscore
            
        Returns:
            (track_id, score) of best option, or (-1, 0.0) if none good
        """
        if not openscores:
            return (-1, 0.0)
        
        # Find receiver with highest score
        best_track_id = max(openscores, key=openscores.get)
        best_score = openscores[best_track_id]
        
        # Check if meets threshold
        if best_score >= min_threshold:
            return (best_track_id, best_score)
        else:
            return (-1, 0.0)
    
    def draw_openscores(
        self,
        frame: np.ndarray,
        tracked_detections: List[Dict[str, Any]],
        openscores: Dict[int, float]
    ) -> np.ndarray:
        """
        Draw openscore visualization on frame
        
        Args:
            frame: Input frame
            tracked_detections: List of tracked detections
            openscores: Dictionary of openscores
            
        Returns:
            Annotated frame
        """
        import cv2
        
        annotated_frame = frame.copy()
        
        for det in tracked_detections:
            if det['class_name'] not in ['receiver', 'player']:
                continue
            
            track_id = det.get('track_id', -1)
            if track_id not in openscores:
                continue
            
            score = openscores[track_id]
            bbox = det['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Color based on score (green = open, red = covered)
            if score >= 70:
                color = (0, 255, 0)  # Green
            elif score >= 50:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw openscore label
            label = f"Open: {score:.1f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(
                annotated_frame,
                (x2 - w - 10, y1),
                (x2, y1 + h + 10),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (x2 - w - 5, y1 + h + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated_frame
