import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import asyncio

from app.models.detection import PlayerDetector
from app.models.tracking import PlayerTracker
from app.models.classification import PlayerClassifier
from app.models.openscore import OpenScoreCalculator


class VideoProcessor:
    """Process NFL videos with detection, tracking, and openscore calculation"""
    
    def __init__(self):
        """Initialize video processor with models"""
        self.detector = PlayerDetector()
        self.tracker = PlayerTracker()
        self.classifier = PlayerClassifier(num_teams=2)
        self.openscore_calc = None  # Will be initialized with video dimensions
        
    async def process(
        self,
        video_path: str,
        task_id: str,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a video file
        
        Args:
            video_path: Path to input video
            task_id: Unique task identifier
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with processing results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Initialize openscore calculator with video dimensions
        self.openscore_calc = OpenScoreCalculator(width, height)
        
        # Prepare output video
        output_path = Path("outputs") / f"{task_id}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Storage for analysis data
        frame_data = []
        all_openscores = {}
        players_detected_set = set()
        
        frame_id = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Run detection
                detections = self.detector.detect(frame, conf_threshold=0.3)
                
                # Update tracker
                tracked_detections = self.tracker.update(detections, frame_id)
                
                # Classify into teams
                tracked_detections = self.classifier.classify(frame, tracked_detections)
                
                # Track unique players
                for det in tracked_detections:
                    track_id = det.get('track_id', -1)
                    if track_id >= 0:
                        players_detected_set.add(track_id)
                
                # Calculate openscores
                openscores = self.openscore_calc.calculate_frame_openscores(
                    tracked_detections,
                    self.tracker,
                    fps
                )
                
                # Store frame data
                frame_data.append({
                    'frame_id': frame_id,
                    'detections': len(tracked_detections),
                    'openscores': openscores
                })
                
                # Aggregate all openscores
                for track_id, score in openscores.items():
                    if track_id not in all_openscores:
                        all_openscores[track_id] = []
                    all_openscores[track_id].append(score)
                
                # Draw visualizations
                annotated_frame = self._annotate_frame(
                    frame,
                    tracked_detections,
                    openscores,
                    frame_id
                )
                
                # Write frame
                out.write(annotated_frame)
                
                # Update progress
                frame_id += 1
                if progress_callback and frame_id % 10 == 0:
                    progress = int((frame_id / total_frames) * 100)
                    progress_callback(progress)
                
                # Allow other async tasks to run
                if frame_id % 30 == 0:
                    await asyncio.sleep(0)
            
        finally:
            cap.release()
            out.release()
        
        # Calculate summary statistics
        openscore_summary = self._calculate_openscore_summary(all_openscores)
        tracking_summary = self.tracker.get_summary()
        
        return {
            'total_frames': frame_id,
            'fps': fps,
            'duration': duration,
            'players_detected': len(players_detected_set),
            'frame_data': frame_data,
            'openscore_summary': openscore_summary,
            'tracking_summary': tracking_summary,
            'output_path': str(output_path)
        }
    
    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracked_detections: list,
        openscores: dict,
        frame_id: int
    ) -> np.ndarray:
        """Annotate frame with all visualizations"""
        annotated = frame.copy()
        
        # Draw team-colored bounding boxes and tracking trails
        annotated = self._draw_team_assignments(annotated, tracked_detections)
        
        # Draw tracking trails
        annotated = self._draw_tracking_trails(annotated, tracked_detections)
        
        # Draw openscores
        annotated = self.openscore_calc.draw_openscores(
            annotated,
            tracked_detections,
            openscores
        )
        
        # Draw frame info
        cv2.putText(
            annotated,
            f"Frame: {frame_id}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Draw best option if available
        if openscores:
            best_track_id, best_score = self.openscore_calc.get_best_option(openscores)
            if best_track_id >= 0:
                cv2.putText(
                    annotated,
                    f"Best Option: ID {best_track_id} (Score: {best_score:.1f})",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
        
        return annotated
    
    def _draw_team_assignments(
        self,
        frame: np.ndarray,
        tracked_detections: list
    ) -> np.ndarray:
        """Draw team-colored bounding boxes on frame"""
        for det in tracked_detections:
            track_id = det['track_id']
            bbox = det['bbox']
            team_id = det.get('team_id', -1)
            team_color = det.get('team_color', (128, 128, 128))
            
            if track_id >= 0:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Draw bounding box with team color (thicker for visibility)
                cv2.rectangle(frame, (x1, y1), (x2, y2), team_color, 3)
                
                # Draw track ID with team color background
                track_label = f"ID: {track_id}"
                text_size = cv2.getTextSize(track_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x, text_y = x1, max(y1 - 5, 20)
                
                # Draw background rectangle for text
                cv2.rectangle(
                    frame,
                    (text_x - 2, text_y - text_size[1] - 2),
                    (text_x + text_size[0] + 2, text_y + 2),
                    team_color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    frame,
                    track_label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return frame
    
    def _draw_tracking_trails(
        self,
        frame: np.ndarray,
        tracked_detections: list
    ) -> np.ndarray:
        """Draw tracking trails in team colors"""
        for det in tracked_detections:
            track_id = det['track_id']
            team_color = det.get('team_color', (128, 128, 128))
            
            if track_id >= 0:
                history = self.tracker.get_track_history(track_id, window=30)
                if len(history) > 1:
                    points = np.array([h['center'] for h in history], dtype=np.int32)
                    cv2.polylines(
                        frame,
                        [points],
                        False,
                        team_color,
                        2
                    )
        
        return frame
    
    def _calculate_openscore_summary(self, all_openscores: dict) -> dict:
        """Calculate summary statistics for openscores"""
        summary = {}
        
        for track_id, scores in all_openscores.items():
            if scores:
                team_id = self.classifier.get_team_assignment(track_id)
                summary[f"player_{track_id}"] = {
                    'avg_openscore': float(np.mean(scores)),
                    'max_openscore': float(np.max(scores)),
                    'min_openscore': float(np.min(scores)),
                    'std_openscore': float(np.std(scores)),
                    'frames': len(scores),
                    'team_id': team_id
                }
        
        return summary
