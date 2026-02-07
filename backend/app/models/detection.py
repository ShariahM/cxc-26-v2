from ultralytics import YOLO
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import cv2


class PlayerDetector:
    """YOLOv8-based player detection for NFL videos"""
    
    def __init__(self, model_path: str = "models/best.pt"):
        """
        Initialize the YOLOv8 detector
        
        Args:
            model_path: Path to your custom YOLOv8 model
        """
        self.model_path = Path(model_path)
        
        # Check if model exists
        if not self.model_path.exists() and not Path("backend") in self.model_path.parts:
            # Try backend relative path
            alt_path = Path("backend") / model_path
            if alt_path.exists():
                self.model_path = alt_path
        
        try:
            self.model = YOLO(str(self.model_path))
            self.model.fuse()  # Fuse model for faster inference
        except Exception as e:
            print(f"Warning: Could not load custom model from {self.model_path}")
            print(f"Using default YOLO model. Error: {e}")
            # Fallback to default YOLO model for development
            self.model = YOLO('yolov8n.pt')
        
        # Class names (customize based on your model)
        self.class_names = {
            0: 'player',
            1: 'quarterback',
            2: 'receiver',
            3: 'defender',
            4: 'ball'
        }
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
        """
        Detect players in a single frame
        
        Args:
            frame: Input frame (numpy array)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detections with bounding boxes and metadata
        """
        # Run inference
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name
                class_name = self.class_names.get(class_id, 'unknown')
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': [
                        float((x1 + x2) / 2),
                        float((y1 + y2) / 2)
                    ],
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                }
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        conf_threshold: float = 0.25
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect players in multiple frames
        
        Args:
            frames: List of input frames
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection lists for each frame
        """
        all_detections = []
        
        # Batch inference for better performance
        results = self.model(frames, conf=conf_threshold, verbose=False)
        
        for result in results:
            frame_detections = []
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, 'unknown')
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'center': [
                        float((x1 + x2) / 2),
                        float((y1 + y2) / 2)
                    ],
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                }
                
                frame_detections.append(detection)
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_labels: Whether to show class labels
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Color mapping for different classes
        colors = {
            'player': (0, 255, 0),      # Green
            'quarterback': (255, 0, 0),  # Blue
            'receiver': (0, 255, 255),   # Yellow
            'defender': (0, 0, 255),     # Red
            'ball': (255, 255, 255)      # White
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(coord) for coord in det['bbox']]
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color
            color = colors.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels:
                label = f"{class_name}: {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return annotated_frame
