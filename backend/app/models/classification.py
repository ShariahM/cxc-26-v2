import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from collections import defaultdict
import cv2


class PlayerClassifier:
    """Classify players into teams based on jersey colors with persistent team IDs"""
    
    def __init__(self, num_teams: int = 2, color_samples_per_player: int = 20):
        """
        Initialize team classifier
        
        Args:
            num_teams: Number of teams (default 2)
            color_samples_per_player: Number of frames to sample colors before assignment
        """
        self.num_teams = num_teams
        self.color_samples_per_player = color_samples_per_player
        
        # Store persistent team assignments by track_id
        self.team_assignments = {}  # track_id -> team_id (0 or 1)
        
        # Store team colors (HSV)
        self.team_colors = {}  # team_id -> (hue_mean, sat_mean, val_mean)
        
        # Collect color samples for clustering
        self.color_samples = defaultdict(list)  # track_id -> list of HSV colors
        
        # Track when teams are determined
        self.teams_determined = False
    
    def extract_jersey_color(
        self,
        frame: np.ndarray,
        bbox: List[float]
    ) -> Optional[np.ndarray]:
        """
        Extract dominant jersey color from bounding box region
        
        Args:
            frame: Input frame (BGR)
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            HSV color as [hue, saturation, value] or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Add padding to focus on jersey (skip head/arms)
            h = y2 - y1
            
            crop_y1 = y1 + int(h * 0.1)  # Skip head
            crop_y2 = y2 - int(h * 0.5)  # Skip feet
            
            region = frame[crop_y1:crop_y2, x1:x2]
            
            if region.size == 0:
                return None
            
            # Convert to HSV
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Calculate mean color (ignoring very dark pixels)
            mask = hsv[:, :, 2] > 30  # Value > 30
            valid_pixels = hsv[mask]
            
            if len(valid_pixels) == 0:
                return None
            
            mean_color = valid_pixels.mean(axis=0)
            return mean_color.astype(np.uint8)
        
        except Exception as e:
            return None
    
    def classify(
        self,
        frame: np.ndarray,
        tracked_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Classify players into teams based on jersey colors
        
        Args:
            frame: Input frame
            tracked_detections: List of tracked detections with track_ids
            
        Returns:
            Detections with added 'team_id' field
        """
        # Collect color samples
        for det in tracked_detections:
            track_id = det['track_id']
            
            if track_id < 0:  # Skip untracked detections
                continue
            
            color = self.extract_jersey_color(frame, det['bbox'])
            
            if color is not None and track_id not in self.team_assignments:
                self.color_samples[track_id].append(color)
        
        # Try to determine teams once we have enough samples
        if not self.teams_determined and len(self.color_samples) >= self.num_teams:
            self._determine_teams()
        
        # Assign team IDs to detections
        for det in tracked_detections:
            track_id = det['track_id']
            
            if track_id in self.team_assignments:
                det['team_id'] = self.team_assignments[track_id]
            else:
                det['team_id'] = -1  # Unclassified
        
        return tracked_detections
    
    def _determine_teams(self) -> None:
        """Determine team assignments using color clustering"""
        try:
            # Prepare color data for clustering
            all_colors = []
            track_ids = []
            
            for track_id, colors in self.color_samples.items():
                if len(colors) >= 3:  # Need minimum samples
                    # Use mean color of samples for this track
                    mean_color = np.array(colors).mean(axis=0)
                    all_colors.append(mean_color)
                    track_ids.append(track_id)
            
            if len(all_colors) < self.num_teams:
                return
            
            # Cluster colors using K-means
            all_colors = np.array(all_colors)
            kmeans = KMeans(n_clusters=self.num_teams, random_state=42, n_init=10)
            labels = kmeans.fit_predict(all_colors)
            
            # Assign teams
            for track_id, team_id in zip(track_ids, labels):
                self.team_assignments[track_id] = int(team_id)
                self.color_samples[track_id] = []  # Clear samples after assignment
            
            # Store team colors
            for team_id, center in enumerate(kmeans.cluster_centers_):
                self.team_colors[team_id] = tuple(center.astype(int))
            
            self.teams_determined = True
        
        except Exception as e:
            print(f"Error determining teams: {e}")
    
    def reassign_team(self, track_id: int, team_id: int) -> None:
        """
        Manually reassign a player to a different team
        
        Args:
            track_id: Track ID of player
            team_id: Team ID to assign (0 or 1)
        """
        if 0 <= team_id < self.num_teams:
            self.team_assignments[track_id] = team_id
            if track_id in self.color_samples:
                del self.color_samples[track_id]
    
    def get_team_assignment(self, track_id: int) -> Optional[int]:
        """Get team ID for a player"""
        return self.team_assignments.get(track_id)
    
    def get_all_assignments(self) -> Dict[int, int]:
        """Get all team assignments"""
        return self.team_assignments.copy()
    
    def get_team_stats(self) -> Dict[int, int]:
        """Get count of players per team"""
        stats = {team_id: 0 for team_id in range(self.num_teams)}
        for team_id in self.team_assignments.values():
            stats[team_id] += 1
        return stats
    
    def reset(self) -> None:
        """Reset classifier (for new game/video)"""
        self.team_assignments = {}
        self.color_samples = defaultdict(list)
        self.team_colors = {}
        self.teams_determined = False

