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
        # Per-player recent raw score history for adaptive scoring.
        self.player_score_history = defaultdict(list)
        self.history_window = 20
    
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
            # If defenders are missing, avoid hard-coding 100.
            # Fall back to receiver movement/separation with a bounded range.
            separation_score = self._calculate_separation_score(receiver_data, tracker)
            return float(np.clip(35.0 + 0.5 * separation_score, 0, 85))
        
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

    def _calculate_adaptive_score(self, track_id: int, raw_score: float) -> float:
        """
        Adapt score to player's recent context.
        Returns a blended score where recent baseline and volatility are considered.
        """
        history = self.player_score_history[track_id]

        if len(history) < 5:
            adaptive = raw_score
        else:
            baseline = float(np.mean(history))
            spread = max(float(np.std(history)), 5.0)
            # 50-centered relative score from player's trend.
            relative = 50.0 + 15.0 * ((raw_score - baseline) / spread)
            relative = float(np.clip(relative, 0, 100))
            adaptive = 0.65 * raw_score + 0.35 * relative

        history.append(float(raw_score))
        if len(history) > self.history_window:
            history.pop(0)

        return float(np.clip(adaptive, 0, 100))
    
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
        # Score ONLY offense players (never defenders).
        offense_players = [
            d for d in tracked_detections
            if d.get('side_role') == 'offense' and d.get('class_name') != 'ball'
        ]
        defense_players = [
            d for d in tracked_detections
            if d.get('side_role') == 'defense' and d.get('class_name') != 'ball'
        ]

        # If offense side-role mapping is not ready yet, skip scoring this frame.
        if not offense_players:
            return {}

        # Defensive fallback: all non-offense tracked players (except ball).
        if not defense_players:
            defense_players = [
                d for d in tracked_detections
                if d.get('track_id', -1) >= 0
                and d.get('class_name') != 'ball'
                and d.get('side_role') != 'offense'
            ]
        
        openscores = {}
        
        for player in offense_players:
            track_id = player.get('track_id', -1)
            if track_id >= 0:
                raw_score = self.calculate_openscore(player, defense_players, tracker, fps)
                adaptive_score = self._calculate_adaptive_score(track_id, raw_score)
                openscores[track_id] = adaptive_score
        
        return openscores

    def calculate_frame_openscores_with_context(
        self,
        tracked_detections: List[Dict[str, Any]],
        tracker,
        fps: float = 30.0
    ) -> Tuple[Dict[int, float], Dict[int, Dict[str, Any]]]:
        """
        Calculate openscores AND collect contextual data for each receiver.
        
        Returns:
            Tuple of (openscores dict, contexts dict)
            contexts maps track_id -> {nearest_defender_distance, num_nearby_defenders,
                                        closing_speed, separation_efficiency, field_diagonal}
        """
        offense_players = [
            d for d in tracked_detections
            if d.get('side_role') == 'offense' and d.get('class_name') != 'ball'
        ]
        defense_players = [
            d for d in tracked_detections
            if d.get('side_role') == 'defense' and d.get('class_name') != 'ball'
        ]

        if not offense_players:
            return {}, {}

        if not defense_players:
            defense_players = [
                d for d in tracked_detections
                if d.get('track_id', -1) >= 0
                and d.get('class_name') != 'ball'
                and d.get('side_role') != 'offense'
            ]

        openscores = {}
        contexts = {}
        field_diagonal = float(np.sqrt(self.field_width**2 + self.field_height**2))
        coverage_radius = min(self.field_width, self.field_height) * 0.15

        for player in offense_players:
            track_id = player.get('track_id', -1)
            if track_id < 0:
                continue

            raw_score = self.calculate_openscore(player, defense_players, tracker, fps)
            adaptive_score = self._calculate_adaptive_score(track_id, raw_score)
            openscores[track_id] = adaptive_score

            # --- Collect context for this player ---
            receiver_pos = player['center']

            # Nearest defender distance
            min_distance = float('inf')
            nearby_count = 0
            for defender in defense_players:
                d_pos = defender['center']
                dist = float(np.sqrt(
                    (receiver_pos[0] - d_pos[0])**2 +
                    (receiver_pos[1] - d_pos[1])**2
                ))
                min_distance = min(min_distance, dist)
                if dist < coverage_radius:
                    nearby_count += 1

            if min_distance == float('inf'):
                min_distance = 0.0

            # Closing speed of nearest defender
            closing_speed = 0.0
            receiver_vel = tracker.calculate_velocity(track_id, fps)
            for defender in defense_players:
                d_track_id = defender.get('track_id', -1)
                if d_track_id < 0:
                    continue
                d_pos = defender['center']
                dist = float(np.sqrt(
                    (receiver_pos[0] - d_pos[0])**2 +
                    (receiver_pos[1] - d_pos[1])**2
                ))
                if abs(dist - min_distance) < 1.0:  # this is the nearest defender
                    d_vel = tracker.calculate_velocity(d_track_id, fps)
                    to_receiver = np.array([
                        receiver_pos[0] - d_pos[0],
                        receiver_pos[1] - d_pos[1]
                    ])
                    norm = np.linalg.norm(to_receiver)
                    if norm > 0:
                        direction = to_receiver / norm
                        rel_vel = np.array([
                            d_vel[0] - receiver_vel[0],
                            d_vel[1] - receiver_vel[1]
                        ])
                        closing_speed = float(np.dot(rel_vel, direction))
                    break

            # Separation efficiency
            history = tracker.get_track_history(track_id, window=15)
            separation_eff = 0.5
            if len(history) >= 5:
                positions = np.array([h['center'] for h in history])
                total_dist = sum(
                    float(np.linalg.norm(positions[i] - positions[i-1]))
                    for i in range(1, len(positions))
                )
                straight_dist = float(np.linalg.norm(positions[-1] - positions[0]))
                if total_dist > 0:
                    separation_eff = straight_dist / total_dist

            contexts[track_id] = {
                'nearest_defender_distance': round(min_distance, 1),
                'num_nearby_defenders': nearby_count,
                'closing_speed': round(closing_speed, 1),
                'separation_efficiency': round(separation_eff, 3),
                'coverage_radius_used': round(coverage_radius, 1),
                'field_diagonal': round(field_diagonal, 1),
            }

        return openscores, contexts
    
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
            if det.get('side_role') != 'offense':
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
            
            # Draw openscore label above the player's head.
            label = f"Adaptive Open: {score:.1f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            center_x = int((x1 + x2) / 2)
            text_x = max(0, center_x - (w // 2))
            text_y = max(h + 8, y1 - 8)
            
            cv2.rectangle(
                annotated_frame,
                (text_x - 5, text_y - h - 5),
                (text_x + w + 5, text_y + 5),
                color,
                -1
            )
            cv2.putText(
                annotated_frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return annotated_frame
