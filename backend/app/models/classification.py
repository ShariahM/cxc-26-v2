import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import cv2


class PlayerClassifier:
    """Classify players into teams based on jersey colors with persistent team IDs.
    
    Uses LAB color space for perceptually uniform distance, filters out grass/skin,
    and uses majority voting over multiple frames for robust assignment.
    Bounding boxes and trails are colored: Team 0 = Blue, Team 1 = Red.
    """
    
    def __init__(self, num_teams: int = 2, warmup_frames: int = 30):
        self.num_teams = num_teams
        self.warmup_frames = warmup_frames  # frames to collect before first clustering
        
        # Persistent team assignments: track_id -> team_id (0 or 1)
        self.team_assignments: Dict[int, int] = {}
        
        # Team cluster centres in LAB space (set after first clustering)
        self.team_centers_lab: Optional[np.ndarray] = None  # shape (num_teams, 3)
        
        # Visualization colours (BGR)
        self.team_viz_colors = {
            0: (255, 0, 0),    # Team 0: Blue
            1: (0, 0, 255),    # Team 1: Red
        }
        
        # Per-player colour history: track_id -> list of LAB colours (one per frame)
        self._color_history: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        # How many colour samples we need before we lock a player's team
        self._min_votes = 5
        
        # Frame counter
        self._frame_count = 0
        self._teams_ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        frame: np.ndarray,
        tracked_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add 'team_id' and 'team_color' to each detection."""
        self._frame_count += 1

        # 1. Extract jersey colour for every player in this frame
        colors_this_frame: Dict[int, np.ndarray] = {}
        for det in tracked_detections:
            tid = det['track_id']
            if tid < 0:
                continue
            lab = self._extract_jersey_lab(frame, det['bbox'])
            if lab is not None:
                self._color_history[tid].append(lab)
                colors_this_frame[tid] = lab

        # 2. If team centres not yet established, try to build them
        if not self._teams_ready:
            if self._frame_count >= self.warmup_frames:
                self._build_team_centers()
            # Even if we just built them, fall through to assignment below

        # 3. Assign teams
        if self._teams_ready:
            self._assign_pending_players()

        # 4. Stamp detections
        for det in tracked_detections:
            tid = det['track_id']
            if tid in self.team_assignments:
                team_id = self.team_assignments[tid]
                det['team_id'] = team_id
                det['team_color'] = self.team_viz_colors[team_id]
            else:
                det['team_id'] = -1
                det['team_color'] = (128, 128, 128)

        return tracked_detections

    # ------------------------------------------------------------------
    # Jersey colour extraction
    # ------------------------------------------------------------------

    def _extract_jersey_lab(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract dominant jersey colour in CIE-LAB from the upper-torso crop."""
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            h = y2 - y1
            w = x2 - x1
            if h < 10 or w < 6:
                return None

            # Upper-torso crop (skip head ~20%, legs ~40%)
            cy1 = max(0, y1 + int(h * 0.2))
            cy2 = min(frame.shape[0], y1 + int(h * 0.6))
            cx1 = max(0, x1 + int(w * 0.2))
            cx2 = min(frame.shape[1], x2 - int(w * 0.2))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None

            # Convert to HSV for masking, LAB for colour
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

            # Build mask: keep jersey-like pixels, reject grass / skin / dark / white
            h_chan = hsv[:, :, 0]
            s_chan = hsv[:, :, 1]
            v_chan = hsv[:, :, 2]

            # Reject very dark or very bright
            mask = (v_chan > 40) & (v_chan < 250)
            # Reject low-saturation (grays / whites) — but keep white jerseys via brightness
            mask = mask & (s_chan > 25)
            # Reject green / grass  (hue roughly 35-85 in OpenCV 0-180 range)
            grass = (h_chan >= 30) & (h_chan <= 90) & (s_chan > 40)
            mask = mask & (~grass)
            # Reject skin tones (hue ~5-25, moderate saturation)
            skin = (h_chan >= 5) & (h_chan <= 25) & (s_chan > 40) & (s_chan < 180)
            mask = mask & (~skin)

            valid = lab[mask]
            if len(valid) < 20:
                # Fallback: just use median of all non-dark pixels
                fallback_mask = v_chan > 50
                valid = lab[fallback_mask]
                if len(valid) < 10:
                    return None

            # Dominant colour = median (fast & robust to outliers)
            return np.median(valid, axis=0).astype(np.float32)

        except Exception:
            return None

    # ------------------------------------------------------------------
    # Team centre estimation
    # ------------------------------------------------------------------

    def _build_team_centers(self) -> None:
        """Cluster all collected player colours into num_teams groups (LAB space)."""
        # Compute per-player representative colour (median of history)
        player_colors = {}
        for tid, hist in self._color_history.items():
            if len(hist) >= 3:
                player_colors[tid] = np.median(np.array(hist), axis=0)

        if len(player_colors) < self.num_teams:
            return

        tids = list(player_colors.keys())
        X = np.array([player_colors[t] for t in tids], dtype=np.float32)

        kmeans = KMeans(n_clusters=self.num_teams, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        self.team_centers_lab = kmeans.cluster_centers_  # (num_teams, 3)
        self._teams_ready = True

        # Assign these initial players
        for tid, label in zip(tids, labels):
            self.team_assignments[tid] = int(label)

        print(f"[TeamClassifier] Centres built from {len(tids)} players.  "
              f"LAB centres: {self.team_centers_lab.tolist()}")

    # ------------------------------------------------------------------
    # Assigning new / pending players
    # ------------------------------------------------------------------

    def _assign_pending_players(self) -> None:
        """Assign any player that has enough colour history but no team yet."""
        if self.team_centers_lab is None:
            return

        for tid, hist in self._color_history.items():
            if tid in self.team_assignments:
                continue  # already assigned — never change
            if len(hist) < self._min_votes:
                continue  # not enough evidence yet

            # Majority-vote: classify each sample, pick the most common label
            votes = []
            for lab_color in hist:
                dists = np.linalg.norm(self.team_centers_lab - lab_color, axis=1)
                votes.append(int(np.argmin(dists)))

            team_id = Counter(votes).most_common(1)[0][0]
            self.team_assignments[tid] = team_id

    # ------------------------------------------------------------------
    # Nearest-team helper (used externally if needed)
    # ------------------------------------------------------------------

    def _classify_to_nearest_team(self, lab_color: np.ndarray) -> int:
        if self.team_centers_lab is None:
            return 0
        dists = np.linalg.norm(self.team_centers_lab - lab_color, axis=1)
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Utility / query methods
    # ------------------------------------------------------------------

    def get_team_assignment(self, track_id: int) -> Optional[int]:
        return self.team_assignments.get(track_id)

    def get_all_assignments(self) -> Dict[int, int]:
        return self.team_assignments.copy()

    def get_team_color_bgr(self, team_id: int) -> Tuple[int, int, int]:
        return self.team_viz_colors.get(team_id, (128, 128, 128))

    def get_team_stats(self) -> Dict[int, int]:
        stats = {i: 0 for i in range(self.num_teams)}
        for t in self.team_assignments.values():
            if t in stats:
                stats[t] += 1
        return stats

    def reassign_team(self, track_id: int, team_id: int) -> None:
        if 0 <= team_id < self.num_teams:
            self.team_assignments[track_id] = team_id

    def reset(self) -> None:
        self.team_assignments = {}
        self._color_history = defaultdict(list)
        self.team_centers_lab = None
        self._teams_ready = False
        self._frame_count = 0
