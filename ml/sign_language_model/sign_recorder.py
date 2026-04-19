import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime, timedelta

from models.core.sign_model import SignModel


class SignRecognitionResult:
    def __init__(
        self, sign_name: str, confidence: float, distance: float, motion_score: float = 1.0
    ):
        self.sign_name = sign_name
        self.confidence = confidence
        self.distance = distance
        self.motion_score = motion_score


class SignRecorder:
    def __init__(
        self,
        reference_signs: pd.DataFrame,
        min_frames=10,
        max_frames=60,
        pause_threshold_frames=8,
        stillness_threshold_frames=5,
        stillness_threshold_dist=0.01,
        confidence_threshold=0.6,
        cooldown_seconds=0.8,
        recognition_interval_frames=5,
    ):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.pause_threshold = pause_threshold_frames
        self.stillness_threshold_frames = stillness_threshold_frames
        self.stillness_threshold_dist = stillness_threshold_dist
        self.confidence_threshold = confidence_threshold
        self.cooldown = cooldown_seconds
        self.recognition_interval = recognition_interval_frames

        self.reference_signs = reference_signs.copy()
        self.reference_signs["distance"] = np.inf

        self.buffer = []
        self.pause_counter = 0
        self.stillness_counter = 0
        self.frames_since_recognition = 0
        self.last_detection_time = datetime.min
        self.last_detected_sign = ""

        self.is_recording = False
        self.current_sign = ""
        self.current_confidence = 0.0
        self.candidate_sign = ""
        self.candidate_confidence = 0.0
        self.phrase_buffer = []

        self._hand_state = {"left": False, "right": False}
        self._transition_detected = False
        self._last_landmarks = None

    def process_results(self, results) -> tuple:
        has_left = results.left_hand_landmarks is not None
        has_right = results.right_hand_landmarks is not None
        has_any_hand = has_left or has_right

        prev_state = self._hand_state.copy()
        self._hand_state = {"left": has_left, "right": has_right}

        raw_landmarks = self._extract_hand_raw(
            results.left_hand_landmarks, results.right_hand_landmarks
        )

        pose_landmarks = self._extract_pose_raw(results.pose_landmarks)

        frame_data = {
            "results": results,
            "timestamp": datetime.now(),
            "has_left": has_left,
            "has_right": has_right,
            "landmarks": raw_landmarks,
            "pose": pose_landmarks,
        }
        self.buffer.append(frame_data)

        if len(self.buffer) > self.max_frames:
            self.buffer.pop(0)

        if self.is_recording:
            if not has_any_hand:
                self.pause_counter += 1
                self.stillness_counter = 0
                if self.pause_counter >= self.pause_threshold:
                    self._finalize_sign()
            else:
                self.pause_counter = 0
                self._update_candidate()
                self._check_stillness(raw_landmarks)

            if len(self.buffer) >= self.max_frames:
                self._finalize_sign()
        else:
            if has_any_hand:
                self.is_recording = True
                self.pause_counter = 0
                self.stillness_counter = 0
                self.candidate_sign = ""
                self.candidate_confidence = 0.0
            elif len(self.buffer) > 0 and self._transition_detected:
                self._finalize_sign()

        self._transition_detected = self._detect_transition(prev_state, self._hand_state)
        self._last_landmarks = raw_landmarks

        return self.current_sign, self.is_recording

    def _check_stillness(self, current_landmarks):
        if self._last_landmarks is None or not self.is_recording:
            return

        if current_landmarks["left"] or current_landmarks["right"]:
            dist = self._landmarks_distance(self._last_landmarks, current_landmarks)
            if dist < self.stillness_threshold_dist:
                self.stillness_counter += 1
                if self.stillness_counter >= self.stillness_threshold_frames:
                    self._finalize_sign()
            else:
                self.stillness_counter = 0

    def _compute_recognition_fast(self, frames: list) -> SignRecognitionResult:
        left_hand_list = []
        right_hand_list = []
        pose_list = []

        for frame in frames:
            res = frame["results"]
            lh = self._extract_hand(res.left_hand_landmarks)
            rh = self._extract_hand(res.right_hand_landmarks)
            left_hand_list.append(lh)
            right_hand_list.append(rh)

            pose_data = frame.get("pose")
            if pose_data:
                pose_list.append([coord for point in pose_data for coord in point])

        if not left_hand_list and not right_hand_list and not pose_list:
            return None

        motion_score = self._compute_motion_score(frames)

        recorded_sign = SignModel(left_hand_list, right_hand_list, pose_list)

        distances = []

        for idx, row in self.reference_signs.iterrows():
            ref_model = row["sign_model"]

            if recorded_sign.has_left_hand != ref_model.has_left_hand:
                continue
            if recorded_sign.has_right_hand != ref_model.has_right_hand:
                continue

            dist = self._compute_simple_distance(recorded_sign, ref_model)
            distances.append((row["name"], dist))

        if not distances:
            return None

        distances.sort(key=lambda x: x[1])
        best_match, best_distance = distances[0]

        match_count = sum(1 for d in distances[:5] if d[0] == best_match)
        consensus_score = match_count / 5.0

        avg_distance = np.mean([d[1] for d in distances[:10]])
        distance_score = (
            max(0.0, 1.0 - (best_distance / (avg_distance * 2.0))) if avg_distance > 0 else 0.0
        )

        confidence = consensus_score * 0.5 + distance_score * 0.3 + motion_score * 0.2
        confidence = min(confidence, 0.85)

        return SignRecognitionResult(best_match, confidence, best_distance, motion_score)

    def _compute_simple_distance(self, recorded: SignModel, reference) -> float:
        distance = 0.0

        if recorded.has_left_hand and len(recorded.lh_embedding) > 0:
            ref_lh = reference.lh_embedding
            if len(ref_lh) > 0:
                rec_arr = np.mean(recorded.lh_embedding, axis=0)
                ref_arr = np.mean(ref_lh, axis=0)
                dist = np.linalg.norm(rec_arr - ref_arr)
                distance += dist

        if recorded.has_right_hand and len(recorded.rh_embedding) > 0:
            ref_rh = reference.rh_embedding
            if len(ref_rh) > 0:
                rec_arr = np.mean(recorded.rh_embedding, axis=0)
                ref_arr = np.mean(ref_rh, axis=0)
                dist = np.linalg.norm(rec_arr - ref_arr)
                distance += dist

        if recorded.has_pose and len(recorded.pose_embedding) > 0:
            ref_pose = getattr(reference, "pose_embedding", [])
            if ref_pose and len(ref_pose) > 0:
                rec_pose_arr = np.mean(recorded.pose_embedding, axis=0)
                ref_pose_arr = np.mean(ref_pose, axis=0)
                if len(rec_pose_arr) == len(ref_pose_arr):
                    pose_dist = np.linalg.norm(rec_pose_arr - ref_pose_arr)
                    distance += pose_dist * 0.5

        return float(distance)

    def _detect_transition(self, prev: dict, curr: dict) -> bool:
        for hand in ["left", "right"]:
            if prev.get(hand) != curr.get(hand):
                return True
        return False

    def _update_candidate(self):
        self.frames_since_recognition += 1

        if self.frames_since_recognition < self.recognition_interval:
            return

        self.frames_since_recognition = 0

        frames_with_hands = [f for f in self.buffer if f["has_left"] or f["has_right"]]

        if len(frames_with_hands) < self.min_frames:
            return

        result = self._compute_recognition_fast(frames_with_hands)
        if result:
            self.candidate_sign = result.sign_name
            self.candidate_confidence = result.confidence

    def _compute_recognition_fast(self, frames: list) -> SignRecognitionResult:
        left_hand_list = []
        right_hand_list = []
        pose_list = []

        for frame in frames:
            res = frame["results"]
            lh = self._extract_hand(res.left_hand_landmarks)
            rh = self._extract_hand(res.right_hand_landmarks)
            left_hand_list.append(lh)
            right_hand_list.append(rh)

            pose_data = frame.get("pose")
            if pose_data:
                pose_list.append([coord for point in pose_data for coord in point])

        if not left_hand_list and not right_hand_list and not pose_list:
            return None

        motion_score = self._compute_motion_score(frames)

        recorded_sign = SignModel(left_hand_list, right_hand_list, pose_list)

        distances = []

        for idx, row in self.reference_signs.iterrows():
            ref_model = row["sign_model"]

            if recorded_sign.has_left_hand != ref_model.has_left_hand:
                continue
            if recorded_sign.has_right_hand != ref_model.has_right_hand:
                continue

            dist = self._compute_simple_distance(recorded_sign, ref_model)
            distances.append((row["name"], dist))

        if not distances:
            return None

        distances.sort(key=lambda x: x[1])
        best_match, best_distance = distances[0]

        match_count = sum(1 for d in distances[:5] if d[0] == best_match)
        consensus_score = match_count / 5.0

        avg_distance = np.mean([d[1] for d in distances[:10]])
        distance_score = (
            max(0.0, 1.0 - (best_distance / (avg_distance * 2.0))) if avg_distance > 0 else 0.0
        )

        confidence = consensus_score * 0.5 + distance_score * 0.3 + motion_score * 0.2
        confidence = min(confidence, 0.85)

        return SignRecognitionResult(best_match, confidence, best_distance, motion_score)

    def _compute_simple_distance(self, recorded: SignModel, reference) -> float:
        distance = 0.0

        if recorded.has_left_hand and len(recorded.lh_embedding) > 0:
            ref_lh = reference.lh_embedding
            if len(ref_lh) > 0:
                rec_arr = np.mean(recorded.lh_embedding, axis=0)
                ref_arr = np.mean(ref_lh, axis=0)
                dist = np.linalg.norm(rec_arr - ref_arr)
                distance += dist

        if recorded.has_right_hand and len(recorded.rh_embedding) > 0:
            ref_rh = reference.rh_embedding
            if len(ref_rh) > 0:
                rec_arr = np.mean(recorded.rh_embedding, axis=0)
                ref_arr = np.mean(ref_rh, axis=0)
                dist = np.linalg.norm(rec_arr - ref_arr)
                distance += dist

        if recorded.has_pose and len(recorded.pose_embedding) > 0:
            ref_pose = getattr(reference, "pose_embedding", [])
            if ref_pose and len(ref_pose) > 0:
                rec_pose_arr = np.mean(recorded.pose_embedding, axis=0)
                ref_pose_arr = np.mean(ref_pose, axis=0)
                if len(rec_pose_arr) == len(ref_pose_arr):
                    pose_dist = np.linalg.norm(rec_pose_arr - ref_pose_arr)
                    distance += pose_dist * 0.5

        return float(distance)

    def _detect_transition(self, prev: dict, curr: dict) -> bool:
        for hand in ["left", "right"]:
            if prev.get(hand) != curr.get(hand):
                return True
        return False

    def _finalize_sign(self):
        if not self.is_recording:
            return

        frames_with_hands = [f for f in self.buffer if f["has_left"] or f["has_right"]]

        if len(frames_with_hands) < self.min_frames:
            self._reset_state()
            return

        now = datetime.now()
        time_since_last = (now - self.last_detection_time).total_seconds()
        if time_since_last < self.cooldown:
            self._reset_state()
            return

        result = self._compute_recognition(frames_with_hands)

        if result and result.confidence >= self.confidence_threshold:
            if result.sign_name != self.last_detected_sign:
                self.current_sign = result.sign_name
                self.current_confidence = result.confidence
                self.last_detected_sign = result.sign_name
                self.last_detection_time = now

                self.phrase_buffer.append(result.sign_name)
                if len(self.phrase_buffer) > 30:
                    self.phrase_buffer.pop(0)

                print(
                    f"[RECONOCIDO] {result.sign_name} (conf: {result.confidence:.2f}, dist: {result.distance:.0f}, motion: {result.motion_score:.2f})"
                )
            else:
                self.last_detection_time = now
        else:
            if result:
                print(
                    f"[DESCARTADO] {result.sign_name} (conf: {result.confidence:.2f}, motion: {result.motion_score:.2f})"
                )

        self.candidate_sign = ""
        self.candidate_confidence = 0.0
        self._reset_state()

    def _compute_recognition(self, frames: list) -> SignRecognitionResult:
        left_hand_list = []
        right_hand_list = []

        for frame in frames:
            res = frame["results"]
            lh = self._extract_hand(res.left_hand_landmarks)
            rh = self._extract_hand(res.right_hand_landmarks)
            left_hand_list.append(lh)
            right_hand_list.append(rh)

        if not left_hand_list and not right_hand_list:
            return None

        motion_score = self._compute_motion_score(frames)

        recorded_sign = SignModel(left_hand_list, right_hand_list)

        distances = []
        trajectory_scores = []

        for idx, row in self.reference_signs.iterrows():
            ref_model = row["sign_model"]

            if recorded_sign.has_left_hand != ref_model.has_left_hand:
                continue
            if recorded_sign.has_right_hand != ref_model.has_right_hand:
                continue

            dtw_dist = self._compute_dtw_distance(recorded_sign, ref_model)
            if dtw_dist < np.inf:
                traj_score = self._compute_trajectory_similarity(
                    left_hand_list,
                    right_hand_list,
                    ref_model.lh_embedding,
                    ref_model.rh_embedding,
                    ref_model.has_left_hand,
                    ref_model.has_right_hand,
                )
                distances.append((row["name"], dtw_dist, traj_score))

        if not distances:
            return None

        distances.sort(key=lambda x: x[1])
        best_match, best_distance, best_traj = distances[0]

        confidence = self._compute_confidence(distances, motion_score, best_traj)

        return SignRecognitionResult(best_match, confidence, best_distance, motion_score)

    def _compute_motion_score(self, frames: list) -> float:
        if len(frames) < 3:
            return 0.5

        motion_scores = []
        for i in range(1, len(frames)):
            prev = frames[i - 1]["landmarks"]
            curr = frames[i]["landmarks"]

            if prev and curr:
                dist = self._landmarks_distance(prev, curr)
                motion_scores.append(dist)

        if not motion_scores:
            return 0.5

        total_motion = np.mean(motion_scores)
        normalized = min(total_motion / 0.5, 2.0)
        return max(0.0, min(normalized, 1.0))

    def _landmarks_distance(self, l1, l2) -> float:
        if not l1 or not l2:
            return 0.0

        dist_l = 0.0
        dist_r = 0.0

        if l1["left"] and l2["left"]:
            arr1 = np.array(l1["left"]).flatten()
            arr2 = np.array(l2["left"]).flatten()
            dist_l = np.linalg.norm(arr1 - arr2)

        if l1["right"] and l2["right"]:
            arr1 = np.array(l1["right"]).flatten()
            arr2 = np.array(l2["right"]).flatten()
            dist_r = np.linalg.norm(arr1 - arr2)

        return dist_l + dist_r

    def _compute_trajectory_similarity(self, rec_l, rec_r, ref_l, ref_r, has_l, has_r) -> float:
        if len(rec_l) < 3 or len(ref_l) < 3:
            return 0.5

        n_phases = 5
        phase_weights = [0.15, 0.2, 0.3, 0.2, 0.15]

        score = 0.0
        total_weight = 0.0

        rec_l_arr = np.array(rec_l) if rec_l else None
        rec_r_arr = np.array(rec_r) if rec_r else None
        ref_l_arr = np.array(ref_l) if (has_l and ref_l) else None
        ref_r_arr = np.array(ref_r) if (has_r and ref_r) else None

        rec_len = (
            len(rec_l_arr)
            if rec_l_arr is not None
            else (len(rec_r_arr) if rec_r_arr is not None else 1)
        )
        ref_len = (
            len(ref_l_arr)
            if ref_l_arr is not None
            else (len(ref_r_arr) if ref_r_arr is not None else 1)
        )

        for i, weight in enumerate(phase_weights):
            rec_start = int(i * rec_len / n_phases)
            rec_end = int((i + 1) * rec_len / n_phases)
            ref_start = int(i * ref_len / n_phases)
            ref_end = int((i + 1) * ref_len / n_phases)

            phase_score = 0.0
            comparisons = 0

            if (
                rec_l_arr is not None
                and ref_l_arr is not None
                and len(rec_l_arr) > 0
                and len(ref_l_arr) > 0
            ):
                rec_phase = rec_l_arr[rec_start:rec_end] if rec_end > rec_start else rec_l_arr[-1:]
                ref_phase = ref_l_arr[ref_start:ref_end] if ref_end > ref_start else ref_l_arr[-1:]
                if len(rec_phase) > 0 and len(ref_phase) > 0:
                    rec_mean = np.mean(rec_phase, axis=0)
                    ref_mean = np.mean(ref_phase, axis=0)
                    if rec_mean.shape == ref_mean.shape:
                        phase_score += 1.0 - min(np.linalg.norm(rec_mean - ref_mean) / 2.0, 1.0)
                        comparisons += 1

            if (
                rec_r_arr is not None
                and ref_r_arr is not None
                and len(rec_r_arr) > 0
                and len(ref_r_arr) > 0
            ):
                rec_phase = rec_r_arr[rec_start:rec_end] if rec_end > rec_start else rec_r_arr[-1:]
                ref_phase = ref_r_arr[ref_start:ref_end] if ref_end > ref_start else ref_r_arr[-1:]
                if len(rec_phase) > 0 and len(ref_phase) > 0:
                    rec_mean = np.mean(rec_phase, axis=0)
                    ref_mean = np.mean(ref_phase, axis=0)
                    if rec_mean.shape == ref_mean.shape:
                        phase_score += 1.0 - min(np.linalg.norm(rec_mean - ref_mean) / 2.0, 1.0)
                        comparisons += 1

            if comparisons > 0:
                score += (phase_score / comparisons) * weight
                total_weight += weight

        return score / total_weight if total_weight > 0 else 0.5

    def _compute_confidence(
        self, distances: list, motion_score: float, trajectory_score: float
    ) -> float:
        if not distances:
            return 0.0

        best_match, best_distance, best_traj = distances[0]

        match_count = sum(1 for d in distances[:7] if d[0] == best_match)
        consensus_score = match_count / 7.0

        avg_distance = np.mean([d[1] for d in distances[:10]])
        distance_score = (
            max(0.0, 1.0 - (best_distance / (avg_distance * 1.2))) if avg_distance > 0 else 0.0
        )

        trajectory_weight = 0.35
        consensus_weight = 0.35
        distance_weight = 0.20
        motion_weight = 0.10

        confidence = (
            trajectory_score * trajectory_weight
            + consensus_score * consensus_weight
            + distance_score * distance_weight
            + motion_score * motion_weight
        )

        return min(confidence, 0.99)

    def _compute_dtw_distance(self, recorded: SignModel, reference) -> float:
        from fastdtw import fastdtw

        distance = 0.0

        if recorded.has_left_hand and len(recorded.lh_embedding) > 0:
            ref_lh = reference.lh_embedding
            if len(ref_lh) > 0:
                dist_l, _ = fastdtw(recorded.lh_embedding, ref_lh)
                distance += dist_l

        if recorded.has_right_hand and len(recorded.rh_embedding) > 0:
            ref_rh = reference.rh_embedding
            if len(ref_rh) > 0:
                dist_r, _ = fastdtw(recorded.rh_embedding, ref_rh)
                distance += dist_r

        return distance

    @staticmethod
    def _extract_hand(landmarks):
        if landmarks:
            points = [[p.x, p.y, p.z] for p in landmarks]
            arr = np.array(points, dtype=np.float64).flatten()
            return arr.tolist()
        return [0.0] * 63

    @staticmethod
    def _extract_hand_raw(left_landmarks, right_landmarks):
        left = None
        right = None

        if left_landmarks:
            left = [[p.x, p.y, p.z] for p in left_landmarks]
        if right_landmarks:
            right = [[p.x, p.y, p.z] for p in right_landmarks]

        return {"left": left, "right": right}

    @staticmethod
    def _extract_pose_raw(pose_landmarks):
        if not pose_landmarks:
            return None

        POSE_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]

        try:
            pose_points = [[p.x, p.y, p.z] for p in pose_landmarks]
            filtered = [pose_points[i] for i in POSE_INDICES if i < len(pose_points)]
            return filtered if filtered else None
        except:
            return None

    def _reset_state(self):
        self.buffer = []
        self.is_recording = False
        self.pause_counter = 0
        self.stillness_counter = 0
        self.frames_since_recognition = 0
        self.current_sign = ""
        self.current_confidence = 0.0

    def get_phrase(self) -> str:
        return " ".join(self.phrase_buffer)

    def clear_phrase(self):
        self.phrase_buffer = []
        self.last_detected_sign = ""
        print("Frase borrada")
