from pathlib import Path

import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands


class VideoFeatureExtractor:
    def __init__(
        self,
        sequence_length: int = 30,
        max_num_hands: int = 2,
    ) -> None:
        self.sequence_length = sequence_length
        self.max_num_hands = max_num_hands

        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @property
    def feature_size_per_frame(self) -> int:
        return 21 * 3 * self.max_num_hands

    @property
    def total_feature_size(self) -> int:
        return (
            self.sequence_length
            * self.feature_size_per_frame
        )

    def extract(
        self,
        video_path: Path | str,
    ) -> np.ndarray | None:
        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Video file was not found: {path}"
            )

        capture = cv2.VideoCapture(str(path))

        if not capture.isOpened():
            raise ValueError(
                f"Unable to open video file: {path.name}"
            )

        frame_features: list[np.ndarray] = []

        total_frames = 0
        detected_frames = 0

        try:
            while True:
                success, frame = capture.read()

                if not success:
                    break

                total_frames += 1

                features = self._extract_frame_features(
                    frame,
                )

                if features is not None:
                    detected_frames += 1
                    frame_features.append(features)

        finally:
            capture.release()

        print(
            f"VIDEO HAND DETECTION => "
            f"{detected_frames}/{total_frames}"
        )

        # =====================================================
        # REQUIRE ENOUGH HAND FRAMES
        # =====================================================

        minimum_hand_frames = 8

        if detected_frames < minimum_hand_frames:
            print(
                "VIDEO => Not enough hand frames."
            )
            return None

        # =====================================================
        # REQUIRE REAL MOVEMENT
        # =====================================================

        movement = self._calculate_movement(
            frame_features,
        )

        print(
            f"VIDEO MOVEMENT => {movement:.6f}"
        )

        minimum_movement = 0.015

        if movement < minimum_movement:
            print(
                "VIDEO => No meaningful sign movement detected."
            )
            return None

        sequence = self._normalise_sequence(
            frame_features,
        )

        return sequence.flatten().astype(
            np.float32,
        )

    def _calculate_movement(
        self,
        frames: list[np.ndarray],
    ) -> float:

        if len(frames) < 2:
            return 0.0

        movements: list[float] = []

        for index in range(
            1,
            len(frames),
        ):
            previous_frame = frames[index - 1]
            current_frame = frames[index]

            difference = np.abs(
                current_frame - previous_frame,
            )

            movements.append(
                float(np.mean(difference))
            )

        if not movements:
            return 0.0

        return float(
            np.mean(movements)
        )

    def _extract_frame_features(
        self,
        frame: np.ndarray,
    ) -> np.ndarray | None:

        if self._hands is None:
            raise RuntimeError(
                "MediaPipe Hands has already been closed."
            )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = self._hands.process(
            rgb_frame,
        )

        if not result.multi_hand_landmarks:
            return None

        detected_hands: list[np.ndarray] = []

        for hand_landmarks in result.multi_hand_landmarks:
            hand_features: list[float] = []

            for landmark in hand_landmarks.landmark:
                hand_features.extend(
                    [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ],
                )

            detected_hands.append(
                np.asarray(
                    hand_features,
                    dtype=np.float32,
                ),
            )

        detected_hands = detected_hands[
            : self.max_num_hands
        ]

        while len(detected_hands) < self.max_num_hands:
            detected_hands.append(
                np.zeros(
                    21 * 3,
                    dtype=np.float32,
                ),
            )

        return np.concatenate(
            detected_hands,
        )

    def _normalise_sequence(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray:
        total_frames = len(
            frames,
        )

        if total_frames == self.sequence_length:
            return np.asarray(
                frames,
                dtype=np.float32,
            )

        if total_frames > self.sequence_length:
            indexes = np.linspace(
                0,
                total_frames - 1,
                self.sequence_length,
            ).astype(int)

            return np.asarray(
                [
                    frames[index]
                    for index in indexes
                ],
                dtype=np.float32,
            )

        padded_frames = list(
            frames,
        )

        last_frame = frames[-1]

        while (
            len(padded_frames)
            < self.sequence_length
        ):
            padded_frames.append(
                last_frame.copy(),
            )

        return np.asarray(
            padded_frames,
            dtype=np.float32,
        )

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
            self._hands = None