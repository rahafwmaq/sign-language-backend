import cv2
import mediapipe as mp
import numpy as np


class ImageFeatureExtractor:
    MAX_HANDS = 2
    LANDMARKS_PER_HAND = 21
    VALUES_PER_LANDMARK = 3

    HAND_FEATURE_SIZE = (
        LANDMARKS_PER_HAND * VALUES_PER_LANDMARK
    )

    FEATURE_SIZE = MAX_HANDS * HAND_FEATURE_SIZE

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=self.MAX_HANDS,
            min_detection_confidence=min_detection_confidence,
        )

    def extract_from_image(
        self,
        image_path: str,
    ) -> np.ndarray | None:
        image = cv2.imread(image_path)

        if image is None:
            return None

        return self.extract_from_array(image)

    def extract_from_array(
        self,
        image: np.ndarray,
    ) -> np.ndarray | None:
        if image.size == 0:
            return None

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        results = self._hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return None

        # دائمًا:
        # أول 63 قيمة = Left
        # آخر 63 قيمة = Right
        hand_slots = {
            "Left": [0.0] * self.HAND_FEATURE_SIZE,
            "Right": [0.0] * self.HAND_FEATURE_SIZE,
        }

        handedness_list = results.multi_handedness or []

        for index, hand_landmarks in enumerate(
            results.multi_hand_landmarks[: self.MAX_HANDS]
        ):
            normalized_hand = self._normalize_hand(
                hand_landmarks,
            )

            if index < len(handedness_list):
                hand_label = (
                    handedness_list[index]
                    .classification[0]
                    .label
                )
            else:
                # احتياط إذا لم يرجع MediaPipe نوع اليد
                hand_label = "Left" if index == 0 else "Right"

            hand_slots[hand_label] = normalized_hand

        features = (
            hand_slots["Left"]
            + hand_slots["Right"]
        )

        return np.asarray(
            features,
            dtype=np.float32,
        )

    def _normalize_hand(
        self,
        hand_landmarks,
    ) -> list[float]:
        landmarks = np.asarray(
            [
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
                for landmark in hand_landmarks.landmark
            ],
            dtype=np.float32,
        )

        # نجعل جميع النقاط نسبةً إلى المعصم.
        wrist = landmarks[0]
        relative_landmarks = landmarks - wrist

        # توحيد الحجم لتقليل تأثير قرب وبعد اليد من الكاميرا.
        scale = float(
            np.max(
                np.linalg.norm(
                    relative_landmarks[:, :2],
                    axis=1,
                )
            )
        )

        if scale > 1e-6:
            relative_landmarks /= scale

        return relative_landmarks.flatten().tolist()

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "ImageFeatureExtractor":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()