import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from shared.paths import VideoPaths
from video_training.services.feature_extractor import (
    VideoFeatureExtractor,
)


class VideoPredictor:
    def __init__(
        self,
        *,
        sequence_length: int = 30,
        max_num_hands: int = 2,
    ) -> None:
        self.sequence_length = sequence_length
        self.max_num_hands = max_num_hands

    def predict(
        self,
        video_path: Path | str,
    ) -> dict[str, Any]:

        # =====================================================
        # CHECK MODEL
        # =====================================================

        if not VideoPaths.model_path.exists():
            raise FileNotFoundError(
                "Video model has not been trained yet."
            )

        # =====================================================
        # CHECK LABELS
        # =====================================================

        if not VideoPaths.labels_path.exists():
            raise FileNotFoundError(
                "Video labels map was not found."
            )

        # =====================================================
        # CHECK VIDEO
        # =====================================================

        path = Path(video_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Video file was not found: {path}"
            )

        # =====================================================
        # CREATE FRESH FEATURE EXTRACTOR
        # =====================================================

        feature_extractor = VideoFeatureExtractor(
            sequence_length=self.sequence_length,
            max_num_hands=self.max_num_hands,
        )

        try:

            # =================================================
            # EXTRACT FEATURES
            # =================================================

            features = feature_extractor.extract(
                path,
            )

            # =================================================
            # NO VALID SIGN DETECTED
            # =================================================

            if features is None:
                result = {
                    "folder": "no_sign",
                    "arabic": "",
                    "english": "",
                    "confidence": 0.0,
                    "detected": False,
                }

                print(
                    f"VIDEO PREDICTION RESULT: {result}"
                )

                return result

            # =================================================
            # LOAD MODEL
            # =================================================

            model = joblib.load(
                VideoPaths.model_path,
            )

            # =================================================
            # VALIDATE FEATURE SIZE
            # =================================================

            if hasattr(model, "n_features_in_"):
                expected_features = int(
                    model.n_features_in_,
                )

                if features.size != expected_features:
                    raise ValueError(
                        "The extracted video features do not match "
                        "the trained model configuration."
                    )

            # =================================================
            # CREATE BATCH
            # =================================================

            features_batch = np.asarray(
                [features],
                dtype=np.float32,
            )

            # =================================================
            # PREDICT
            # =================================================

            predicted_folder = str(
                model.predict(
                    features_batch,
                )[0]
            )

            # =================================================
            # CONFIDENCE
            # =================================================

            confidence = self._get_confidence(
                model=model,
                features_batch=features_batch,
                predicted_folder=predicted_folder,
            )

            # =================================================
            # LABEL MAP
            # =================================================

            labels_map = self._load_labels_map()

            label_data = labels_map.get(
                predicted_folder,
                {
                    "arabic": "",
                    "english": predicted_folder,
                    "folder": predicted_folder,
                },
            )

            # =================================================
            # RESULT
            # =================================================

            result = {
                "folder": predicted_folder,
                "arabic": str(
                    label_data.get(
                        "arabic",
                        "",
                    ),
                ),
                "english": str(
                    label_data.get(
                        "english",
                        predicted_folder,
                    ),
                ),
                "confidence": round(
                    confidence,
                    2,
                ),
                "detected": True,
            }

            print(
                f"VIDEO PREDICTION RESULT: {result}"
            )

            return result

        finally:

            try:
                feature_extractor.close()

            except Exception as error:
                print(
                    f"VIDEO FEATURE EXTRACTOR CLOSE WARNING: {error}"
                )

    # =========================================================
    # LABELS MAP
    # =========================================================

    def _load_labels_map(
        self,
    ) -> dict[str, dict[str, str]]:

        try:

            content = VideoPaths.labels_path.read_text(
                encoding="utf-8",
            )

            decoded = json.loads(
                content,
            )

        except OSError as error:
            raise ValueError(
                "Unable to read the video labels map."
            ) from error

        except json.JSONDecodeError as error:
            raise ValueError(
                "Video labels map contains invalid JSON."
            ) from error

        if not isinstance(
            decoded,
            dict,
        ):
            raise ValueError(
                "Video labels map must contain a JSON object."
            )

        labels_map = {
            str(key): value
            for key, value in decoded.items()
            if isinstance(
                value,
                dict,
            )
        }

        if not labels_map:
            raise ValueError(
                "Video labels map does not contain valid labels."
            )

        return labels_map

    # =========================================================
    # CONFIDENCE
    # =========================================================

    @staticmethod
    def _get_confidence(
        *,
        model: Any,
        features_batch: np.ndarray,
        predicted_folder: str,
    ) -> float:

        if not hasattr(
            model,
            "predict_proba",
        ):
            return 0.0

        probabilities = model.predict_proba(
            features_batch,
        )[0]

        classes = [
            str(item)
            for item in model.classes_
        ]

        if predicted_folder not in classes:
            return 0.0

        class_index = classes.index(
            predicted_folder,
        )

        return float(
            probabilities[class_index] * 100
        )