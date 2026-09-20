from typing import Any

import numpy as np

from image_sign.repositories.model_repository import ImageModelRepository
from image_sign.services.feature_extractor import ImageFeatureExtractor


class ImageModelPredictor:
    def __init__(self) -> None:
        self.repository = ImageModelRepository()

        self._model: Any | None = None
        self._labels_map: dict[str, Any] | None = None

    def predict(
        self,
        image_path: str,
    ) -> dict[str, Any]:
        with ImageFeatureExtractor() as extractor:
            features = extractor.extract_from_image(
                image_path,
            )

        if features is None:
            return {
                "status": "no_hand_detected",
                "label": "",
                "arabic": "",
                "english": "",
                "confidence": 0.0,
            }

        model = self._get_model()
        labels_map = self._get_labels_map()

        input_features = features.reshape(1, -1)

        prediction = str(
            model.predict(input_features)[0]
        )

        confidence = self._calculate_confidence(
            model=model,
            input_features=input_features,
        )

        label_info = labels_map.get(prediction)

        if not isinstance(label_info, dict):
            label_info = {
                "arabic": prediction,
                "english": "",
            }

        return {
            "status": "success",
            "label": prediction,
            "arabic": str(
                label_info.get(
                    "arabic",
                    prediction,
                )
            ),
            "english": str(
                label_info.get(
                    "english",
                    "",
                )
            ),
            "confidence": confidence,
        }

    def reload_model(self) -> None:
        self._model = self.repository.load_model()
        self._labels_map = (
            self.repository.load_labels_map()
        )

    def clear_cache(self) -> None:
        self._model = None
        self._labels_map = None

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self.repository.load_model()

        return self._model

    def _get_labels_map(
        self,
    ) -> dict[str, Any]:
        if self._labels_map is None:
            self._labels_map = (
                self.repository.load_labels_map()
            )

        return self._labels_map

    @staticmethod
    def _calculate_confidence(
        *,
        model: Any,
        input_features: np.ndarray,
    ) -> float:
        if not hasattr(model, "predict_proba"):
            return 0.0

        probabilities = model.predict_proba(
            input_features,
        )[0]

        confidence = float(
            np.max(probabilities) * 100
        )

        return round(confidence, 2)