import json
from pathlib import Path
from typing import Any

import joblib

from image_sign.managers.metadata_manager import (
    ImageMetadataManager,
)
from image_sign.models.training_result import (
    ImageTrainingResult,
)
from shared.file_manager import FileManager
from shared.paths import AppPaths, ImagePaths


class ImageModelRepository:
    def __init__(self) -> None:
        FileManager.ensure_dir(AppPaths.models_dir)

        self.model_path = ImagePaths.model_path
        self.labels_path = ImagePaths.labels_path
        self.metadata_manager = ImageMetadataManager()

    def save_model(self, model: Any) -> Path:
        FileManager.ensure_dir(
            self.model_path.parent,
        )

        joblib.dump(
            model,
            self.model_path,
        )

        return self.model_path

    def load_model(self) -> Any:
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Image model not found. "
                "Please train the model first."
            )

        return joblib.load(
            self.model_path,
        )

    def model_exists(self) -> bool:
        return self.model_path.exists()

    def save_labels_map(
        self,
        labels_map: dict[str, Any],
    ) -> Path:
        if not labels_map:
            raise ValueError(
                "Labels map cannot be empty."
            )

        FileManager.ensure_dir(
            self.labels_path.parent,
        )

        self.labels_path.write_text(
            json.dumps(
                labels_map,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return self.labels_path

    def load_labels_map(self) -> dict[str, Any]:
        if not self.labels_path.exists():
            raise FileNotFoundError(
                "Image labels map not found. "
                "Please train the model first."
            )

        content = self.labels_path.read_text(
            encoding="utf-8",
        ).strip()

        if not content:
            raise ValueError(
                "Image labels map is empty."
            )

        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid image labels map: {error}"
            ) from error

        if not isinstance(decoded, dict):
            raise ValueError(
                "Image labels map must contain "
                "a JSON object."
            )

        return decoded

    def save_training_metadata(
        self,
        *,
        classes: int,
        samples: int,
        accuracy: float,
        valid_samples: int,
        skipped_samples: int,
        training_time_seconds: float,
    ) -> ImageTrainingResult:
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Cannot save metadata because "
                "the image model does not exist."
            )

        model_size_mb = (
            self.model_path.stat().st_size
            / (1024 * 1024)
        )

        metadata = self.metadata_manager.save(
            classes=classes,
            samples=samples,
            accuracy=accuracy,
            model_path=str(self.model_path),
            valid_samples=valid_samples,
            skipped_samples=skipped_samples,
            training_time_seconds=(
                training_time_seconds
            ),
            model_size_mb=model_size_mb,
        )

        return ImageTrainingResult.from_dict(
            metadata,
        )

    def load_training_metadata(
        self,
    ) -> ImageTrainingResult | None:
        metadata = self.metadata_manager.read()

        if metadata is None:
            return None

        return ImageTrainingResult.from_dict(
            metadata,
        )

    def delete_all(self) -> dict[str, bool]:
        model_deleted = FileManager.delete_file(
            self.model_path,
        )

        labels_deleted = FileManager.delete_file(
            self.labels_path,
        )

        metadata_deleted = (
            self.metadata_manager.delete()
        )

        return {
            "model_deleted": model_deleted,
            "labels_deleted": labels_deleted,
            "metadata_deleted": metadata_deleted,
        }