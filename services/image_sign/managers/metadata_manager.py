import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from shared.file_manager import FileManager
from shared.paths import AppPaths, ImagePaths


class ImageMetadataManager:
    def __init__(self) -> None:
        FileManager.ensure_dir(AppPaths.models_dir)
        self.metadata_path = ImagePaths.metadata_path

    def read(self) -> dict[str, Any] | None:
        if not self.metadata_path.exists():
            return None

        content = self.metadata_path.read_text(
            encoding="utf-8",
        ).strip()

        if not content:
            return None

        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid image metadata file: {error}"
            ) from error

        if not isinstance(decoded, dict):
            raise ValueError(
                "Image metadata must contain a JSON object."
            )

        return decoded

    def save(
        self,
        *,
        classes: int,
        samples: int,
        accuracy: float,
        model_path: str,
        valid_samples: int,
        skipped_samples: int,
        training_time_seconds: float,
        model_size_mb: float,
    ) -> dict[str, Any]:
        if classes < 1:
            raise ValueError("Classes count must be greater than zero.")

        if samples < 1:
            raise ValueError("Samples count must be greater than zero.")

        if valid_samples < 0 or skipped_samples < 0:
            raise ValueError("Sample counts cannot be negative.")

        if not 0 <= accuracy <= 100:
            raise ValueError("Accuracy must be between 0 and 100.")

        if training_time_seconds < 0:
            raise ValueError("Training time cannot be negative.")

        if model_size_mb < 0:
            raise ValueError("Model size cannot be negative.")

        if not model_path.strip():
            raise ValueError("Model path is required.")
        
        current_metadata = self.read()

        metadata: dict[str, Any] = {
            "model_id": str(uuid4()),
            "model_status": "ready",
            "model_version": self._next_model_version(
                current_metadata,
            ),
            "dataset_version": self._next_dataset_version(
                current_metadata,
            ),
            "classes": classes,
            "samples": samples,
            "valid_samples": valid_samples,
            "skipped_samples": skipped_samples,
            "accuracy": round(accuracy, 2),
            "training_time_seconds": round(
                training_time_seconds,
                2,
            ),
            "model_size_mb": round(model_size_mb, 2),
            "model_path": model_path,
            "trained_at": datetime.now(
                timezone.utc,
            ).isoformat(),
        }

        FileManager.ensure_dir(
            self.metadata_path.parent,
        )

        self.metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return metadata

    def delete(self) -> bool:
        return FileManager.delete_file(
            self.metadata_path,
        )

    def exists(self) -> bool:
        return self.metadata_path.exists()

    def get_model_status(self) -> str:
        metadata = self.read()

        if metadata is None:
            return "not_trained"

        return str(
            metadata.get(
                "model_status",
                "unknown",
            )
        )

    def _next_model_version(
        self,
        current_metadata: dict[str, Any] | None,
    ) -> str:
        if current_metadata is None:
            return "v1.0"

        current_version = str(
            current_metadata.get(
                "model_version",
                "v1.0",
            )
        )

        try:
            version_text = current_version.removeprefix("v")
            major_text, minor_text = version_text.split(".")

            major = int(major_text)
            minor = int(minor_text) + 1

            return f"v{major}.{minor}"

        except (
            ValueError,
            AttributeError,
        ):
            return "v1.0"

    def _next_dataset_version(
        self,
        current_metadata: dict[str, Any] | None,
    ) -> int:
        if current_metadata is None:
            return 1

        current_version = current_metadata.get(
            "dataset_version",
            0,
        )

        try:
            return int(current_version) + 1
        except (
            TypeError,
            ValueError,
        ):
            return 1