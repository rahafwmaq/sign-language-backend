from datetime import datetime, timezone

from pydantic import BaseModel


class VideoTrainingMetadataModel(BaseModel):
    model_id: str | None = None
    model_status: str = "not_trained"
    model_version: str = "v0.0"

    dataset_version: int = 0

    classes: int = 0
    samples: int = 0
    valid_samples: int = 0
    skipped_samples: int = 0

    accuracy: float | None = None
    training_time_seconds: float | None = None
    model_size_mb: float | None = None

    model_path: str | None = None
    trained_at: str | None = None

    @classmethod
    def empty(cls) -> "VideoTrainingMetadataModel":
        return cls()

    def mark_trained(
        self,
        *,
        model_id: str,
        model_version: str,
        dataset_version: int,
        classes: int,
        samples: int,
        valid_samples: int,
        skipped_samples: int,
        accuracy: float,
        training_time_seconds: float,
        model_size_mb: float,
        model_path: str,
    ) -> "VideoTrainingMetadataModel":
        return self.model_copy(
            update={
                "model_id": model_id,
                "model_status": "ready",
                "model_version": model_version,
                "dataset_version": dataset_version,
                "classes": classes,
                "samples": samples,
                "valid_samples": valid_samples,
                "skipped_samples": skipped_samples,
                "accuracy": accuracy,
                "training_time_seconds": training_time_seconds,
                "model_size_mb": model_size_mb,
                "model_path": model_path,
                "trained_at": datetime.now(
                    timezone.utc,
                ).isoformat(),
            },
        )