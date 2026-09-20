from typing import Any

from video_training.repositories.video_repository import VideoRepository
from video_training.services.trainer import VideoModelTrainer


class VideoManager:
    def __init__(self) -> None:
        self.repository = VideoRepository()

    # ---------------- Training ----------------

    def train(self) -> dict[str, Any]:
        trainer = VideoModelTrainer()

        return trainer.train()

    # ---------------- Dataset ----------------

    def get_dataset(self) -> dict[str, Any]:
        return self.repository.load_dataset()

    # ---------------- Metadata ----------------

    def get_metadata(self) -> dict[str, Any]:
        return self.repository.load_metadata()

    # ---------------- Stats ----------------

    def get_stats(self) -> dict[str, Any]:
        dataset = self.get_dataset()
        metadata = self.get_metadata()

        labels = dataset.get("labels", [])

        total_samples = sum(
            int(label.get("samples", 0))
            for label in labels
            if isinstance(label, dict)
        )

        return {
            "dataset": {
                "labels_count": len(labels),
                "samples": total_samples,
                "labels": labels,
            },
            "model": {
                "model_status": metadata.get(
                    "model_status",
                    "not_trained",
                ),
                "model_id": metadata.get("model_id"),
                "model_version": metadata.get(
                    "model_version",
                    "v0.0",
                ),
                "dataset_version": metadata.get(
                    "dataset_version",
                    0,
                ),
                "classes": metadata.get(
                    "classes",
                    0,
                ),
                "samples": metadata.get(
                    "samples",
                    0,
                ),
                "valid_samples": metadata.get(
                    "valid_samples",
                    0,
                ),
                "skipped_samples": metadata.get(
                    "skipped_samples",
                    0,
                ),
                "accuracy": metadata.get("accuracy"),
                "training_time_seconds": metadata.get(
                    "training_time_seconds",
                ),
                "model_size_mb": metadata.get(
                    "model_size_mb",
                ),
                "model_path": metadata.get(
                    "model_path",
                ),
                "trained_at": metadata.get(
                    "trained_at",
                ),
            },
        }