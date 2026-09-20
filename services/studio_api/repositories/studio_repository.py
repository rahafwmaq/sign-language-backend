import json
from pathlib import Path
from typing import Any

from shared.paths import ImagePaths, VideoPaths


class StudioRepository:
    # =========================================================
    # Public
    # =========================================================

    def get_dashboard_data(self) -> dict[str, Any]:
        image = self._get_model_summary(
            model_type="image",
            dataset_dir=ImagePaths.dataset_dir,
            metadata_path=ImagePaths.metadata_path,
        )

        video = self._get_model_summary(
            model_type="video",
            dataset_dir=VideoPaths.dataset_dir,
            metadata_path=VideoPaths.metadata_path,
        )

        total_labels = (
            image["labels"]
            + video["labels"]
        )

        total_samples = (
            image["samples"]
            + video["samples"]
        )

        total_trainings = (
            image["trainings"]
            + video["trainings"]
        )

        accuracies = [
            value
            for value in [
                image["accuracy"],
                video["accuracy"],
            ]
            if value is not None
        ]

        overall_accuracy = (
            round(
                sum(accuracies) / len(accuracies),
                2,
            )
            if accuracies
            else 0.0
        )

        latest_model = self._get_latest_model(
            image=image,
            video=video,
        )

        return {
            "image": image,
            "video": video,
            "overall": {
                "labels": total_labels,
                "samples": total_samples,
                "trainings": total_trainings,
                "accuracy": overall_accuracy,
                "latest_model_type": latest_model[
                    "model_type"
                ],
                "model_status": latest_model[
                    "model_status"
                ],
                "model_version": latest_model[
                    "model_version"
                ],
                "last_training": latest_model[
                    "last_training"
                ],
            },
        }

    def get_models_data(self) -> dict[str, Any]:
        image_labels = self._read_labels(
            ImagePaths.dataset_dir / "labels.json",
        )

        video_labels = self._read_labels(
            VideoPaths.dataset_dir / "labels.json",
        )

        image_summary = self._get_model_summary(
            model_type="image",
            dataset_dir=ImagePaths.dataset_dir,
            metadata_path=ImagePaths.metadata_path,
        )

        video_summary = self._get_model_summary(
            model_type="video",
            dataset_dir=VideoPaths.dataset_dir,
            metadata_path=VideoPaths.metadata_path,
        )

        return {
            "image": {
                "summary": image_summary,
                "dataset": image_labels,
            },
            "video": {
                "summary": video_summary,
                "dataset": video_labels,
            },
            "overall": {
                "labels": (
                    image_summary["labels"]
                    + video_summary["labels"]
                ),
                "samples": (
                    image_summary["samples"]
                    + video_summary["samples"]
                ),
                "trainings": (
                    image_summary["trainings"]
                    + video_summary["trainings"]
                ),
            },
        }

    # =========================================================
    # Helpers
    # =========================================================

    def _get_model_summary(
        self,
        *,
        model_type: str,
        dataset_dir: Path,
        metadata_path: Path,
    ) -> dict[str, Any]:
        labels = self._read_labels(
            dataset_dir / "labels.json",
        )

        metadata = self._read_json_object(
            metadata_path,
        )

        samples = sum(
            int(item.get("samples", 0))
            for item in labels
            if isinstance(item, dict)
        )

        return {
            "model_type": model_type,
            "labels": len(labels),
            "samples": samples,
            "accuracy": self._to_float_or_none(
                metadata.get("accuracy"),
            ),
            "trainings": self._get_training_count(
                metadata,
            ),
            "model_status": metadata.get(
                "model_status",
                "not_trained",
            ),
            "model_version": metadata.get(
                "model_version",
                "v0.0",
            ),
            "dataset_version": self._to_int(
                metadata.get("dataset_version"),
            ),
            "valid_samples": self._to_int(
                metadata.get("valid_samples"),
            ),
            "skipped_samples": self._to_int(
                metadata.get("skipped_samples"),
            ),
            "training_time_seconds": (
                self._to_float_or_none(
                    metadata.get(
                        "training_time_seconds",
                    )
                )
            ),
            "model_size_mb": self._to_float_or_none(
                metadata.get("model_size_mb"),
            ),
            "last_training": metadata.get(
                "trained_at",
            ),
        }

    def _get_training_count(
        self,
        metadata: dict[str, Any],
    ) -> int:
        explicit_trainings = metadata.get(
            "trainings",
        )

        if explicit_trainings is not None:
            return self._to_int(
                explicit_trainings,
            )

        # عندك dataset_version يزيد مع كل تدريب ناجح.
        return self._to_int(
            metadata.get("dataset_version"),
        )

    def _get_latest_model(
        self,
        *,
        image: dict[str, Any],
        video: dict[str, Any],
    ) -> dict[str, Any]:
        image_date = image.get(
            "last_training",
        )

        video_date = video.get(
            "last_training",
        )

        if image_date and video_date:
            if str(image_date) >= str(video_date):
                return image

            return video

        if image_date:
            return image

        if video_date:
            return video

        return {
            "model_type": "none",
            "model_status": "not_trained",
            "model_version": "v0.0",
            "last_training": None,
        }

    def _read_labels(
        self,
        labels_path: Path,
    ) -> list[dict[str, Any]]:
        if not labels_path.exists():
            return []

        try:
            decoded = json.loads(
                labels_path.read_text(
                    encoding="utf-8",
                ),
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return []

        if not isinstance(decoded, list):
            return []

        return [
            item
            for item in decoded
            if isinstance(item, dict)
        ]

    def _read_json_object(
        self,
        file_path: Path,
    ) -> dict[str, Any]:
        if not file_path.exists():
            return {}

        try:
            decoded = json.loads(
                file_path.read_text(
                    encoding="utf-8",
                ),
            )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            return {}

        return (
            decoded
            if isinstance(decoded, dict)
            else {}
        )

    @staticmethod
    def _to_int(
        value: Any,
    ) -> int:
        try:
            return int(value or 0)
        except (
            TypeError,
            ValueError,
        ):
            return 0

    @staticmethod
    def _to_float_or_none(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return None