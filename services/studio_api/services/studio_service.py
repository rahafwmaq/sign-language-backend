from datetime import datetime
from typing import Any

from studio_api.models.dashboard_model import (
    DashboardDataModel,
    DashboardResponseModel,
)
from studio_api.models.studio_model import (
    AIStudioDataModel,
    AIStudioResponseModel,
    ModelStatsModel,
)
from studio_api.repositories.studio_repository import (
    StudioRepository,
)

class StudioService:
    def __init__(
        self,
        repository: StudioRepository | None = None,
    ) -> None:
        self.repository = repository or StudioRepository()

    def get_dashboard(self) -> DashboardResponseModel:
        image_stats = self._get_image_stats()
        video_stats = self._get_video_stats()

        total_words = (
            image_stats.classes
            + video_stats.classes
        )

        total_samples = (
            image_stats.samples
            + video_stats.samples
        )

        total_trainings = (
            image_stats.trainings
            + video_stats.trainings
        )

        overall_accuracy = self._calculate_accuracy(
            image_stats=image_stats,
            video_stats=video_stats,
        )

        latest_model = self._get_latest_model(
            image_stats=image_stats,
            video_stats=video_stats,
        )

        return DashboardResponseModel(
            data=DashboardDataModel(
                words=total_words,
                samples=total_samples,
                accuracy=overall_accuracy,
                trainings=total_trainings,
                model_status=latest_model.model_status,
                model_version=latest_model.model_version,
                latest_model_type=latest_model.model_type,
                last_training=latest_model.trained_at,
            ),
        )

    def get_models(self) -> AIStudioResponseModel:
        return AIStudioResponseModel(
            data=AIStudioDataModel(
                image=self._get_image_stats(),
                video=self._get_video_stats(),
            ),
        )

    def _get_image_stats(self) -> ModelStatsModel:
        metadata = self.repository.get_image_metadata()

        return self._normalise_metadata(
            metadata=metadata,
            model_type="image",
        )

    def _get_video_stats(self) -> ModelStatsModel:
        metadata = self.repository.get_video_metadata()

        return self._normalise_metadata(
            metadata=metadata,
            model_type="video",
        )

    def _normalise_metadata(
        self,
        metadata: dict[str, Any],
        model_type: str,
    ) -> ModelStatsModel:
        samples = self._to_int(
            metadata.get("samples"),
        )

        return ModelStatsModel(
            model_type=model_type,
            model_id=str(
                metadata.get("model_id", ""),
            ),
            model_status=str(
                metadata.get(
                    "model_status",
                    metadata.get("status", "not_trained"),
                ),
            ),
            model_version=str(
                metadata.get("model_version", "v0.0"),
            ),
            dataset_version=self._to_int(
                metadata.get("dataset_version"),
            ),
            classes=self._to_int(
                metadata.get(
                    "classes",
                    metadata.get("words", 0),
                ),
            ),
            samples=samples,
            valid_samples=self._to_int(
                metadata.get(
                    "valid_samples",
                    samples,
                ),
            ),
            skipped_samples=self._to_int(
                metadata.get("skipped_samples"),
            ),
            accuracy=self._to_float(
                metadata.get("accuracy"),
            ),
            trainings=self._to_int(
                metadata.get(
                    "trainings",
                    metadata.get(
                        "training_count",
                        0,
                    ),
                ),
            ),
            training_time_seconds=self._to_float(
                metadata.get(
                    "training_time_seconds",
                    metadata.get(
                        "training_time",
                        0.0,
                    ),
                ),
            ),
            model_size_mb=self._to_float(
                metadata.get("model_size_mb"),
            ),
            model_path=str(
                metadata.get("model_path", ""),
            ),
            trained_at=self._read_training_date(
                metadata,
            ),
        )

    def _calculate_accuracy(
        self,
        image_stats: ModelStatsModel,
        video_stats: ModelStatsModel,
    ) -> float:
        trained_models = [
            model
            for model in [
                image_stats,
                video_stats,
            ]
            if model.model_status == "ready"
            and model.valid_samples > 0
        ]

        if not trained_models:
            return 0.0

        total_valid_samples = sum(
            model.valid_samples
            for model in trained_models
        )

        if total_valid_samples == 0:
            return 0.0

        weighted_accuracy = sum(
            model.accuracy * model.valid_samples
            for model in trained_models
        ) / total_valid_samples

        return round(weighted_accuracy, 2)

    def _get_latest_model(
        self,
        image_stats: ModelStatsModel,
        video_stats: ModelStatsModel,
    ) -> ModelStatsModel:
        models = [
            image_stats,
            video_stats,
        ]

        trained_models = [
            model
            for model in models
            if model.trained_at
        ]

        if not trained_models:
            return ModelStatsModel(
                model_type="none",
            )

        return max(
            trained_models,
            key=lambda model: self._parse_date(
                model.trained_at,
            ),
        )

    @staticmethod
    def _read_training_date(
        metadata: dict[str, Any],
    ) -> str | None:
        value = metadata.get(
            "trained_at",
            metadata.get("last_training"),
        )

        if value is None:
            return None

        text = str(value).strip()

        return text or None

    @staticmethod
    def _parse_date(
        value: str | None,
    ) -> datetime:
        if not value:
            return datetime.min

        try:
            return datetime.fromisoformat(
                value.replace("Z", "+00:00"),
            )
        except ValueError:
            return datetime.min

    @staticmethod
    def _to_int(value: Any) -> int:
        if value is None:
            return 0

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return 0.0

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0