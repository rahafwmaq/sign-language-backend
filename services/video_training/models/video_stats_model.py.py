from pydantic import BaseModel, Field


class VideoDatasetLabelModel(BaseModel):
    arabic: str
    english: str
    folder: str
    samples: int


class VideoDatasetModel(BaseModel):
    labels_count: int = 0
    samples: int = 0

    labels: list[VideoDatasetLabelModel] = Field(
        default_factory=list,
    )


class VideoModelInfo(BaseModel):
    model_status: str = "not_trained"

    model_id: str | None = None

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


class VideoStatsModel(BaseModel):
    dataset: VideoDatasetModel = Field(
        default_factory=VideoDatasetModel,
    )

    model: VideoModelInfo = Field(
        default_factory=VideoModelInfo,
    )