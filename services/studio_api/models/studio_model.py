from pydantic import BaseModel


class ModelStatsModel(BaseModel):
    model_type: str

    model_id: str = ""
    model_status: str = "not_trained"
    model_version: str = "v0.0"

    dataset_version: int = 0

    classes: int = 0
    samples: int = 0
    valid_samples: int = 0
    skipped_samples: int = 0

    accuracy: float = 0.0
    trainings: int = 0

    training_time_seconds: float = 0.0
    model_size_mb: float = 0.0

    model_path: str = ""
    trained_at: str | None = None


class AIStudioDataModel(BaseModel):
    image: ModelStatsModel
    video: ModelStatsModel


class AIStudioResponseModel(BaseModel):
    status: str = "success"
    data: AIStudioDataModel