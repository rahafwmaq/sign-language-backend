from pydantic import BaseModel


class DashboardDataModel(BaseModel):
    words: int
    samples: int
    accuracy: float
    trainings: int

    model_status: str
    model_version: str
    latest_model_type: str | None = None
    last_training: str | None = None


class DashboardResponseModel(BaseModel):
    status: str = "success"
    data: DashboardDataModel