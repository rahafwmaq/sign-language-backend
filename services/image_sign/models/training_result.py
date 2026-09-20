from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ImageTrainingResult:
    model_id: str
    model_status: str
    model_version: str
    dataset_version: int

    classes: int
    samples: int
    valid_samples: int
    skipped_samples: int

    accuracy: float
    training_time_seconds: float
    model_size_mb: float

    model_path: str
    trained_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "ImageTrainingResult":
        
        try:
            return cls(
                model_id=str(data.get("model_id", "")),
                model_status=str(
                    data.get("model_status", "unknown")
                ),
                model_version=str(
                    data.get("model_version", "v1.0")
                ),
                dataset_version=int(
                    data.get("dataset_version", 1)
                ),
                classes=int(data.get("classes", 0)),
                samples=int(data.get("samples", 0)),
                valid_samples=int(
                    data.get("valid_samples", 0)
                ),
                skipped_samples=int(
                    data.get("skipped_samples", 0)
                ),
                accuracy=float(
                    data.get("accuracy", 0.0)
                ),
                training_time_seconds=float(
                    data.get(
                        "training_time_seconds",
                        0.0,
                    )
                ),
                model_size_mb=float(
                    data.get("model_size_mb", 0.0)
                ),
                model_path=str(
                    data.get("model_path", "")
                ),
                trained_at=str(
                    data.get("trained_at", "")
                ),
            )
        
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid training result: {error}"
        ) from error

    @property
    def is_ready(self) -> bool:
        return self.model_status.lower() == "ready"
    
    @property
    def has_accuracy(self) -> bool:
        return 0 < self.accuracy <= 100