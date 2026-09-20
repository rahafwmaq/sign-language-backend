from fastapi import APIRouter, HTTPException

from image_sign.managers.dataset_manager import ImageDatasetManager
from image_sign.repositories.model_repository import ImageModelRepository


router = APIRouter()


@router.get("/stats")
def get_image_stats() -> dict:
    try:
        dataset_manager = ImageDatasetManager()
        repository = ImageModelRepository()

        dataset_stats = dataset_manager.get_stats()
        training_result = repository.load_training_metadata()

        return {
            "dataset": dataset_stats,
            "model": (
                training_result.to_dict()
                if training_result is not None
                else {
                    "model_status": "not_trained",
                    "model_id": None,
                    "model_version": None,
                    "dataset_version": 0,
                    "classes": 0,
                    "samples": 0,
                    "valid_samples": 0,
                    "skipped_samples": 0,
                    "accuracy": None,
                    "training_time_seconds": None,
                    "model_size_mb": None,
                    "model_path": None,
                    "trained_at": None,
                }
            ),
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read image statistics: {error}",
        ) from error