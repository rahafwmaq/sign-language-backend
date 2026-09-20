from fastapi import APIRouter, HTTPException

from image_sign.managers.dataset_manager import ImageDatasetManager
from image_sign.repositories.model_repository import ImageModelRepository
from shared.file_manager import FileManager
from shared.paths import AppPaths


router = APIRouter()


@router.delete("/reset")
def reset_image_data() -> dict:
    try:
        dataset_manager = ImageDatasetManager()
        repository = ImageModelRepository()

        dataset_result = dataset_manager.reset()
        model_result = repository.delete_all()

        FileManager.delete_dir(
            AppPaths.uploads_dir / "predict_images",
        )

        return {
            "status": "success",
            "message": "Image dataset and model files were reset successfully.",
            "dataset": dataset_result,
            "model": model_result,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image reset failed: {error}",
        ) from error