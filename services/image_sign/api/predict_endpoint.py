from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from image_sign.services.predictor import ImageModelPredictor
from shared.file_manager import FileManager
from shared.paths import AppPaths
from shared.zip_manager import ZipManager


router = APIRouter()

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# Create once and reuse it.
predictor = ImageModelPredictor()


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
) -> dict:
    filename = file.filename or "prediction.jpg"
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, and PNG images are supported.",
        )

    predict_dir = FileManager.ensure_dir(
        AppPaths.uploads_dir / "predict_images",
    )

    image_path = (
        predict_dir
        / f"{uuid4()}{extension}"
    )

    try:
        ZipManager.save_upload_file(
            upload_file=file,
            destination=image_path,
        )

        return predictor.predict(
            str(image_path),
        )

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Image prediction failed: {error}",
        ) from error

    finally:
        await file.close()

        FileManager.delete_file(
            image_path,
        )