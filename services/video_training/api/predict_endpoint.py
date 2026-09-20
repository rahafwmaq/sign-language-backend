import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from shared.file_manager import FileManager
from shared.paths import AppPaths
from video_training.services.predictor import VideoPredictor


router = APIRouter()


ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


@router.post("/predict")
async def predict_video(
    file: UploadFile = File(...),
) -> dict:

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a valid video file. "
                "Supported formats: MP4, MOV, AVI, MKV, and WEBM."
            ),
        )

    FileManager.ensure_dir(
        AppPaths.uploads_dir,
    )

    temp_video_path = (
        AppPaths.uploads_dir
        / f"video_prediction_{uuid4().hex}{extension}"
    )

    try:

        # =====================================================
        # SAVE TEMP VIDEO
        # =====================================================

        with temp_video_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer,
            )

        print(
            f"VIDEO PREDICT FILE: {temp_video_path}"
        )

        print(
            f"VIDEO PREDICT SIZE: "
            f"{temp_video_path.stat().st_size} bytes"
        )

        # =====================================================
        # CREATE FRESH PREDICTOR
        # =====================================================

        predictor = VideoPredictor()

        # =====================================================
        # PREDICT
        # =====================================================

        prediction = predictor.predict(
            temp_video_path,
        )

        print(
            f"VIDEO PREDICTION RESULT: {prediction}"
        )

        # =====================================================
        # NO SIGN
        # =====================================================

        detected = bool(
            prediction.get(
                "detected",
                True,
            )
        )

        if not detected:
            return {
                "status": "success",
                "message": "No valid sign detected.",
                "prediction": prediction,
            }

        # =====================================================
        # SIGN DETECTED
        # =====================================================

        return {
            "status": "success",
            "message": (
                "Video prediction completed successfully."
            ),
            "prediction": prediction,
        }

    except FileNotFoundError as error:

        print(
            f"VIDEO FILE ERROR: {error}"
        )

        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except ValueError as error:

        print(
            f"VIDEO VALUE ERROR: {error}"
        )

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:

        print(
            f"VIDEO PREDICTION ERROR: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Video prediction failed: {error}"
            ),
        ) from error

    finally:

        await file.close()

        FileManager.delete_file(
            temp_video_path,
        )