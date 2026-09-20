import json
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from shared.file_manager import FileManager
from shared.paths import AppPaths, VideoPaths
from shared.zip_manager import ZipManager
from video_training.managers.video_manager import VideoManager


router = APIRouter()


@router.post("/train")
async def train_video_model(
    file: UploadFile = File(...),
) -> dict:
    filename = file.filename or "video_dataset.zip"

    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded video dataset must be a ZIP file.",
        )

    FileManager.ensure_dir(AppPaths.uploads_dir)

    zip_path = AppPaths.uploads_dir / "video_dataset.zip"
    temp_dataset_dir = AppPaths.uploads_dir / "video_dataset_temp"

    try:
        ZipManager.save_upload_file(
            upload_file=file,
            destination=zip_path,
        )

        ZipManager.extract(
            zip_path=zip_path,
            destination=temp_dataset_dir,
            clean_before_extract=True,
        )

        uploaded_labels_file = temp_dataset_dir / "labels.json"

        if not uploaded_labels_file.exists():
            raise ValueError(
                "labels.json must exist at the root of the uploaded ZIP file."
            )

        merge_video_dataset(
            source_dir=temp_dataset_dir,
            destination_dir=VideoPaths.dataset_dir,
        )

        merged_labels_file = VideoPaths.dataset_dir / "labels.json"

        if not merged_labels_file.exists():
            raise ValueError(
                "Unable to create the merged labels.json file."
            )

        manager = VideoManager()

        result = manager.train()

        return {
            "status": "success",
            "message": "Video dataset merged and model trained successfully.",
            "result": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Video training failed: {error}",
        ) from error

    finally:
        await file.close()

        FileManager.delete_file(zip_path)

        if temp_dataset_dir.exists():
            shutil.rmtree(
                temp_dataset_dir,
                ignore_errors=True,
            )


def merge_video_dataset(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_labels = _read_labels(
        source_dir / "labels.json",
    )

    destination_labels = _read_labels(
        destination_dir / "labels.json",
        allow_missing=True,
    )

    labels_by_folder = {
        str(item["folder"]): item
        for item in destination_labels
        if isinstance(item, dict) and item.get("folder")
    }

    for label in source_labels:
        if not isinstance(label, dict):
            continue

        folder_name = str(
            label.get("folder", ""),
        ).strip()

        if not folder_name:
            raise ValueError(
                "Every label must contain a valid folder name."
            )

        if (
            folder_name in {".", ".."}
            or "/" in folder_name
            or "\\" in folder_name
        ):
            raise ValueError(
                f"Invalid folder name: {folder_name}"
            )

        source_folder = source_dir / folder_name
        destination_folder = destination_dir / folder_name

        if not source_folder.is_dir():
            raise ValueError(
                f"Video folder '{folder_name}' does not exist in the uploaded ZIP."
            )

        if destination_folder.exists():
            shutil.rmtree(destination_folder)

        shutil.copytree(
            source_folder,
            destination_folder,
        )

        labels_by_folder[folder_name] = label

    merged_labels = sorted(
        labels_by_folder.values(),
        key=lambda item: item["folder"],
    )

    (destination_dir / "labels.json").write_text(
        json.dumps(
            merged_labels,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_labels(
    file_path: Path,
    allow_missing: bool = False,
) -> list[dict]:
    if not file_path.exists():
        if allow_missing:
            return []

        raise ValueError(
            f"labels.json was not found: {file_path}"
        )

    content = file_path.read_text(
        encoding="utf-8",
    ).strip()

    if not content:
        if allow_missing:
            return []

        raise ValueError(
            "labels.json is empty."
        )

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "labels.json contains invalid JSON."
        ) from error

    if not isinstance(decoded, list):
        raise ValueError(
            "labels.json must contain a JSON list."
        )

    labels = [
        item
        for item in decoded
        if isinstance(item, dict)
    ]

    if not labels and not allow_missing:
        raise ValueError(
            "labels.json does not contain any valid labels."
        )

    return labels