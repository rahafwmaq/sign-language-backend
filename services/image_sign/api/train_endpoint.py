import json
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from image_sign.services.trainer import ImageModelTrainer
from shared.file_manager import FileManager
from shared.paths import AppPaths, ImagePaths
from shared.zip_manager import ZipManager


router = APIRouter()

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


@router.post("/train")
async def train_image_model(
    file: UploadFile = File(...),
) -> dict:
    filename = file.filename or "image_dataset.zip"

    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded image dataset must be a ZIP file.",
        )

    FileManager.ensure_dir(AppPaths.uploads_dir)

    request_id = uuid4().hex

    zip_path = (
        AppPaths.uploads_dir
        / f"image_dataset_{request_id}.zip"
    )

    temp_dataset_dir = (
        AppPaths.uploads_dir
        / f"image_dataset_temp_{request_id}"
    )

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

        uploaded_labels_file = (
            temp_dataset_dir / "labels.json"
        )

        if not uploaded_labels_file.exists():
            raise ValueError(
                "labels.json must exist at the root "
                "of the uploaded ZIP file."
            )

        merged_labels = merge_image_dataset(
            source_dir=temp_dataset_dir,
            destination_dir=ImagePaths.dataset_dir,
        )

        active_labels = [
            label
            for label in merged_labels
            if int(label.get("samples", 0)) > 0
        ]

        total_samples = sum(
            int(label.get("samples", 0))
            for label in active_labels
        )

        # RandomForest cannot train with one class.
        # Save the dataset and wait for another label.
        if len(active_labels) < 2:
            return {
                "status": "dataset_saved",
                "message": (
                    "Dataset saved successfully. "
                    "Add another label to start model training."
                ),
                "result": {
                    "model_status": (
                        "waiting_for_more_labels"
                    ),
                    "classes": len(active_labels),
                    "samples": total_samples,
                    "valid_samples": total_samples,
                    "skipped_samples": 0,
                    "accuracy": None,
                    "training_time_seconds": 0.0,
                    "model_version": "v0.0",
                    "model_path": None,
                    "trained_at": None,
                },
            }

        trainer = ImageModelTrainer()
        result = trainer.train()

        return {
            "status": "success",
            "message": (
                "Image dataset updated and model "
                "trained successfully."
            ),
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
            detail=f"Image training failed: {error}",
        ) from error

    finally:
        await file.close()

        FileManager.delete_file(zip_path)

        if temp_dataset_dir.exists():
            shutil.rmtree(
                temp_dataset_dir,
                ignore_errors=True,
            )


def merge_image_dataset(
    source_dir: Path,
    destination_dir: Path,
) -> list[dict]:
    """
    Add new images to the existing dataset.

    Existing label folders are NOT deleted. New images are
    appended using unique sequential filenames.
    """
    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    uploaded_labels = _read_labels(
        source_dir / "labels.json",
    )

    existing_labels = _read_labels(
        destination_dir / "labels.json",
        allow_missing=True,
    )

    labels_by_folder = {
        str(item["folder"]): item
        for item in existing_labels
        if isinstance(item, dict)
        and item.get("folder")
    }

    for uploaded_label in uploaded_labels:
        folder_name = str(
            uploaded_label.get("folder", ""),
        ).strip()

        arabic = str(
            uploaded_label.get("arabic", ""),
        ).strip()

        english = str(
            uploaded_label.get("english", ""),
        ).strip()

        _validate_folder_name(folder_name)

        if not arabic or not english:
            raise ValueError(
                "Every label requires Arabic and English values."
            )

        source_label_dir = source_dir / folder_name

        if not source_label_dir.is_dir():
            raise ValueError(
                f"Image folder '{folder_name}' does not "
                "exist in the uploaded ZIP."
            )

        destination_label_dir = (
            destination_dir / folder_name
        )

        destination_label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing_count = len(
            _get_image_files(
                destination_label_dir,
            )
        )

        uploaded_files = _get_image_files(
            source_label_dir,
        )

        if not uploaded_files:
            raise ValueError(
                f"No supported images were found "
                f"inside '{folder_name}'."
            )

        for offset, source_file in enumerate(
            uploaded_files,
            start=1,
        ):
            sample_number = existing_count + offset

            destination_file = (
                destination_label_dir
                / (
                    f"{folder_name}_"
                    f"{sample_number:04d}"
                    f"{source_file.suffix.lower()}"
                )
            )

            # Avoid an unlikely filename collision.
            while destination_file.exists():
                sample_number += 1

                destination_file = (
                    destination_label_dir
                    / (
                        f"{folder_name}_"
                        f"{sample_number:04d}"
                        f"{source_file.suffix.lower()}"
                    )
                )

            shutil.copy2(
                source_file,
                destination_file,
            )

        total_folder_samples = len(
            _get_image_files(
                destination_label_dir,
            )
        )

        labels_by_folder[folder_name] = {
            "arabic": arabic,
            "english": english,
            "folder": folder_name,
            "samples": total_folder_samples,
        }

    merged_labels = sorted(
        labels_by_folder.values(),
        key=lambda item: str(item["folder"]),
    )

    labels_path = destination_dir / "labels.json"

    labels_path.write_text(
        json.dumps(
            merged_labels,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return merged_labels


def _get_image_files(
    directory: Path,
) -> list[Path]:
    return sorted(
        file_path
        for file_path in directory.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_IMAGE_EXTENSIONS
        )
    )


def _validate_folder_name(
    folder_name: str,
) -> None:
    if not folder_name:
        raise ValueError(
            "Every label requires a valid folder name."
        )

    if (
        folder_name in {".", ".."}
        or "/" in folder_name
        or "\\" in folder_name
    ):
        raise ValueError(
            f"Invalid folder name: '{folder_name}'."
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
            "labels.json does not contain valid labels."
        )

    return labels