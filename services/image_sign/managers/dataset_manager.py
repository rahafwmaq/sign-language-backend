import json
from pathlib import Path
from typing import Any

from shared.file_manager import FileManager
from shared.paths import ImagePaths


class ImageDatasetManager:
    labels_file_name = "labels.json"
    supported_extensions = ("jpg", "jpeg", "png")

    def __init__(self) -> None:
        self.dataset_dir = FileManager.ensure_dir(
            ImagePaths.dataset_dir,
        )
        self.labels_file = self.dataset_dir / self.labels_file_name

    def read_labels(self) -> list[dict[str, Any]]:
        if not self.labels_file.exists():
            return []

        content = self.labels_file.read_text(
            encoding="utf-8",
        ).strip()

        if not content:
            return []

        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid labels.json file: {error}"
            ) from error

        if not isinstance(decoded, list):
            raise ValueError(
                "labels.json must contain a JSON list."
            )

        return decoded

    def write_labels(
        self,
        labels: list[dict[str, Any]],
    ) -> None:
        FileManager.ensure_dir(self.labels_file.parent)

        self.labels_file.write_text(
            json.dumps(
                labels,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def get_or_create_label(
        self,
        arabic: str,
        english: str,
        folder: str,
    ) -> dict[str, Any]:
        arabic = arabic.strip()
        english = english.strip()
        folder = self._validate_folder_name(folder)

        if not arabic:
            raise ValueError("Arabic label is required.")

        if not english:
            raise ValueError("English label is required.")

        labels = self.read_labels()

        for item in labels:
            if item.get("folder") == folder:
                return item

        label_folder = FileManager.ensure_dir(
            self.dataset_dir / folder,
        )

        new_label: dict[str, Any] = {
            "arabic": arabic,
            "english": english,
            "folder": label_folder.name,
            "samples": 0,
        }

        labels.append(new_label)
        self.write_labels(labels)

        return new_label

    def update_sample_count(
        self,
        folder: str,
    ) -> int:
        folder = self._validate_folder_name(folder)
        label_folder = self.dataset_dir / folder

        image_files = FileManager.list_files(
            label_folder,
            self.supported_extensions,
            recursive=False,
        )

        sample_count = len(image_files)
        labels = self.read_labels()

        for item in labels:
            if item.get("folder") == folder:
                item["samples"] = sample_count
                self.write_labels(labels)
                return sample_count

        return 0

    def refresh_all_sample_counts(
        self,
    ) -> list[dict[str, Any]]:
        labels = self.read_labels()

        if not labels:
            return []

        changed = False

        for item in labels:
            folder = str(item.get("folder", "")).strip()

            if not folder:
                if item.get("samples", 0) != 0:
                    item["samples"] = 0
                    changed = True
                continue

            folder = self._validate_folder_name(folder)
            label_folder = self.dataset_dir / folder

            image_files = FileManager.list_files(
                label_folder,
                self.supported_extensions,
                recursive=False,
            )

            sample_count = len(image_files)

            if item.get("samples") != sample_count:
                item["samples"] = sample_count
                changed = True

        if changed:
            self.write_labels(labels)

        return labels

    def get_stats(self) -> dict[str, Any]:
        labels = self.refresh_all_sample_counts()

        total_labels = len(labels)
        total_samples = sum(
            int(item.get("samples", 0))
            for item in labels
        )

        return {
            "labels_count": total_labels,
            "samples": total_samples,
            "labels": labels,
        }

    def reset(self) -> dict[str, str]:
        FileManager.delete_dir(
            ImagePaths.dataset_dir,
        )

        self.dataset_dir = FileManager.ensure_dir(
            ImagePaths.dataset_dir,
        )
        self.labels_file = self.dataset_dir / self.labels_file_name

        return {
            "status": "success",
            "message": "Image dataset reset successfully.",
        }

    def _validate_folder_name(self, folder: str) -> str:
        folder = folder.strip()

        if not folder:
            raise ValueError("Folder name is required.")

        folder_path = Path(folder)

        if (
            folder_path.is_absolute()
            or ".." in folder_path.parts
            or len(folder_path.parts) != 1
        ):
            raise ValueError("Invalid folder name.")

        return folder