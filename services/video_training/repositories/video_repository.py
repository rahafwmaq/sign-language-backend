import json
from datetime import datetime, timezone
from typing import Any

from shared.file_manager import FileManager
from shared.paths import VideoPaths


class VideoRepository:
    # ---------------- Dataset ----------------

    def load_dataset(self) -> dict[str, list[dict[str, Any]]]:
        labels_file = VideoPaths.dataset_dir / "labels.json"

        if not labels_file.exists():
            return {
                "labels": [],
            }

        try:
            decoded = json.loads(
                labels_file.read_text(
                    encoding="utf-8",
                ),
            )
        except (OSError, json.JSONDecodeError):
            return {
                "labels": [],
            }

        if not isinstance(decoded, list):
            return {
                "labels": [],
            }

        labels = [
            item
            for item in decoded
            if isinstance(item, dict)
        ]

        return {
            "labels": labels,
        }

    # ---------------- Metadata ----------------

    def load_metadata(self) -> dict[str, Any]:
        default_metadata = self._default_metadata()

        if not VideoPaths.metadata_path.exists():
            return default_metadata

        try:
            decoded = json.loads(
                VideoPaths.metadata_path.read_text(
                    encoding="utf-8",
                ),
            )
        except (OSError, json.JSONDecodeError):
            return default_metadata

        if not isinstance(decoded, dict):
            return default_metadata

        return {
            **default_metadata,
            **decoded,
        }

    def save_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        FileManager.ensure_dir(
            VideoPaths.metadata_path.parent,
        )

        metadata_to_save = {
            **metadata,
            "trained_at": datetime.now(
                timezone.utc,
            ).isoformat(),
        }

        VideoPaths.metadata_path.write_text(
            json.dumps(
                metadata_to_save,
                ensure_ascii=False,
                indent=4,
            ),
            encoding="utf-8",
        )

    # ---------------- Versions ----------------

    def next_model_version(self) -> str:
        metadata = self.load_metadata()

        current_version = str(
            metadata.get(
                "model_version",
                "v0.0",
            ),
        )

        try:
            version_numbers = current_version.removeprefix(
                "v",
            ).split(".")

            if len(version_numbers) != 2:
                return "v1.0"

            major = int(version_numbers[0])
            minor = int(version_numbers[1])

            return f"v{major}.{minor + 1}"

        except (TypeError, ValueError):
            return "v1.0"

    def next_dataset_version(self) -> int:
        metadata = self.load_metadata()

        current_version = metadata.get(
            "dataset_version",
            0,
        )

        try:
            return int(current_version) + 1
        except (TypeError, ValueError):
            return 1

    # ---------------- Defaults ----------------

    @staticmethod
    def _default_metadata() -> dict[str, Any]:
        return {
            "model_status": "not_trained",
            "model_id": None,
            "model_version": "v0.0",
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