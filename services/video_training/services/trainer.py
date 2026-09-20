import json
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from shared.file_manager import FileManager
from shared.paths import VideoPaths
from video_training.repositories.video_repository import (
    VideoRepository,
)
from video_training.services.feature_extractor import (
    VideoFeatureExtractor,
)


ALLOWED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}


class VideoModelTrainer:
    def __init__(
        self,
        *,
        sequence_length: int = 30,
        max_num_hands: int = 2,
    ) -> None:
        self.repository = VideoRepository()

        self.feature_extractor = VideoFeatureExtractor(
            sequence_length=sequence_length,
            max_num_hands=max_num_hands,
        )

    def train(self) -> dict[str, Any]:
        started_at = time.perf_counter()

        dataset = self.repository.load_dataset()
        labels = dataset.get("labels", [])

        # Allow training with one label only.
        if not labels:
            raise ValueError(
                "At least one video label is required for training."
            )

        features: list[np.ndarray] = []
        targets: list[str] = []

        total_samples = 0
        valid_samples = 0
        skipped_samples = 0

        label_details: dict[str, dict[str, str]] = {}

        try:
            for label in labels:
                if not isinstance(label, dict):
                    continue

                folder_name = str(
                    label.get("folder", ""),
                ).strip()

                arabic_label = str(
                    label.get("arabic", ""),
                ).strip()

                english_label = str(
                    label.get("english", ""),
                ).strip()

                if not folder_name:
                    continue

                label_folder = (
                    VideoPaths.dataset_dir / folder_name
                )

                if not label_folder.is_dir():
                    continue

                video_files = self._get_video_files(
                    label_folder,
                )

                total_samples += len(video_files)

                if not video_files:
                    continue

                label_details[folder_name] = {
                    "arabic": arabic_label,
                    "english": english_label,
                    "folder": folder_name,
                }

                for video_path in video_files:
                    try:
                        extracted_features = (
                            self.feature_extractor.extract(
                                video_path,
                            )
                        )

                        if extracted_features is None:
                            skipped_samples += 1
                            continue

                        if (
                            extracted_features.ndim != 1
                            or extracted_features.size == 0
                        ):
                            skipped_samples += 1
                            continue

                        features.append(
                            extracted_features.astype(
                                np.float32,
                            ),
                        )

                        targets.append(folder_name)
                        valid_samples += 1

                    except (
                        FileNotFoundError,
                        ValueError,
                        OSError,
                    ) as error:
                        skipped_samples += 1

                        print(
                            "Skipped video "
                            f"'{video_path.name}': {error}"
                        )

            if total_samples == 0:
                raise ValueError(
                    "No video files were found in the dataset."
                )

            if not features:
                raise ValueError(
                    "No valid video samples were found for training. "
                    "Make sure hands are visible in the videos."
                )

            self._validate_feature_sizes(features)

            trained_classes = sorted(
                set(targets),
            )

            # One valid label is enough.
            if not trained_classes:
                raise ValueError(
                    "No valid video label was found for training."
                )

            class_counts = Counter(targets)

            # Do not fail training because an old label
            # has only one valid sample.
            for label, count in class_counts.items():
                print(
                    f"Video label '{label}' has "
                    f"{count} valid sample(s)."
                )

            x_data = np.stack(
                features,
            ).astype(np.float32)

            y_data = np.asarray(
                targets,
                dtype=str,
            )

            model = RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )

            accuracy = self._fit_and_evaluate(
                model=model,
                x_data=x_data,
                y_data=y_data,
            )

            FileManager.ensure_dir(
                VideoPaths.model_path.parent,
            )

            joblib.dump(
                model,
                VideoPaths.model_path,
            )

            if not VideoPaths.model_path.exists():
                raise OSError(
                    "The trained video model could not be saved."
                )

            labels_map = self._create_labels_map(
                trained_classes=trained_classes,
                label_details=label_details,
            )

            VideoPaths.labels_path.write_text(
                json.dumps(
                    labels_map,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            if not VideoPaths.labels_path.exists():
                raise OSError(
                    "The video labels map could not be saved."
                )

            training_time_seconds = round(
                time.perf_counter() - started_at,
                2,
            )

            model_size_mb = round(
                VideoPaths.model_path.stat().st_size
                / (1024 * 1024),
                2,
            )

            metadata = {
                "model_id": str(uuid.uuid4()),
                "model_status": "ready",
                "model_version": (
                    self.repository.next_model_version()
                ),
                "dataset_version": (
                    self.repository.next_dataset_version()
                ),
                "classes": len(trained_classes),
                "samples": total_samples,
                "valid_samples": valid_samples,
                "skipped_samples": skipped_samples,
                "accuracy": round(accuracy, 2),
                "training_time_seconds": (
                    training_time_seconds
                ),
                "model_size_mb": model_size_mb,
                "model_path": str(
                    VideoPaths.model_path.resolve(),
                ),
            }

            self.repository.save_metadata(
                metadata,
            )

            return self.repository.load_metadata()

        finally:
            self.feature_extractor.close()

    def _fit_and_evaluate(
        self,
        *,
        model: RandomForestClassifier,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> float:
        class_counts = Counter(
            y_data.tolist(),
        )

        class_count = len(class_counts)
        sample_count = len(y_data)

        # One-label training:
        # train on all available samples.
        if class_count == 1:
            model.fit(
                x_data,
                y_data,
            )

            # Accuracy is not meaningful for one class.
            return 0.0

        can_split = (
            sample_count >= class_count * 2
            and all(
                count >= 2
                for count in class_counts.values()
            )
        )

        # Some labels may only have one valid sample.
        # In that case, train on the full dataset
        # instead of failing the training.
        if not can_split:
            model.fit(
                x_data,
                y_data,
            )

            return 0.0

        test_size = max(
            class_count,
            int(round(sample_count * 0.2)),
        )

        maximum_test_size = (
            sample_count - class_count
        )

        test_size = min(
            test_size,
            maximum_test_size,
        )

        if test_size < class_count:
            model.fit(
                x_data,
                y_data,
            )

            return 0.0

        x_train, x_test, y_train, y_test = (
            train_test_split(
                x_data,
                y_data,
                test_size=test_size,
                random_state=42,
                stratify=y_data,
            )
        )

        model.fit(
            x_train,
            y_train,
        )

        predictions = model.predict(
            x_test,
        )

        return float(
            accuracy_score(
                y_test,
                predictions,
            )
            * 100
        )

    @staticmethod
    def _validate_feature_sizes(
        features: list[np.ndarray],
    ) -> None:
        feature_sizes = {
            feature.shape[0]
            for feature in features
        }

        if len(feature_sizes) != 1:
            raise ValueError(
                "Extracted video features have inconsistent sizes."
            )

    @staticmethod
    def _get_video_files(
        label_folder: Path,
    ) -> list[Path]:
        return sorted(
            file_path
            for file_path in label_folder.rglob("*")
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in ALLOWED_VIDEO_EXTENSIONS
            )
        )

    @staticmethod
    def _create_labels_map(
        *,
        trained_classes: list[str],
        label_details: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        return {
            folder_name: label_details.get(
                folder_name,
                {
                    "arabic": "",
                    "english": folder_name,
                    "folder": folder_name,
                },
            )
            for folder_name in trained_classes
        }