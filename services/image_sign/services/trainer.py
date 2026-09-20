import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from image_sign.managers.dataset_manager import ImageDatasetManager
from image_sign.repositories.model_repository import ImageModelRepository
from image_sign.services.feature_extractor import ImageFeatureExtractor


class ImageModelTrainer:
    supported_extensions = {".jpg", ".jpeg", ".png"}

    def __init__(self) -> None:
        self.dataset_manager = ImageDatasetManager()
        self.repository = ImageModelRepository()

    def train(self) -> dict[str, Any]:
        training_start = time.perf_counter()

        labels = self.dataset_manager.read_labels()

        if not labels:
            raise ValueError(
                "No labels found. Please collect image samples first."
            )

        features_list: list[np.ndarray] = []
        targets: list[str] = []
        label_map: dict[str, dict[str, str]] = {}

        skipped_samples = 0
        total_samples = 0

        with ImageFeatureExtractor() as extractor:
            for index, item in enumerate(labels, start=1):
                arabic = str(item.get("arabic", "")).strip()
                english = str(item.get("english", "")).strip()
                folder = str(item.get("folder", "")).strip()

                if not arabic or not english or not folder:
                    continue

                label_map[arabic] = {
                    "id": str(index),
                    "arabic": arabic,
                    "english": english,
                }

                folder_path = (
                    self.dataset_manager.dataset_dir / folder
                )

                if not folder_path.exists():
                    continue

                image_files = sorted(
                    file
                    for file in folder_path.iterdir()
                    if file.is_file()
                    and file.suffix.lower()
                    in self.supported_extensions
                )

                total_samples += len(image_files)

                for image_file in image_files:
                    features = extractor.extract_from_image(
                        str(image_file)
                    )

                    if features is None:
                        skipped_samples += 1
                        continue

                    features_list.append(features)
                    targets.append(arabic)

        valid_samples = len(features_list)

        if valid_samples < 4:
            raise ValueError(
                "Not enough valid images. "
                "Collect at least 4 images where hands are clearly detected."
            )

        unique_labels = sorted(set(targets))

        if len(unique_labels) < 2:
            raise ValueError(
                "At least 2 different labels are required for training."
            )

        label_counts = {
            label: targets.count(label)
            for label in unique_labels
        }

        insufficient_labels = [
            label
            for label, count in label_counts.items()
            if count < 2
        ]

        if insufficient_labels:
            raise ValueError(
                "Each label needs at least 2 valid images. "
                f"Insufficient labels: {insufficient_labels}"
            )

        x = np.asarray(
            features_list,
            dtype=np.float32,
        )
        y = np.asarray(targets)

        test_count = max(
            len(unique_labels),
            round(valid_samples * 0.2),
        )

        if test_count >= valid_samples:
            raise ValueError(
                "The dataset is too small for a reliable train/test split."
            )

        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_count,
            random_state=42,
            stratify=y,
        )

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        accuracy = float(
            accuracy_score(
                y_test,
                predictions,
            )
            * 100
        )

        self.repository.save_model(model)
        self.repository.save_labels_map(label_map)

        training_time_seconds = (
            time.perf_counter() - training_start
        )

        training_result = (
            self.repository.save_training_metadata(
                classes=len(unique_labels),
                samples=total_samples,
                valid_samples=valid_samples,
                skipped_samples=skipped_samples,
                accuracy=accuracy,
                training_time_seconds=training_time_seconds,
            )
        )

        return training_result.to_dict()