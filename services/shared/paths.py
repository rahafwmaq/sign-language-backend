from pathlib import Path


class AppPaths:
    base_dir = Path(__file__).resolve().parent.parent

    uploads_dir = base_dir / "uploads"
    models_dir = base_dir / "models"


class ImagePaths:
    dataset_dir = AppPaths.uploads_dir / "image_dataset_v2"

    model_path = (
        AppPaths.models_dir
        / "image_v2"
        / "image_sign_model.pkl"
    )

    labels_path = (
        AppPaths.models_dir
        / "image_v2"
        / "image_labels_map.json"
    )

    metadata_path = (
        AppPaths.models_dir
        / "image_v2"
        / "image_training_metadata.json"
    )


class VideoPaths:
    dataset_dir = AppPaths.uploads_dir / "video_dataset"

    model_path = AppPaths.models_dir / "video_sign_model.pkl"
    labels_path = AppPaths.models_dir / "video_labels_map.json"
    metadata_path = AppPaths.models_dir / "video_training_metadata.json"