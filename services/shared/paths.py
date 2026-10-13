from pathlib import Path


class AppPaths:
    base_dir = Path(__file__).resolve().parent.parent

    uploads_dir = base_dir / "uploads"
    models_dir = base_dir / "models"


class ImagePaths:

    # =========================
    # IMAGE V2 (ACTIVE)
    # =========================
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

    # =========================
    # IMAGE V1 (ACTIVE)
    # =========================

    # dataset_dir = AppPaths.uploads_dir / "image_dataset_v1"

    # model_path = (
    #     AppPaths.models_dir
    #     / "image_v1"
    #     / "image_sign_model.pkl"
    # )

    # labels_path = (
    #     AppPaths.models_dir
    #     / "image_v1"
    #     / "image_labels_map.json"
    # )

    # metadata_path = (
    #     AppPaths.models_dir
    #     / "image_v1"
    #     / "image_training_metadata.json"
    # )


class VideoPaths:
    # =========================
    # VIDEO V2 (ACTIVE)
    # =========================

    dataset_dir = AppPaths.uploads_dir / "video_dataset_v2"

    model_path = (
        AppPaths.models_dir
        / "video_v2"
        / "video_sign_model.pkl"
    )

    labels_path = (
        AppPaths.models_dir
        / "video_v2"
        / "video_labels_map.json"
    )

    metadata_path = (
        AppPaths.models_dir
        / "video_v2"
        / "video_training_metadata.json"
    )

    # =========================
    # VIDEO V1 (OLD)
    # =========================

    # dataset_dir_v1 = AppPaths.uploads_dir / "video_dataset_v1"

    # model_path_v1 = (
    #     AppPaths.models_dir
    #     / "video_v1"
    #     / "video_sign_model.pkl"
    # )

    # labels_path_v1 = (
    #     AppPaths.models_dir
    #     / "video_v1"
    #     / "video_labels_map.json"
    # )

    # metadata_path_v1 = (
    #     AppPaths.models_dir
    #     / "video_v1"
    #     / "video_training_metadata.json"
    # )
