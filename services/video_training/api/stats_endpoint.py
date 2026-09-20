from fastapi import APIRouter

from video_training.managers.video_manager import VideoManager


router = APIRouter()


@router.get("/stats")
def get_video_stats() -> dict:
    manager = VideoManager()

    return manager.get_stats()