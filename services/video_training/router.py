from fastapi import APIRouter

from video_training.api.train_endpoint import router as train_router
from video_training.api.stats_endpoint import router as stats_router
from video_training.api.predict_endpoint import router as predict_router

router = APIRouter(
    prefix="/video",
    tags=["Video Sign AI"],
)

router.include_router(train_router)
router.include_router(stats_router)
router.include_router(predict_router)