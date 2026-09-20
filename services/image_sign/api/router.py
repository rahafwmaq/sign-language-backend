from fastapi import APIRouter

from image_sign.api.predict_endpoint import router as predict_router
from image_sign.api.reset_endpoint import router as reset_router
from image_sign.api.stats_endpoint import router as stats_router
from image_sign.api.train_endpoint import router as train_router


router = APIRouter(
    prefix="/image",
    tags=["Image Sign"],
)

router.include_router(train_router)
router.include_router(predict_router)
router.include_router(stats_router)
router.include_router(reset_router)