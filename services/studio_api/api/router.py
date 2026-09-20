from fastapi import APIRouter

from studio_api.api.ai_studio_endpoint import (
    router as ai_studio_router,
)
from studio_api.api.dashboard_endpoint import (
    router as dashboard_router,
)


router = APIRouter(
    prefix="/studio",
    tags=["Studio"],
)

router.include_router(
    dashboard_router,
)

router.include_router(
    ai_studio_router,
)