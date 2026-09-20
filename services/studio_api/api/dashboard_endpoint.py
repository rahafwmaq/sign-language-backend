from fastapi import APIRouter

from studio_api.repositories.studio_repository import (
    StudioRepository,
)


router = APIRouter()


@router.get("/dashboard")
def get_dashboard() -> dict:
    repository = StudioRepository()

    return {
        "status": "success",
        "data": repository.get_dashboard_data(),
    }