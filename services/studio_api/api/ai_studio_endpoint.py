from fastapi import APIRouter

from studio_api.repositories.studio_repository import (
    StudioRepository,
)


router = APIRouter()


@router.get("/models")
def get_models() -> dict:
    repository = StudioRepository()

    return {
        "status": "success",
        "data": repository.get_models_data(),
    }