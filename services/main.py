from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from image_sign.api.router import router as image_router
from studio_api.api.router import router as studio_router
from video_training.router import router as video_router


app = FastAPI(
    title="GOSI Sign AI API",
    version="1.0.0",
    description="GOSI Sign Language AI Platform API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # غيّريها لاحقًا للإنتاج
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router)
app.include_router(video_router)
app.include_router(studio_router)


@app.get(
    "/",
    tags=["Health"],
    summary="Health Check",
)
def home() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "GOSI Sign AI API",
        "version": "1.0.0",
    }