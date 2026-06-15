from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.detect_routes import detect_router
from app.core.config import apply_runtime_environment, settings
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    apply_runtime_environment(settings)
    setup_logging(settings.log_file)

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(detect_router)
    app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

    @app.get("/")
    async def index():
        return FileResponse("app/web/static/index.html")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.app_port)


if __name__ == "__main__":
    main()
