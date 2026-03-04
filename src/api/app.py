"""FastAPI application for Eviction Lab yearly model scoring."""

import logging
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.exceptions import CountyNotFoundError, ScoringError
from src.api.schemas import MetadataResponse, ScoreRequest, ScoreResponse
from src.services.eviction_lab_scoring_service import (
    ScoringServiceError,
    get_scoring_service,
)

LOGGER = logging.getLogger(__name__)


def _get_cors_origins_from_env() -> list[str]:
    """Read CORS origins from API_CORS_ORIGINS env var."""
    raw_value = os.getenv("API_CORS_ORIGINS", "")
    if raw_value.strip() == "":
        return []

    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="Eviction Risk Demo API",
        version="0.1.0",
        description="HTTP API for Eviction Lab yearly county risk scoring.",
    )

    allowed_origins = _get_cors_origins_from_env()
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.mount("/demo", StaticFiles(directory="web", html=True), name="demo")

    @app.exception_handler(CountyNotFoundError)
    async def county_not_found_handler(
        request: Request, exc: CountyNotFoundError
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ScoringError)
    async def scoring_error_handler(
        request: Request, exc: ScoringError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        LOGGER.exception("Unexpected error: %s", exc)
        return JSONResponse(
            status_code=500, content={"detail": "An unexpected error occurred."}
        )

    @app.get("/")
    def root() -> RedirectResponse:
        """Redirect root requests to the static demo site."""
        return RedirectResponse(url="/demo/")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Return API health status for simple availability checks."""
        return {"status": "ok"}

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        """Return model metadata used by docs and UI about pages."""
        service = get_scoring_service()

        try:
            payload = service.get_metadata()
            return MetadataResponse(**payload)
        except ScoringServiceError as error:
            raise HTTPException(status_code=error.status_code, detail=error.to_detail()) from error

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        """Score one county for one year, defaulting to latest when omitted."""
        service = get_scoring_service()

        try:
            payload = service.score_county(
                county_fips=request.county_fips,
                as_of_year=request.as_of_year,
            )
            return ScoreResponse(**payload)
        except ScoringServiceError as error:
            if error.status_code == 404:
                raise CountyNotFoundError(request.county_fips) from error
            raise HTTPException(status_code=error.status_code, detail=error.to_detail()) from error

    @app.post("/score/batch", response_model=list[ScoreResponse])
    def score_batch(requests: list[ScoreRequest]) -> list[ScoreResponse]:
        """Score multiple county-year requests in one API call."""
        service = get_scoring_service()
        responses: list[ScoreResponse] = []

        for request in requests:
            try:
                payload = service.score_county(
                    county_fips=request.county_fips,
                    as_of_year=request.as_of_year,
                )
                responses.append(ScoreResponse(**payload))
            except ScoringServiceError as error:
                if error.status_code == 404:
                    raise CountyNotFoundError(request.county_fips) from error
                raise HTTPException(status_code=error.status_code, detail=error.to_detail()) from error

        return responses

    return app


app = create_app()
