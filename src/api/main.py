import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.database import get_stats, init_db, log_prediction
from src.api.schemas import ExplainRequest, ExplainResponse, PredictionRequest, PredictionResponse, StatsResponse
from src.api.services import _risk_level
from src.api.services import explain as explain_service
from src.api.services import predict as predict_service
from src.config import config
from src.config.gdrive_loader import ensure_models  # adapte le chemin si besoin


def _normalize_language(language: str | None) -> str | None:
    """Normalize language tags to lowercase BCP47-like form (e.g. en-us, fr-fr)."""
    if not language:
        return None
    normalized = language.strip().lower().replace("_", "-")
    return normalized or None


def _is_non_fr_en(language: str | None) -> bool:
    """Return True when language primary subtag is neither French nor English."""
    normalized = _normalize_language(language)
    if not normalized:
        return False
    primary_subtag = normalized.split("-", 1)[0]
    return primary_subtag not in {"fr", "en"}


def _emit_predict_audit_log(
    *,
    model_type: str,
    source_language: str | None,
    client_origin: str,
    is_non_fr_en: bool,
) -> None:
    """Emit one structured log line to Cloud Logging-friendly stdout."""
    payload = {
        "event": "predict_request",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "source_language": source_language,
        "source_language_normalized": _normalize_language(source_language),
        "client_origin": client_origin,
        "is_non_fr_en": is_non_fr_en,
    }
    print(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB tables et télécharge les modèles au démarrage."""
    ensure_models(config.MODELS_DIR, config.GDRIVE_MODEL_FOLDER_ID)
    init_db()
    yield


app = FastAPI(
    title="Mental Health Signal Detector API",
    description="API for detecting mental health signals in text using machine learning models.",
    version="1.0.0",
    lifespan=lifespan,
)

allow_all_origins = "*" in config.CORS_ALLOWED_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all_origins else config.CORS_ALLOWED_ORIGINS,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with quick API usage hints."""
    return {
        "message": "Mental Health Signal Detector API",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "explain": "POST /explain",
            "stats": "GET /stats",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify that the API is running."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest, background_tasks: BackgroundTasks, http_request: Request) -> PredictionResponse:
    """Endpoint to predict mental health signals from input text."""
    client_origin = http_request.headers.get("X-Client-Origin", "unknown").strip().lower() or "unknown"
    source_language = _normalize_language(request.source_language)

    try:
        result = predict_service(request.text, request.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    non_fr_en = _is_non_fr_en(source_language)
    _emit_predict_audit_log(
        model_type=request.model_type,
        source_language=source_language,
        client_origin=client_origin,
        is_non_fr_en=non_fr_en,
    )

    background_tasks.add_task(
        log_prediction,
        request.text,
        request.model_type,
        result["label"],
        result["probability"],
        _risk_level(result["probability"]),
        source_language,
    )
    return PredictionResponse(**result)


@app.post("/explain")
def explain(request: ExplainRequest) -> ExplainResponse:
    """Endpoint to predict and return word-level explanation details."""
    try:
        result = explain_service(
            text=request.text,
            model_type=request.model_type,
            threshold=request.threshold,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExplainResponse(**result)


@app.get("/stats")
def stats() -> StatsResponse:
    """Return aggregated prediction statistics from the database."""
    return StatsResponse(**get_stats())
