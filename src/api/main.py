from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.database import get_stats, init_db, log_prediction
from src.api.schemas import ExplainRequest, ExplainResponse, PredictionRequest, PredictionResponse, StatsResponse
from src.api.services import _risk_level
from src.api.services import explain as explain_service
from src.api.services import predict as predict_service
from src.config import config
from src.config.gdrive_loader import ensure_models  # adapte le chemin si besoin


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
def predict(request: PredictionRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    """Endpoint to predict mental health signals from input text."""
    try:
        result = predict_service(request.text, request.model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(
        log_prediction,
        request.text,
        request.model_type,
        result["label"],
        result["probability"],
        _risk_level(result["probability"]),
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
