from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .core.config import settings
from .api.v1.api import api_router
import sys

# Ensure backend root is in sys.path
sys.path.append(str(settings.BASE_DIR))
sys.path.append(str(settings.BASE_DIR / "backend"))

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS Config
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)
from app.routers import analytics, market
app.include_router(analytics.router)
app.include_router(market.router)

@app.on_event("startup")
async def startup_event():
    from app.core.scheduler import start_scheduler
    start_scheduler()

@app.get("/")
def root():
    return {"message": "Welcome to CryptoVertex API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
