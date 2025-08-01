from fastapi import FastAPI
from mangum import Mangum
from .routers import summarize, classify
from .config import settings
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="TermWise Backend API",
    description="API for the TermWise browser extension to summarize legal documents.",
    version="0.1.0",
)

#--- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(summarize.router, prefix="/api")
app.include_router(classify.router, prefix="/api")

@app.get("/", tags=["Root"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the TermWise API!"}

# Create the Lambda handler
handler = Mangum(app)