from fastapi import FastAPI
from mangum import Mangum
from .routers import summarize, classify
from .config import settings

app = FastAPI(
    title="TermWise Backend API",
    description="API for the TermWise browser extension to summarize legal documents.",
    version="0.1.0",
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