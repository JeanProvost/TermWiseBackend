from fastapi import FastAPI
from .routers import summarize

app = FastAPI(
    title="Term Wise Backend",
    description="AI-powered document summarization API",
    version="0.1.0",
)

app.include_router(summarize.router)

@app.get("/")
def health_check():
    return {"status": "active"}