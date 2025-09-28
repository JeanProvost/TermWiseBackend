<div align="center">
  <img src="./.github/assets/TermWise_Placeholder_Logo.png" alt="TermWise Logo" width="200"/>
</div>

# TermWise Backend

This repository contains the backend API for the TermWise browser extension. It's a serverless API built with FastAPI, designed to receive legal documents, analyze them using a local AI model, and return a structured, easy-to-understand summary.

## Architecture

The backend is designed as a serverless application that can be deployed on AWS, ensuring scalability and cost-efficiency. For local development and testing, it runs a quantized language model directly on your GPU for fast, private document analysis.

```mermaid
graph LR
    A[Browser Extension] -->|HTTPS POST| B[FastAPI Backend]
    B -->|Process| C[Local AI Model<br/>Phi-3-mini-4k]
    C -->|Structured Summary| B
    B -->|JSON Response| A
    
    subgraph "Local GPU"
        C
    end
```

## Key Features

- **Local AI Processing**: Uses Microsoft's Phi-3-mini-4k model for fast, private document analysis
- **GPU Acceleration**: Optimized to run on NVIDIA GPUs with 4-bit quantization
- **Structured Output**: Returns well-formatted JSON with document type, summary, key terms, and section breakdowns
- **Fast API**: RESTful endpoints for document summarization and classification

## Configuration

All runtime settings are centrally managed through [`app/config.py`](app/config.py) using Pydantic Settings.  The app automatically loads `.env` plus an environment-specific overlay if present (for example `.env.local`, `.env.staging`, `.env.production`).

1. Duplicate `.env.example` to `.env` for local development and fill in the optional values.
2. Create additional files such as `.env.production` with production-ready overrides.
3. Set the `APP_ENV` environment variable to choose which overlay is loaded (defaults to `local`).

Key environment variables:

| Variable | Description |
| --- | --- |
| `APP_ENV` | Environment label (`local`, `staging`, `production`). Drives which `.env.<env>` file is included. |
| `MODEL_PROVIDER` | `huggingface` (default local Transformers) or `bedrock` (AWS managed inference). |
| `MODEL_NAME` | Hugging Face model identifier when using the local provider. |
| `HF_TOKEN` | Optional Hugging Face token for private models. |
| `BEDROCK_MODEL_ID` | AWS Bedrock model identifier (e.g. `anthropic.claude-3-sonnet-20240229`). Required when `MODEL_PROVIDER=bedrock`. |
| `BEDROCK_REGION` | AWS region to call Bedrock in; falls back to `AWS_REGION` if omitted. |
| `BEDROCK_PROFILE` / `BEDROCK_ASSUME_ROLE_ARN` | Optional profile or role to assume when building the Bedrock client. |

When `MODEL_PROVIDER=bedrock`, the backend skips local model loading and routes generation through AWS Bedrock using the provided credentials.  For local development keep `MODEL_PROVIDER=huggingface` so the model is downloaded once and cached on disk.

## Getting Started

Follow these steps to set up and run the project locally for development and testing.

### Prerequisites

- Python 3.9+ (Note: Python 3.13 may have compatibility issues with some dependencies)
- NVIDIA GPU with at least 8GB VRAM (recommended) or CPU fallback
- CUDA-compatible GPU drivers
- Git for cloning the repository

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd TermWiseBackend
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
    **Note**: The first time you run the application, it will download the AI model (~4GB). This is a one-time download that will be cached locally.

4.  **GPU Setup (Optional but Recommended):**
    If you have an NVIDIA GPU, the application will automatically use it for faster processing. The model uses 4-bit quantization to minimize VRAM usage (~2-3GB).

### Running the Application Locally

Run the local development server using `uvicorn`:

```bash
python -m uvicorn app.main:app --reload
```

The server will start and listen for requests on `http://127.0.0.1:8000`. The `--reload` flag will automatically restart the server whenever you make changes to the code.

**First Run**: The initial request will trigger the model download and loading, which may take a few minutes. Subsequent requests will be much faster.

### Testing the API

You can test the running API using `curl` or Postman.

**Using `curl`:**

```bash
curl -X POST "http://127.0.0.1:8000/api/summarize/" \
-H "Content-Type: application/json" \
-d '{"text": "Your legal document text goes here..."}'
```

**Note the trailing slash in the URL!**

**Using Postman:**

1.  Create a new **POST** request.
2.  Set the URL to `http://127.0.0.1:8000/api/summarize/` (with trailing slash).
3.  Go to the **Body** tab, select **raw**, and choose **JSON** from the dropdown.
4.  Paste the following into the body:
    ```json
    {
        "text": "Your legal document text goes here..."
    }
    ```
5.  Click **Send**.

### API Response Format

The API returns a structured JSON response:

```json
{
  "document_type": "Privacy Policy",
  "overall_summary": "This document outlines how user data is collected and used...",
  "key_terms": [
    {
      "term": "Personal Information",
      "definition": "Data that can identify you as an individual..."
    }
  ],
  "sectional_summaries": [
    {
      "section_title": "Data Collection",
      "detailed_summary": "The service collects user data through..."
    }
  ]
}
```

## Performance Optimization

The backend is optimized for performance:

- **4-bit Quantization**: Reduces model size from ~8GB to ~2GB with minimal quality loss
- **GPU Acceleration**: Automatically uses CUDA-capable GPUs when available
- **Lazy Loading**: Model loads only on first request to minimize startup time

## Deployment

### Local GPU or Hugging Face Inference

The defaults assume a local GPU and Hugging Face hosted weights.  Use `.env.local` for these settings:

```dotenv
APP_ENV=local
MODEL_PROVIDER=huggingface
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
USE_GPU=true
TORCH_DTYPE=bfloat16
```

### AWS Bedrock

To deploy on AWS with Bedrock managed inference:

1. Ensure the instance or container has IAM permissions for Bedrock (via role or static keys).
2. Create `.env.production` with Bedrock settings, for example:

  ```dotenv
  APP_ENV=production
  MODEL_PROVIDER=bedrock
  AWS_REGION=us-east-1
  BEDROCK_REGION=us-east-1
  BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229
  BEDROCK_PROFILE=termwise-prod
  ```

3. Set `APP_ENV=production` in the process environment before launching the service.
4. Start the FastAPI app behind `uvicorn`, ECS/EKS, or another process manager.  No large model downloads occur because inference is delegated to Bedrock.

The repository already includes dependencies (`boto3`) needed for Bedrock.  The summarizer automatically builds a Bedrock Runtime client using the configured region, optional profile, and role assumption.  Adjust the environment to supply AWS credentials through IAM roles, environment variables, or AWS profiles following standard AWS security practices.

### Serverless (Lambda)

For production deployment on AWS Lambda, the application includes:
- AWS Lambda handler via Mangum
- Serverless-friendly configuration
- Environment variable support

## Troubleshooting

- **Model Download Issues**: If you're behind a proxy, configure your proxy settings before running
- **GPU Memory Errors**: The 4-bit quantization should work on GPUs with 8GB+ VRAM
- **Slow First Request**: This is normal - the model needs to load into memory
- **JSON Parsing Errors**: The model occasionally produces imperfect JSON; the app includes fallback handling