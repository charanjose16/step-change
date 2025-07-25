# StepChange Backend

## Overview
This project is a FastAPI-based backend for automated code quality assessment and business logic extraction, with a focus on Scala and multi-language codebases. It leverages Large Language Models (LLMs) for advanced code analysis, integrates with AWS S3 for file storage, and uses a vector store (FAISS or PostgreSQL) for efficient document embedding and retrieval. The backend provides secure authentication, project analysis, and graph generation APIs.

## Features
- **Automated Code Quality Assessment**: Analyze Scala and other codebases for quality and business logic using LLMs.
- **Business Logic Extraction**: Extracts requirements, workflows, and key functionalities from code files.
- **Authentication**: JWT-based user signup and login endpoints.
- **S3 Integration**: Download and process code folders from AWS S3 buckets.
- **Vector Store Support**: Store and search code/document embeddings using FAISS or PostgreSQL.
- **Project Graph Generation**: Generate hierarchical and entity relationship graphs for projects.
- **Comprehensive Logging**: File and console logging with detailed tracebacks.

## Tech Stack
- **Python 3.10+**
- **FastAPI**
- **SQLAlchemy** (PostgreSQL backend)
- **boto3** (AWS S3)
- **FAISS** (Vector search)
- **tiktoken** (Tokenization)
- **OpenAI/Azure/Bedrock LLMs**

## Installation
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd server
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**
   - Copy `.env.example` to `.env` and fill in the required values (see below).

## Environment Variables
The following environment variables are required (see `app/config/settings.py`):
- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET` - Secret key for JWT tokens
- `JWT_ALGORITHM` - JWT signing algorithm (e.g., HS256)
- `AUTH_USERNAME`, `AUTH_PASSWORD` - Default admin credentials
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BUCKET_NAME` - AWS S3 credentials
- LLM config: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, etc. (if using Azure/OpenAI)

## Running the Server
You can start the backend with:
```bash
python run_backend.py
```
Or directly with Uvicorn:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

## API Endpoints
### Authentication
- `POST /api/v1/auth/signup` — Register a new user
- `POST /api/v1/auth/login` — Login and receive JWT token

### Analysis (protected)
- `POST /api/v1/analysis` — Analyze an S3 folder and generate requirements
- `POST /api/v1/analysis/graphs` — Generate graphs from file requirements
- `POST /api/v1/analysis/project-graph` — Generate a unified D3 hierarchical project graph

### Health
- `GET /health` — Health check endpoint

## Example `.env` file
```
DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/stepchange
JWT_SECRET=your_jwt_secret
JWT_ALGORITHM=HS256
AUTH_USERNAME=admin
AUTH_PASSWORD=adminpassword
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=ap-south-2
BUCKET_NAME=step-change
# LLM config (Azure/OpenAI/Bedrock)
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_MODEL=your_model
```

## Dependencies
See `requirements.txt` for the full list. Key packages include:
- fastapi
- sqlalchemy
- boto3
- faiss-cpu
- tiktoken
- openai
- uvicorn

## Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE) 