import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import auth
from app.config.bedrock_llm import llm_config
from app.dependencies import get_current_user
from app.utils import logger
from app.config.dbConfig import engine, Base
from app.routers import analysis
import time
import sys
import asyncio


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    yield

app = FastAPI(
    title="Scala code assesment checker",
    description="This API asseses the code quality of scala codebase",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5177",
    "http://localhost:5178"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Public endpoints (auth)
app.include_router(auth.router, prefix="/api/v1", tags=["Authentication"])

# Protected endpoints
app.include_router(
    analysis.router,
    prefix="/api/v1",
    tags=["Analysis"],
    dependencies=[Depends(get_current_user)]
)


@app.get("/health", tags=["Health"])
async def health_check():
    llm_status = "configured" if llm_config._llm and llm_config._embed_model else "not configured"
    return {"status": "ok", "llm": llm_status}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} {response.status_code} {duration:.2f}s"
    )
    return response

if __name__ == "__main__":
    import uvicorn
    load_dotenv()
    port = int(os.getenv("PORT", 8002))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)