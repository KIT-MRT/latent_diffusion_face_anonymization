"""FastAPI server for LDFA anonymization."""

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger as loguru_logger
import uvicorn

# Add parent directory to path for LDFA imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from server.api.routes import router
from server.api.dependencies import services
from server.services.detector_manager import DetectorManager
from server.services.anonymization import AnonymizationService
from server.services.content_filter import ContentFilterService


def setup_logging(config: dict):
    """Configure logging."""
    log_config = config.get('logging', {})
    
    # Remove default handlers
    loguru_logger.remove()
    
    # Add console handler
    loguru_logger.add(
        sink=sys.stderr,
        format=log_config.get('format', "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"),
        level=log_config.get('level', 'INFO'),
    )
    
    # Add file handler
    log_file = log_config.get('file', '/tmp/anonymization/server.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    loguru_logger.add(
        sink=log_file,
        format=log_config['format'],
        level=log_config.get('level', 'INFO'),
        rotation="10 MB",
        retention="7 days",
    )
    
    # Configure standard logging to use loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            loguru_logger.opt(depth=6, exception=record.exc_info).log(
                record.levelname, record.getMessage()
            )
    
    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    
    return loguru_logger


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config' / 'server_config.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    loguru_logger.info("Starting LDFA Anonymization Server...")
    
    try:
        # Load config
        config = load_config()
        
        # Setup logging
        setup_logging(config)
        loguru_logger.info("Configuration loaded")
        
        # Initialize services
        services.initialize(config)
        
        # Initialize content filter
        content_filter = ContentFilterService(config.get('content_filter', {}))
        services.content_filter = content_filter
        loguru_logger.info("Content filter initialized")
        
        # Initialize detector manager
        detector_config = config.get('detectors', {})
        detector_manager = DetectorManager(detector_config)
        detector_manager.initialize()
        services.detector_manager = detector_manager
        loguru_logger.info("Detector manager initialized")
        
        # Initialize multi-GPU endpoint pool if LDA is available
        endpoint_pool = None
        gpu_config = config.get('gpu', {})
        
        if gpu_config.get('enabled', False):
            try:
                from diffusion_face_anonymisation.multi_gpu_utils import APIEndpointPool
                
                base_port = gpu_config.get('base_port', 7860)
                gpu_count = gpu_config.get('count', 1)
                api_timeout = config.get('api', {}).get('timeout', 120)

                # Compose layout: each SD container binds 7860 and is reached
                # by service name (sd-api-gpu0, sd-api-gpu1, ...). Host layout:
                # 127.0.0.1 with base_port + i. Env vars let the generator
                # switch modes without rebuilding the image.
                host_template = os.environ.get("SD_API_HOST_TEMPLATE", "127.0.0.1")
                port_override_raw = os.environ.get("SD_API_PORT")
                port_override = int(port_override_raw) if port_override_raw else None

                loguru_logger.info(f"Initializing API endpoint pool for {gpu_count} GPUs...")
                endpoint_pool = APIEndpointPool(
                    base_port=base_port,
                    gpu_count=gpu_count,
                    timeout=api_timeout,
                    host_template=host_template,
                    port_override=port_override,
                )
                loguru_logger.info(
                    f"API endpoint pool initialized: {endpoint_pool.endpoints}"
                )
                
            except Exception as e:
                loguru_logger.warning(f"Failed to initialize multi-GPU pool: {e}. LDA will use single GPU.")
        
        # Initialize anonymization service
        anon_config = config.get('methods', {})
        anon_service = AnonymizationService(
            config=anon_config,
            detector_manager=detector_manager,
            endpoint_pool=endpoint_pool,
        )
        services.anonymization_service = anon_service
        loguru_logger.info("Anonymization service initialized")
        
        loguru_logger.info("✓ Server startup complete")
        
    except Exception as e:
        loguru_logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    loguru_logger.info("Shutting down server...")
    
    if services.anonymization_service and services.anonymization_service.endpoint_pool:
        services.anonymization_service.endpoint_pool.close()
        loguru_logger.info("API endpoint pool closed")
    
    loguru_logger.info("Server shutdown complete")


def create_app():
    """Create FastAPI application."""
    app = FastAPI(
        title="LDFA Anonymization API",
        description="API for face, body, and license plate anonymization using Latent Diffusion",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(router, prefix="/api")
    
    @app.get("/")
    async def root():
        return {
            "message": "LDFA Anonymization API",
            "docs": "/docs",
            "health": "/api/health"
        }
    
    return app


app = create_app()


if __name__ == "__main__":
    # Load config for server settings
    config = load_config()
    server_config = config.get('server', {})
    
    uvicorn.run(
        "main:app",
        host=server_config.get('host', '0.0.0.0'),
        port=server_config.get('port', 8000),
        reload=server_config.get('reload', False),
        workers=server_config.get('workers', 1),
    )
