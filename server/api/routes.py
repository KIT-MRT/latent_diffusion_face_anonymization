"""FastAPI routes for anonymization API."""

import logging
import time
import base64
import io
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import numpy as np

from .models import (
    AnonymizationResponse,
    DetectionResult,
    HealthResponse,
    MethodsResponse,
)
from .dependencies import services

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and dependencies."""
    try:
        sd_api_available = False
        gpu_count = 0
        
        # Check SD API availability
        if services.config.get('gpu', {}).get('enabled', False):
            try:
                import os
                import requests
                base_port = services.config['gpu'].get('base_port', 7860)
                gpu_count = services.config['gpu'].get('count', 1)

                host_template = os.environ.get("SD_API_HOST_TEMPLATE", "127.0.0.1")
                port_override = os.environ.get("SD_API_PORT")
                host = host_template.format(gpu=0) if "{gpu}" in host_template else host_template
                port = int(port_override) if port_override else base_port
                response = requests.get(
                    f"http://{host}:{port}/sdapi/v1/options",
                    timeout=5
                )
                sd_api_available = response.status_code == 200
            except Exception as e:
                logger.warning(f"SD API check failed: {e}")
        
        # Check content filter
        content_filter = services.get_content_filter()
        content_filter_loaded = content_filter.enabled if content_filter else False
        
        # Check detectors
        detector_manager = services.get_detector_manager()
        detectors_loaded = detector_manager.is_initialized()
        
        return HealthResponse(
            status="healthy",
            sd_api_available=sd_api_available,
            gpu_count=gpu_count,
            content_filter_loaded=content_filter_loaded,
            detectors_loaded=detectors_loaded,
            message="All services operational" if detectors_loaded else "Detectors not initialized"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            sd_api_available=False,
            gpu_count=0,
            content_filter_loaded=False,
            detectors_loaded=False,
            message=str(e)
        )


@router.get("/methods", response_model=MethodsResponse)
async def get_available_methods():
    """Get available anonymization methods and detectors."""
    return MethodsResponse(
        targets={
            "body": ["white", "gauss", "pixel", "lda"],
            "lp": ["white", "gauss", "pixel"],
            "face": ["white", "gauss", "pixel", "lda"],
        },
        detectors=["sam3", "yolo"]
    )


@router.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_image(
    image: UploadFile = File(..., description="Image to anonymize"),
    targets: str = Form(default="body,lp", description="Comma-separated targets"),
    body_method: str = Form(default="pixel", description="Body anonymization method"),
    lp_method: str = Form(default="pixel", description="LP anonymization method"),
    detector: str = Form(default="sam3", description="Detector type"),
    body_threshold: float = Form(default=0.5, ge=0, le=1),
    lp_threshold: float = Form(default=0.5, ge=0, le=1),
    return_masks: bool = Form(default=True),
    return_original: bool = Form(default=True),
):
    """
    Anonymize faces, bodies, and/or license plates in an image.
    
    Returns anonymized image and optionally masks/detections.
    """
    start_time = time.time()
    
    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image type")
        
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(img)
        
        # Content filter check
        content_filter = services.get_content_filter()
        filter_result = content_filter.check_image(img_np)
        
        warnings = []
        if not filter_result:
            logger.warning(f"Content filter blocked image: {filter_result.reason}")
            return AnonymizationResponse(
                success=False,
                processing_time_ms=(time.time() - start_time) * 1000,
                method_used={},
                warnings=[filter_result.reason],
                error="Content filter: Image contains inappropriate content"
            )
        
        if filter_result.nsfw_score > 0.3:  # Log even if below threshold
            warnings.append(f"Low-level NSFW score: {filter_result.nsfw_score:.3f}")
        
        # Parse targets
        target_list = [t.strip().lower() for t in targets.split(',')]
        
        # Build methods dict
        methods = {}
        if 'body' in target_list:
            methods['body'] = body_method
        if 'lp' in target_list:
            methods['lp'] = lp_method
        
        # Get services
        anon_service = services.get_anonymization_service()
        detector_manager = services.get_detector_manager()
        
        # Reinitialize detector if different
        if detector_manager.detector_type != detector:
            logger.info(f"Switching detector to {detector}")
            detector_manager.initialize(detector)
        
        # Run anonymization
        logger.info(f"Anonymizing image with targets={target_list}, methods={methods}")
        
        anon_img, detections, metadata = anon_service.anonymize(
            image=img,
            targets=target_list,
            methods=methods,
        )
        
        # Get visualization data
        viz_data = anon_service.get_detection_data_for_visualization(detections, img)
        
        # Build response
        anon_img_bytes = io.BytesIO()
        anon_img.save(anon_img_bytes, format='PNG')
        anon_img_b64 = base64.b64encode(anon_img_bytes.getvalue()).decode('utf-8')
        
        response = AnonymizationResponse(
            success=True,
            anonymized_image=f"data:image/png;base64,{anon_img_b64}",
            processing_time_ms=metadata['processing_time_ms'],
            method_used=metadata['methods_used'],
            warnings=warnings,
            detections={},
        )
        
        # Add original image if requested
        if return_original:
            orig_bytes = io.BytesIO()
            img.save(orig_bytes, format='PNG')
            orig_b64 = base64.b64encode(orig_bytes.getvalue()).decode('utf-8')
            response.original_image = f"data:image/png;base64,{orig_b64}"
        
        # Add detections if masks requested
        if return_masks:
            response.detections = {
                'body': [
                    DetectionResult(
                        bbox=d['bbox'],
                        confidence=d['confidence'],
                        mask=f"data:image/png;base64,{d['mask']}" if 'mask' in d else None
                    )
                    for d in viz_data['body']
                ],
                'lp': [
                    DetectionResult(
                        bbox=d['bbox'],
                        confidence=d['confidence'],
                        mask=f"data:image/png;base64,{d['mask']}" if 'mask' in d else None
                    )
                    for d in viz_data['lp']
                ],
            }
        
        logger.info(f"Anonymization completed in {metadata['processing_time_ms']:.1f}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anonymization failed: {e}", exc_info=True)
        return AnonymizationResponse(
            success=False,
            processing_time_ms=(time.time() - start_time) * 1000,
            method_used={},
            error=str(e)
        )


@router.post("/detect")
async def detect_objects(
    image: UploadFile = File(...),
    targets: str = Form(default="body,lp"),
    detector: str = Form(default="sam3"),
    threshold: float = Form(default=0.5),
):
    """
    Detect objects without anonymization.
    Returns masks and bounding boxes.
    """
    try:
        # Read image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Save to temp file for detector
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img.save(f)
            temp_path = Path(f.name)
        
        try:
            detector_manager = services.get_detector_manager()
            
            # Reinitialize if needed
            if detector_manager.detector_type != detector:
                detector_manager.initialize(detector)
            
            # Run detection
            target_list = [t.strip() for t in targets.split(',')]
            detections = detector_manager.detect_all(temp_path, target_list)
            
            # Format response
            anon_service = services.get_anonymization_service()
            viz_data = anon_service.get_detection_data_for_visualization(detections, img)
            
            return {
                'success': True,
                'detections': {
                    'body': [
                        {'bbox': d['bbox'], 'confidence': d['confidence'], 'mask': d.get('mask')}
                        for d in viz_data['body']
                    ],
                    'lp': [
                        {'bbox': d['bbox'], 'confidence': d['confidence'], 'mask': d.get('mask')}
                        for d in viz_data['lp']
                    ],
                },
                'counts': {
                    'body': len(detections.get('body', [])),
                    'lp': len(detections.get('lp', [])),
                }
            }
            
        finally:
            temp_path.unlink()
            
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}
