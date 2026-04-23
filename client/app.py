#!/usr/bin/env python3
"""Gradio web interface for LDFA anonymization demo."""

import argparse
import base64
import io
import logging
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import requests
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnonymizationClient:
    """HTTP client for anonymization API."""
    
    def __init__(self, server_url: str, timeout: int = 120):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def check_health(self) -> dict:
        """Check server health."""
        try:
            response = self.session.get(
                f"{self.server_url}/api/health",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_methods(self) -> dict:
        """Get available methods."""
        try:
            response = self.session.get(
                f"{self.server_url}/api/methods",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {'targets': {}, 'detectors': []}
    
    def anonymize(
        self,
        image: Image.Image,
        targets: list,
        body_method: str,
        lp_method: str,
        detector: str = 'sam3',
        body_threshold: float = 0.5,
        lp_threshold: float = 0.5,
        return_masks: bool = True,
        return_original: bool = True,
    ) -> dict:
        """
        Send image for anonymization.
        
        Returns dict with:
        - success: bool
        - anonymized_image: PIL Image or None
        - original_image: PIL Image or None
        - detections: dict with masks/bboxes
        - processing_time_ms: float
        - error: str or None
        """
        try:
            # Convert image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Prepare form data
            files = {'image': ('image.png', img_bytes, 'image/png')}
            
            data = {
                'targets': ','.join(targets),
                'body_method': body_method,
                'lp_method': lp_method,
                'detector': detector,
                'body_threshold': body_threshold,
                'lp_threshold': lp_threshold,
                'return_masks': return_masks,
                'return_original': return_original,
            }
            
            # Send request
            response = self.session.post(
                f"{self.server_url}/api/anonymize",
                files=files,
                data=data,
                timeout=self.timeout,
            )
            
            result = response.json()
            
            if not result.get('success', False):
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown error'),
                    'anonymized_image': None,
                    'original_image': None,
                    'detections': None,
                    'processing_time_ms': result.get('processing_time_ms', 0),
                }
            
            # Decode images
            anon_img = None
            orig_img = None
            
            if result.get('anonymized_image'):
                anon_img = self._decode_image(result['anonymized_image'])
            
            if result.get('original_image'):
                orig_img = self._decode_image(result['original_image'])
            
            return {
                'success': True,
                'anonymized_image': anon_img,
                'original_image': orig_img,
                'detections': result.get('detections'),
                'processing_time_ms': result.get('processing_time_ms', 0),
                'method_used': result.get('method_used', {}),
                'warnings': result.get('warnings', []),
                'error': None,
            }
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': f'Request timed out after {self.timeout}s',
                'anonymized_image': None,
                'original_image': None,
                'detections': None,
                'processing_time_ms': 0,
            }
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'anonymized_image': None,
                'original_image': None,
                'detections': None,
                'processing_time_ms': 0,
            }
    
    def _decode_image(self, data_url: str) -> Image.Image:
        """Decode base64 data URL to PIL Image."""
        if ',' in data_url:
            data_url = data_url.split(',', 1)[1]
        img_bytes = base64.b64decode(data_url)
        return Image.open(io.BytesIO(img_bytes))


# Global client instance
client: Optional[AnonymizationClient] = None


def check_server_connection():
    """Check server connection and return status message."""
    global client
    if client is None:
        return "❌ Not configured", "error"
    
    health = client.check_health()
    
    if health.get('status') == 'healthy':
        gpu_count = health.get('gpu_count', 0)
        return f"✅ Connected ({gpu_count} GPUs)", "success"
    else:
        return f"❌ {health.get('message', 'Unknown error')}", "error"


def anonymize_image(
    image,
    targets,
    body_method,
    lp_method,
    detector,
    body_threshold,
    lp_threshold,
    progress=gr.Progress(),
):
    """Process image anonymization request."""
    global client
    
    if image is None:
        return None, None, None, "Please capture or upload an image first", ""
    
    if client is None:
        return None, None, None, "Server not configured. Start app with --server-url", ""
    
    # Check connection
    health_status, _ = check_server_connection()
    if "❌" in health_status:
        return None, None, None, f"Server error: {health_status}", ""
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Show progress
    progress(0, desc="Sending to server...")
    
    # Call API
    result = client.anonymize(
        image=image,
        targets=targets if targets else [],
        body_method=body_method,
        lp_method=lp_method,
        detector=detector,
        body_threshold=body_threshold,
        lp_threshold=lp_threshold,
        return_masks=True,
        return_original=True,
    )
    
    if not result['success']:
        return None, image, None, f"Error: {result['error']}", ""
    
    # Build status message
    status_parts = []
    if result.get('method_used'):
        methods = result['method_used']
        status_parts.append(f"Methods: {', '.join(f'{k}={v}' for k, v in methods.items())}")
    
    status_parts.append(f"Time: {result['processing_time_ms'] / 1000:.2f}s")
    
    if result.get('warnings'):
        status_parts.append(f"Warnings: {', '.join(result['warnings'])}")
    
    status = " | ".join(status_parts)
    
    # Create overlay with detections
    overlay = None
    if result.get('detections'):
        overlay = create_detection_overlay(image, result['detections'])
    
    return result['anonymized_image'], result['original_image'], overlay, "✅ Success", status


def create_detection_overlay(
    image: Image.Image,
    detections: dict,
    alpha: float = 0.5,
) -> Image.Image:
    """Create overlay image with detection masks."""
    # Create blank overlay
    overlay = np.zeros((*image.size[::-1], 4), dtype=np.uint8)  # RGBA
    
    # Color codes (RGBA)
    colors = {
        'body': (0, 120, 255, 180),      # Blue
        'lp': (255, 215, 0, 180),        # Yellow
        'face': (0, 255, 0, 180),        # Green
    }
    
    # Draw masks
    for target_type in ['body', 'lp', 'face']:
        if target_type not in detections:
            continue
        
        color = colors.get(target_type, (255, 255, 255, 180))
        
        for det in detections[target_type]:
            if not det.get('mask'):
                continue
            
            # Decode mask
            mask_data = det['mask']
            if ',' in mask_data:
                mask_data = mask_data.split(',', 1)[1]
            
            mask_bytes = base64.b64decode(mask_data)
            mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
            mask_np = np.array(mask)
            
            # Create colored mask
            colored_mask = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
            colored_mask[mask_np > 0] = color
            
            # Blend with overlay
            overlay = cv2.addWeighted(
                overlay, 1.0,
                colored_mask, alpha,
                0
            )
    
    return Image.fromarray(overlay, mode='RGBA')


def create_app(server_url: str):
    """Create Gradio interface."""
    global client
    client = AnonymizationClient(server_url)
    
    # Check connection on startup
    health_status, status_type = check_server_connection()
    
    with gr.Blocks(title="LDFA Anonymization Demo", theme=gr.themes.Base()) as demo:
        gr.Markdown("""
        # 🔒 LDFA Anonymization Demo
        
        Real-time face, body, and license plate anonymization using Latent Diffusion.
        """)
        
        with gr.Row():
            # Left panel: Input and controls
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Input")
                
                # Webcam capture
                webcam = gr.Image(
                    label="Webcam",
                    sources=["webcam"],
                    type="numpy",
                    height=300,
                )
                
                capture_btn = gr.Button("📸 Capture Frame", variant="primary")
                
                # Or upload
                upload_image = gr.Image(
                    label="Or upload image",
                    sources=["upload"],
                    type="numpy",
                )
                
                gr.Markdown("### ⚙️ Settings")
                
                # Targets
                targets_check = gr.CheckboxGroup(
                    label="Targets to Anonymize",
                    choices=["body", "lp"],
                    value=["body", "lp"],
                )
                
                # Detector
                detector_dropdown = gr.Dropdown(
                    label="Detector",
                    choices=["sam3", "yolo"],
                    value="sam3",
                )
                
                # Methods
                with gr.Row():
                    body_method = gr.Dropdown(
                        label="Body Method",
                        choices=["white", "gauss", "pixel", "lda"],
                        value="pixel",
                    )
                    lp_method = gr.Dropdown(
                        label="LP Method",
                        choices=["white", "gauss", "pixel"],
                        value="pixel",
                    )
                
                # Thresholds
                with gr.Row():
                    body_threshold = gr.Slider(
                        label="Body Threshold",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                    )
                    lp_threshold = gr.Slider(
                        label="LP Threshold",
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                    )
                
                # Anonymize button
                anon_btn = gr.Button("🔒 Anonymize", variant="primary", size="lg")
                
                # Server status
                server_status = gr.Textbox(
                    label="Server Status",
                    value=health_status,
                    interactive=False,
                )
            
            # Right panel: Results
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Results")
                
                # Comparison viewer
                comparison = gr.Image(
                    label="Comparison (Original ←→ Anonymized)",
                    type="pil",
                    height=400,
                )
                
                # Detection overlay
                overlay_viewer = gr.Image(
                    label="Detection Overlay",
                    type="pil",
                    height=400,
                    visible=True,
                )
                
                # Status
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                )
                
                # Download buttons
                with gr.Row():
                    download_anon = gr.DownloadButton(
                        label="⬇️ Download Anonymized",
                        variant="secondary",
                    )
                    download_orig = gr.DownloadButton(
                        label="⬇️ Download Original",
                        variant="secondary",
                    )
        
        # State to store captured image
        captured_image = gr.State(value=None)
        
        # Wire up capture button
        def capture_frame(webcam_img):
            return webcam_img
        
        capture_btn.click(
            fn=capture_frame,
            inputs=[webcam],
            outputs=[captured_image],
        ).then(
            fn=lambda x: x,
            inputs=[captured_image],
            outputs=[upload_image],
        )
        
        # Wire up anonymize button
        anon_btn.click(
            fn=anonymize_image,
            inputs=[
                upload_image,
                targets_check,
                body_method,
                lp_method,
                detector_dropdown,
                body_threshold,
                lp_threshold,
            ],
            outputs=[comparison, upload_image, overlay_viewer, status_text, status_text],
        )
        
        # Download handlers
        download_anon.click(
            fn=lambda x: x,
            inputs=[comparison],
            outputs=[download_anon],
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Note**: LDA method takes 10-20 seconds. Other methods are near-instant.
        Images are processed on a remote server with 10x NVIDIA Ada 6000 GPUs.
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="LDFA Anonymization Demo Client")
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the anonymization API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run Gradio app on",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link",
    )
    
    args = parser.parse_args()
    
    logger.info(f"Connecting to server: {args.server_url}")
    logger.info(f"Starting Gradio app on port {args.port}")
    
    demo = create_app(args.server_url)
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
