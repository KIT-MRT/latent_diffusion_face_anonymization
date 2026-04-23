#!/usr/bin/env python3
"""Gradio web interface for LDFA anonymization demo.

Flow:
  1. Webcam preview (live) on the top-left.
  2. "Capture Frame" snapshots the current webcam frame into the fixed
     preview below it.
  3. "Anonymize" POSTs the captured frame to the server.
  4. Result is shown on the right as a before/after reveal slider
     (original on the left, anonymized on the right — drag the divider).
"""

import argparse
import base64
import io
import logging
from typing import Optional

import gradio as gr
import numpy as np
import requests
from PIL import Image
from gradio_imageslider import ImageSlider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnonymizationClient:
    def __init__(self, server_url: str, timeout: int = 120):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def check_health(self) -> dict:
        try:
            r = self.session.get(f"{self.server_url}/api/health", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def anonymize(
        self,
        image: Image.Image,
        targets: list,
        body_method: str,
        lp_method: str,
        detector: str,
        body_threshold: float,
        lp_threshold: float,
    ) -> dict:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        files = {"image": ("image.png", buf, "image/png")}
        # FastAPI's bool form parser is strict about truthy strings — send
        # lowercase to avoid surprises.
        data = {
            "targets": ",".join(targets),
            "body_method": body_method,
            "lp_method": lp_method,
            "detector": detector,
            "body_threshold": body_threshold,
            "lp_threshold": lp_threshold,
            "return_masks": "true",
            "return_original": "true",
        }
        r = self.session.post(
            f"{self.server_url}/api/anonymize",
            files=files,
            data=data,
            timeout=self.timeout,
        )
        result = r.json()
        logger.info(
            "server response: success=%s, has_anon=%s, has_orig=%s, det_keys=%s",
            result.get("success"),
            bool(result.get("anonymized_image")),
            bool(result.get("original_image")),
            list((result.get("detections") or {}).keys()),
        )
        if not result.get("success", False):
            return {"success": False, "error": result.get("error", "Unknown error")}

        anon = self._decode(result["anonymized_image"]) if result.get("anonymized_image") else None
        orig = self._decode(result["original_image"]) if result.get("original_image") else image
        return {
            "success": True,
            "anonymized_image": anon,
            "original_image": orig,
            "processing_time_ms": result.get("processing_time_ms", 0),
            "method_used": result.get("method_used", {}),
            "detections": result.get("detections", {}),
        }

    @staticmethod
    def _decode(data_url: str) -> Image.Image:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(data_url)))


client: Optional[AnonymizationClient] = None


def server_banner() -> str:
    if client is None:
        return "❌ Not configured"
    h = client.check_health()
    if h.get("status") == "healthy":
        return f"✅ Connected ({h.get('gpu_count', 0)} GPUs)"
    return f"❌ {h.get('message', 'Unknown error')}"


def capture_frame(webcam_frame):
    """Snapshot the current webcam frame into the fixed preview."""
    if webcam_frame is None:
        return gr.update(), "⚠ Webcam has no frame yet — allow permissions and wait a moment."
    return webcam_frame, "📸 Frame captured. Click **Anonymize** to process."


def run_anonymize(
    captured,
    targets,
    body_method,
    lp_method,
    detector,
    body_threshold,
    lp_threshold,
    progress=gr.Progress(),
):
    if captured is None:
        return gr.update(), "⚠ Capture a frame first."
    if client is None:
        return gr.update(), "❌ Server not configured (pass --server-url)."

    image = Image.fromarray(captured) if isinstance(captured, np.ndarray) else captured

    progress(0.2, desc="Sending to server...")
    result = client.anonymize(
        image=image,
        targets=list(targets) if targets else [],
        body_method=body_method,
        lp_method=lp_method,
        detector=detector,
        body_threshold=body_threshold,
        lp_threshold=lp_threshold,
    )
    if not result["success"]:
        return gr.update(), f"❌ {result['error']}"

    orig_img = result.get("original_image") or image
    anon_img = result.get("anonymized_image")
    if anon_img is None:
        return gr.update(), "❌ Server returned no anonymized image"

    dets = result.get("detections") or {}
    counts = ", ".join(f"{k}:{len(v)}" for k, v in dets.items() if v) or "no detections"
    status = f"✅ {counts} · {result['processing_time_ms'] / 1000:.2f}s"
    logger.info(
        "slider inputs: orig=%s anon=%s",
        f"{orig_img.size} {orig_img.mode}" if orig_img else None,
        f"{anon_img.size} {anon_img.mode}" if anon_img else None,
    )
    return (orig_img, anon_img), status


def create_app(server_url: str):
    global client
    client = AnonymizationClient(server_url)

    with gr.Blocks(title="LDFA Anonymization Demo", theme=gr.themes.Base()) as demo:
        gr.Markdown("# 🔒 LDFA Anonymization Demo")

        with gr.Row():
            # ─── Left: webcam + captured preview + controls ─────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Webcam")
                webcam = gr.Image(
                    label="Live preview",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    height=260,
                    show_label=False,
                )
                capture_btn = gr.Button("📸 Capture Frame", variant="secondary")

                gr.Markdown("### 🖼 Captured frame")
                captured = gr.Image(
                    label="Captured",
                    type="numpy",
                    interactive=False,
                    height=260,
                    show_label=False,
                )

                gr.Markdown("### ⚙ Settings")
                targets_check = gr.CheckboxGroup(
                    label="Targets",
                    choices=["body", "lp"],
                    value=["body", "lp"],
                )
                detector_dropdown = gr.Dropdown(
                    label="Detector", choices=["sam3", "yolo"], value="sam3"
                )
                with gr.Row():
                    body_method = gr.Dropdown(
                        label="Body method",
                        choices=["white", "gauss", "pixel", "lda"],
                        value="pixel",
                    )
                    lp_method = gr.Dropdown(
                        label="LP method",
                        choices=["white", "gauss", "pixel"],
                        value="pixel",
                    )
                with gr.Row():
                    body_threshold = gr.Slider(
                        label="Body threshold", minimum=0.1, maximum=0.9,
                        value=0.5, step=0.05,
                    )
                    lp_threshold = gr.Slider(
                        label="LP threshold", minimum=0.1, maximum=0.9,
                        value=0.5, step=0.05,
                    )

                anon_btn = gr.Button("🔒 Anonymize", variant="primary", size="lg")
                server_status = gr.Textbox(
                    label="Server", value=server_banner(), interactive=False
                )

            # ─── Right: before/after slider ─────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Result — drag the divider to compare")
                comparison = ImageSlider(
                    label="← Original    Anonymized →",
                    type="pil",
                    height=620,
                )
                status_text = gr.Textbox(label="Status", interactive=False)

        # Wiring
        capture_btn.click(
            fn=capture_frame,
            inputs=[webcam],
            outputs=[captured, status_text],
        )
        anon_btn.click(
            fn=run_anonymize,
            inputs=[
                captured,
                targets_check,
                body_method,
                lp_method,
                detector_dropdown,
                body_threshold,
                lp_threshold,
            ],
            outputs=[comparison, status_text],
        )

        gr.Markdown(
            "---\n"
            "**Note:** `lda` takes 10–20s on a GPU, the other methods are near-instant."
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LDFA Anonymization Demo Client")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    logger.info(f"Connecting to server: {args.server_url}")
    demo = create_app(args.server_url)
    demo.launch(server_port=args.port, share=args.share, show_error=True)


if __name__ == "__main__":
    main()
