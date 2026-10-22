# 🎬 LDFA Live Demo - Quick Start

Real-time webcam anonymization demo for presentations and high-profile events.

## 🚀 2-Minute Setup

### Server (Remote GPU)

```bash
# SSH to server
ssh your-server.com

# Build Docker images (one-time, ~5 minutes)
docker build -t ldfa .
docker build -t ldfa-api -f server/Dockerfile .

# Start with 1 GPU for testing
./deploy/setup_server.sh start 1

# Or start with all 10 GPUs for production
./deploy/setup_server.sh start 10
```

Wait 30-60 seconds for services to start.

For detailed setup, see `docs/SERVER_QUICK_START.md`

### Client (Your Laptop)

```bash
# Install dependencies
pip install -r client/requirements.txt

# Start demo
python client/app.py --server-url http://your-server.com:8000
```

Open browser: **http://localhost:7860**

---

## 📖 Demo Flow (for Presentations)

### 1. Introduction (30 seconds)
- Show webcam feed
- Explain privacy challenges in computer vision
- Mention: "Real-time anonymization using diffusion models"

### 2. Quick Demo - Pixel Method (1 minute)
- Capture frame
- Click "Anonymize" (pixel method)
- Show result: "Instant anonymization with pixelation"
- Highlight: Bodies + license plates detected automatically

### 3. Quality Demo - LDA Method (30 seconds setup + 15s processing)
- Switch method to "lda"
- Explain: "Latent Diffusion generates realistic faces"
- Click "Anonymize"
- While processing: Explain multi-GPU setup (10x Ada 6000)
- Show result: "Photorealistic anonymization"

### 4. Technical Highlights (optional)
- Toggle detection overlay to show masks
- Mention SAM3 detector accuracy
- Content filtering for inappropriate content

---

## 🎯 Demo Tips

### For Best Results

**Lighting:**
- Bright, even lighting
- Avoid strong shadows
- Face the light source

**Background:**
- Simple backgrounds work best
- Avoid cluttered scenes
- Indoor environments preferred

**Positioning:**
- Full body visible for body anonymization
- License plates clearly visible
- 2-5 meters distance optimal

### Backup Plan

Always have pre-processed images ready:

```bash
# Process test images beforehand
python scripts/anonymize.py --image_dir test_images/ --output_dir demo_output/ \
    --targets body lp --method pixel
```

### Keyboard Shortcuts

- **Space**: Capture frame
- **Enter**: Anonymize
- **Escape**: Clear results

---

## ⚙️ Configuration Quick Reference

### Change Methods

| Use Case | Body Method | LP Method | Speed |
|----------|-------------|-----------|-------|
| **Live Demo** | pixel | pixel | <1s |
| **Quality Demo** | lda | pixel | 15s |
| **Testing** | white | white | <1s |

### Adjust Detection

Lower threshold = more detections (but more false positives)
- Default: 0.5
- Sensitive: 0.3
- Conservative: 0.7

---

## 🐛 Quick Troubleshooting

### "Server not connected"
```bash
# Check server
curl http://your-server.com:8000/api/health

# Check service status
ssh your-server.com
./deploy/setup_server.sh status

# Restart services
./deploy/setup_server.sh restart 10

# View logs
docker-compose -f docker-compose.server.yml logs -f anonymization-api
```

### "Detection failed"
- Try YOLO detector instead of SAM3 (faster, less accurate)
- Lower threshold to 0.3
- Ensure good lighting

### "LDA timeout"
- Increase timeout in `client/app.py`: `timeout=180`
- Check SD API status: `./deploy/setup_server.sh test 10`
- View logs: `docker-compose -f docker-compose.server.yml logs -f sd-api-gpu0`
- Fallback to pixel method

---

## 📊 Performance Expectations

| Scenario | Time | Notes |
|----------|------|-------|
| Pixel/Gauss/White | <2s | Network + detection |
| LDA (single person) | 10-15s | Diffusion generation |
| LDA (multiple people) | 15-20s | Parallel processing |

---

## 🎤 Presentation Script

> "Today we're demonstrating real-time privacy protection using state-of-the-art diffusion models.
>
> Our system automatically detects and anonymizes people and license plates in video streams.
>
> [Capture frame, click anonymize]
>
> With fast methods like pixelation, we achieve near real-time performance.
>
> [Switch to LDA]
>
> For higher quality, we can use latent diffusion to generate photorealistic replacements,
> powered by 10 NVIDIA Ada 6000 GPUs running in parallel.
>
> This technology is crucial for privacy-compliant computer vision in public spaces."

---

## 📞 Emergency Contacts

**Technical Issues:**
- Check logs: `docker-compose -f docker-compose.demo.yml logs -f`
- Health check: `python deploy/health_check.py --server http://server:8000`
- Restart: `docker-compose -f docker-compose.demo.yml restart`

**During Demo:**
- Have backup images ready
- Use pixel method if LDA fails
- Pre-recorded video as last resort

---

## ✅ Pre-Demo Checklist

- [ ] Server running (`curl server:8000/api/health`)
- [ ] All 10 GPUs healthy (`python deploy/health_check.py --check-sd-apis`)
- [ ] Client connected (green status in UI)
- [ ] Webcam working
- [ ] Test anonymization (pixel method)
- [ ] Test LDA (one image)
- [ ] Backup images prepared
- [ ] Network stable (run speedtest)
- [ ] Audio/visual working
- [ ] Presentation slides ready

---

**Good luck with your demo! 🚀**

For detailed documentation, see `docs/DEMO_SETUP.md`
