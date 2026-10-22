#!/bin/bash
# Quick wrapper for setup_server_compose.py
# Usage: ./deploy/setup_server.sh [start|stop|restart|status|test] [gpu_count]

set -e

ACTION=${1:-start}
GPU_COUNT=${2:-1}  # Default to 1 GPU for testing

cd "$(dirname "$0")/.."

echo "=========================================="
echo "LDFA Server Deployment"
echo "=========================================="
echo "Action: $ACTION"
echo "GPUs: $GPU_COUNT"
echo "=========================================="
echo ""

# Run the Python script
python3 deploy/setup_server_compose.py --action "$ACTION" --gpu-count "$GPU_COUNT"

echo ""
echo "=========================================="
if [ "$ACTION" = "start" ] || [ "$ACTION" = "restart" ]; then
    echo "✅ Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Wait 30-60 seconds for services to fully start"
    echo "2. Check status: ./deploy/setup_server.sh status"
    echo "3. Test endpoints: ./deploy/setup_server.sh test $GPU_COUNT"
    echo "4. View logs: docker-compose -f docker-compose.server.yml logs -f"
    echo ""
    echo "Client command:"
    echo "  python client/app.py --server-url http://localhost:8000"
fi
echo "=========================================="
