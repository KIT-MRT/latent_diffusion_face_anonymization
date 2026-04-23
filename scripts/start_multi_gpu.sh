#!/bin/bash
# Start multi-GPU SD API containers manually

CONFIG=${1:-config-everything.yaml}

# Extract config values using python
COUNT=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['gpu']['count'])")
BASE_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['gpu']['base_port'])")

# Get device_ids as a comma-separated list
DEVICE_IDS_RAW=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); d=c['gpu'].get('device_ids', list(range(c['gpu']['count']))); print(','.join(map(str,d)))")

# Convert to array
IFS=',' read -ra DEVICE_IDS <<< "$DEVICE_IDS_RAW"

echo "Starting $COUNT GPUs on ports $BASE_PORT to $((BASE_PORT + COUNT - 1))"
echo "Device IDs: ${DEVICE_IDS[*]}"

# First remove any existing containers
for i in $(seq 0 $((COUNT-1))); do
    GPU_ID=${DEVICE_IDS[$i]}
    docker rm -f "sd-api-gpu${GPU_ID}" 2>/dev/null
done

for i in $(seq 0 $((COUNT-1))); do
    GPU_ID=${DEVICE_IDS[$i]}
    PORT=$((BASE_PORT + i))
    NAME="sd-api-gpu${GPU_ID}"
    
    echo "Starting $NAME on port $PORT with GPU $GPU_ID..."
    docker run -d --name "$NAME" \
        --gpus "device=$GPU_ID" \
        --shm-size 16g \
        -p "${PORT}:7860" \
        -e CUDA_VISIBLE_DEVICES="$GPU_ID" \
        ldfa python3 webui.py --listen --api --xformers
    
    echo "Waiting 15s for $NAME to initialize..."
    sleep 15
done

echo "All containers started. Check status with: docker ps"
