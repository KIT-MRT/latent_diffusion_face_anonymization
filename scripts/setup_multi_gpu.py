#!/usr/bin/env python3
"""
Generate docker-compose.yml for multi-GPU setup and manage services.

This script reads config.yaml and:
1. Generates docker-compose.generated.yml with N GPU services
2. Starts/stops/restarts the services
3. Verifies GPU availability
"""

import yaml
import subprocess
import sys
from pathlib import Path
import argparse
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy config.yaml.example to {config_path} and adjust settings."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ["gpu", "api", "paths"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")

    if "count" not in config["gpu"]:
        raise ValueError("Missing required field: gpu.count")

    if "base_port" not in config["gpu"]:
        raise ValueError("Missing required field: gpu.base_port")

    if "weights_dir" not in config["paths"]:
        raise ValueError("Missing required field: paths.weights_dir")

    return config


def check_gpu_availability(gpu_count: int) -> bool:
    """Verify required GPUs are available using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True
        )
        available_gpus = len(
            [line for line in result.stdout.strip().split("\n") if line]
        )

        if available_gpus < gpu_count:
            logger.error(
                f"Requested {gpu_count} GPUs but only {available_gpus} available"
            )
            return False

        logger.info(f"✓ Found {available_gpus} GPUs, using {gpu_count}")
        return True

    except subprocess.CalledProcessError:
        logger.error("Failed to run nvidia-smi. Are NVIDIA drivers installed?")
        return False
    except FileNotFoundError:
        logger.error("nvidia-smi not found. Are NVIDIA drivers installed?")
        return False


def generate_docker_compose(config: dict) -> dict:
    """Generate docker-compose services dict from config."""
    gpu_count = config["gpu"]["count"]
    device_ids = config["gpu"].get("device_ids", list(range(gpu_count)))
    base_port = config["gpu"]["base_port"]
    weights_dir = config["paths"].get("weights_dir", "")

    if weights_dir.startswith("/path/to"):
        weights_dir = ""

    if len(device_ids) != gpu_count:
        logger.warning(
            f"device_ids length ({len(device_ids)}) doesn't match count ({gpu_count}). "
            f"Using first {gpu_count} IDs."
        )
        device_ids = device_ids[:gpu_count]

    services = {}

    for i, gpu_id in enumerate(device_ids):
        port = base_port + i
        service_name = f"sd-api-gpu{gpu_id}"

        logger.info(
            f"  Creating service: {service_name} on port {port} for GPU {gpu_id}"
        )

        services[service_name] = {
            "image": "ldfa",
            "container_name": service_name,
            "build": {"context": ".", "dockerfile": "Dockerfile"},
            "ports": [f"{port}:7860"],
            "environment": {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
            },
            "shm_size": "16g",
            "volumes": (
                [f"{weights_dir}:/opt/gui/models/Stable-diffusion"]
                if weights_dir
                else []
            ),
            "entrypoint": "python3 webui.py --listen --api --xformers",
            "restart": "unless-stopped",
        }

    return {"services": services}


def write_docker_compose(compose_dict: dict, output_path: Path):
    """Write docker-compose dict to YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(compose_dict, f, default_flow_style=False, sort_keys=False)
    logger.info(f"✓ Generated {output_path}")


def start_services(compose_file: Path, build: bool = False):
    """Start all docker-compose services."""
    logger.info("Starting SD API services...")

    cmd = [
        "docker-compose",
        "--log-level",
        "DEBUG",
        "-f",
        str(compose_file),
        "up",
        "-d",
    ]
    if build:
        cmd.append("--build")

    try:
        subprocess.run(cmd, check=True)
        logger.info("✓ All services started successfully")
        logger.info("\nWait ~30-60 seconds for Stable Diffusion models to load...")
        logger.info(
            "You can check status with: docker-compose -f docker-compose.generated.yml logs -f"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start services: {e}")
        sys.exit(1)

    logger.info("Waiting 15s before starting more services...")
    time.sleep(15)

    for i in range(1, 8):
        try:
            subprocess.run(
                [
                    "docker-compose",
                    "--log-level",
                    "DEBUG",
                    "-f",
                    str(compose_file),
                    "up",
                    "-d",
                    f"sd-api-gpu{i}",
                ],
                check=True,
            )
            logger.info(f"✓ Started sd-api-gpu{i}")
            time.sleep(12)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to start sd-api-gpu{i}: {e}")


def stop_services(compose_file: Path):
    """Stop all docker-compose services."""
    logger.info("Stopping SD API services...")

    try:
        subprocess.run(["docker-compose", "-f", str(compose_file), "down"], check=True)
        logger.info("✓ All services stopped")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to stop services: {e}")
        sys.exit(1)


def show_status(compose_file: Path):
    """Show status of docker-compose services."""
    try:
        subprocess.run(["docker-compose", "-f", str(compose_file), "ps"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get status: {e}")
        sys.exit(1)


def test_endpoints(config: dict):
    """Test if all API endpoints are responding."""
    import requests

    base_port = config["gpu"]["base_port"]
    gpu_count = config["gpu"]["count"]

    logger.info("\nTesting API endpoints...")

    all_ok = True
    for i in range(gpu_count):
        port = base_port + i
        url = f"http://127.0.0.1:{port}/sdapi/v1/options"

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"  ✓ Port {port}: OK")
            else:
                logger.warning(f"  ✗ Port {port}: HTTP {response.status_code}")
                all_ok = False
        except requests.exceptions.ConnectionError:
            logger.warning(
                f"  ✗ Port {port}: Connection refused (service may still be starting)"
            )
            all_ok = False
        except requests.exceptions.Timeout:
            logger.warning(f"  ✗ Port {port}: Timeout")
            all_ok = False
        except Exception as e:
            logger.warning(f"  ✗ Port {port}: {e}")
            all_ok = False

    if all_ok:
        logger.info("\n✓ All endpoints are responding!")
    else:
        logger.warning(
            "\n⚠ Some endpoints are not responding yet. "
            "Services may still be starting up (can take 30-60 seconds)."
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup and manage multi-GPU SD API services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate docker-compose and start services
  python setup_multi_gpu.py --action start --config config.yaml
  
  # Stop all services
  python setup_multi_gpu.py --action stop
  
  # Restart services with rebuild
  python setup_multi_gpu.py --action restart --build
  
  # Show service status
  python setup_multi_gpu.py --action status
  
  # Test if endpoints are responding
  python setup_multi_gpu.py --action test --config config.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--action",
        choices=["start", "stop", "restart", "status", "test", "generate"],
        default="start",
        help="Action to perform (default: start)",
    )
    parser.add_argument(
        "--build", action="store_true", help="Rebuild Docker images when starting"
    )
    parser.add_argument(
        "--compose-file",
        type=str,
        default="docker-compose.generated.yml",
        help="Output docker-compose file (default: docker-compose.generated.yml)",
    )

    args = parser.parse_args()

    # Paths
    config_path = Path(args.config)
    compose_file = Path(args.compose_file)

    # Actions that don't need config
    if args.action in ["stop", "status"]:
        if not compose_file.exists():
            logger.error(f"Compose file not found: {compose_file}")
            logger.error("Run with --action start first to generate it")
            sys.exit(1)

        if args.action == "stop":
            stop_services(compose_file)
        elif args.action == "status":
            show_status(compose_file)

        return

    # Actions that need config
    try:
        config = load_config(config_path)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Test endpoints
    if args.action == "test":
        test_endpoints(config)
        return

    # Check GPU availability
    if not check_gpu_availability(config["gpu"]["count"]):
        logger.error("GPU check failed. Aborting.")
        sys.exit(1)

    # Generate docker-compose.yml
    logger.info(
        f"\nGenerating docker-compose configuration for {config['gpu']['count']} GPUs..."
    )
    compose_dict = generate_docker_compose(config)
    write_docker_compose(compose_dict, compose_file)

    if args.action == "generate":
        logger.info(f"\n✓ Generated {compose_file}")
        logger.info(f"Run with --action start to launch services")
        return

    # Start/restart services
    if args.action == "start":
        start_services(compose_file, build=args.build)
    elif args.action == "restart":
        stop_services(compose_file)
        start_services(compose_file, build=args.build)


if __name__ == "__main__":
    main()
