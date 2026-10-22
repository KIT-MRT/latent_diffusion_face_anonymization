#!/usr/bin/env python3
"""
Generate docker-compose.yml for server deployment with configurable GPU count.

This script reads server_config.yaml and:
1. Generates docker-compose.server.yml with N GPU services for SD API
2. Adds the FastAPI anonymization service
3. Starts/stops/restarts the services
4. Verifies all endpoints are responding

Usage:
    python deploy/setup_server_compose.py --action start --gpu-count 1  # Single GPU
    python deploy/setup_server_compose.py --action start --gpu-count 10  # All 10 GPUs
    python deploy/setup_server_compose.py --action stop
    python deploy/setup_server_compose.py --action status
"""

import yaml
import subprocess
import sys
from pathlib import Path
import argparse
import logging
import time
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        logger.info("Using default configuration")
        return {
            "gpu": {
                "count": 1,
                "base_port": 7860,
            },
            "server": {
                "port": 8000,
            },
            "content_filter": {
                "enabled": True,
            }
        }
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Ensure required fields exist
    if "gpu" not in config:
        config["gpu"] = {}
    if "count" not in config["gpu"]:
        config["gpu"]["count"] = 1
    if "base_port" not in config["gpu"]:
        config["gpu"]["base_port"] = 7860
    if "server" not in config:
        config["server"] = {}
    if "port" not in config["server"]:
        config["server"]["port"] = 8000
    
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

        logger.info(f"✓ Found {available_gpus} GPUs, will use {gpu_count}")
        return True

    except subprocess.CalledProcessError:
        logger.error("Failed to run nvidia-smi. Are NVIDIA drivers installed?")
        return False
    except FileNotFoundError:
        logger.error("nvidia-smi not found. Are NVIDIA drivers installed?")
        return False


def generate_docker_compose(gpu_count: int, base_port: int = 7860, server_port: int = 8000) -> dict:
    """Generate docker-compose configuration."""
    
    services = {}

    import os
    owner = os.environ.get("CONTAINER_OWNER", os.environ.get("USER", "unknown"))

    # Generate SD API services for each GPU
    for i in range(gpu_count):
        port = base_port + i
        service_name = f"sd-api-gpu{i}"

        logger.info(f"  Creating service: {service_name} on port {port} for GPU {i}")

        services[service_name] = {
            "image": "ldfa",
            "container_name": service_name,
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile"
            },
            "ports": [f"{port}:7860"],
            "environment": {
                "CUDA_VISIBLE_DEVICES": str(i),
                "CONTAINER_OWNER": owner,
            },
            "shm_size": "16g",
            "restart": "unless-stopped",
            "entrypoint": "python3 webui.py --listen --api --xformers",
            "healthcheck": {
                "test": ["CMD", "curl", "-f", f"http://localhost:7860/sdapi/v1/options"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3,
                "start_period": "60s"
            },
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "device_ids": [str(i)],
                                "capabilities": ["gpu"]
                            }
                        ]
                    }
                }
            }
        }
    
    # FastAPI anonymization service. Uses the pre-built ldfa-api image
    # (run `bash server/build.sh` first — the build needs an HF token as a
    # BuildKit secret, which compose can't pass through). Shares GPU 0 with
    # sd-api-gpu0; SAM3 is a few GB, SD is a few GB, they coexist fine on a
    # 48GB card. Image CMD already pins `--workers 1` (one SAM3 copy).
    # CONTAINER_OWNER identifies the container to the lab-host reaper that
    # sweeps unowned containers. Already set on sd-api-gpu* above.
    # SD_API_HOST_TEMPLATE / SD_API_PORT point the endpoint pool at the sd-api
    # service names on the compose network (each SD container binds 7860
    # internally; the host-port mapping is only for external access).
    services["anonymization-api"] = {
        "image": "ldfa-api",
        "container_name": "ldfa-anonymization-api",
        "ports": [f"{server_port}:8000"],
        "environment": {
            "CONTAINER_OWNER": owner,
            "SD_API_HOST_TEMPLATE": "sd-api-gpu{gpu}",
            "SD_API_PORT": "7860",
        },
        # Config is baked into the image (edit server/config/*.yaml + rebuild
        # — the COPY server/ layer rebuilds in seconds). No bind mount, so the
        # compose file is safe to place anywhere.
        "volumes": [
            "./logs:/var/log/anonymization"
        ],
        "depends_on": [f"sd-api-gpu0"],
        "restart": "unless-stopped",
        "healthcheck": {
            "test": ["CMD", "curl", "-f", "http://localhost:8000/api/health"],
            "interval": "30s",
            "timeout": "10s",
            "retries": 3,
            "start_period": "60s"
        },
        "deploy": {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "device_ids": ["0"],
                            "capabilities": ["gpu"]
                        }
                    ]
                }
            }
        }
    }
    
    return {
        "version": "3.8",
        "services": services,
        "networks": {
            "ldfa-network": {
                "driver": "bridge"
            }
        }
    }


def write_docker_compose(compose_dict: dict, output_path: Path):
    """Write docker-compose dict to YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(compose_dict, f, default_flow_style=False, sort_keys=False)
    logger.info(f"✓ Generated {output_path}")


def start_services(compose_file: Path, build: bool = False, gpu_count: int = 1):
    """Start all docker-compose services."""
    logger.info(f"Starting services for {gpu_count} GPU(s)...")
    
    cmd = [
        "docker-compose",
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
        logger.info(f"\n⏳ Wait ~30-60 seconds for Stable Diffusion to load...")
        logger.info(
            f"Check status with: docker-compose -f {compose_file} logs -f"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start services: {e}")
        sys.exit(1)


def stop_services(compose_file: Path):
    """Stop all docker-compose services."""
    logger.info("Stopping all services...")
    
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


def test_endpoints(gpu_count: int, base_port: int = 7860, server_port: int = 8000):
    """Test if all API endpoints are responding."""
    logger.info("\nTesting API endpoints...")
    
    all_ok = True
    
    # Test SD API endpoints
    for i in range(gpu_count):
        port = base_port + i
        url = f"http://127.0.0.1:{port}/sdapi/v1/options"
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"  ✓ SD API GPU {i} (port {port}): OK")
            else:
                logger.warning(f"  ✗ SD API GPU {i} (port {port}): HTTP {response.status_code}")
                all_ok = False
        except requests.exceptions.ConnectionError:
            logger.warning(f"  ✗ SD API GPU {i} (port {port}): Connection refused")
            all_ok = False
        except requests.exceptions.Timeout:
            logger.warning(f"  ✗ SD API GPU {i} (port {port}): Timeout")
            all_ok = False
        except Exception as e:
            logger.warning(f"  ✗ SD API GPU {i} (port {port}): {e}")
            all_ok = False
    
    # Test FastAPI server
    url = f"http://127.0.0.1:{server_port}/api/health"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"  ✓ FastAPI (port {server_port}): OK - {data.get('status', 'unknown')}")
        else:
            logger.warning(f"  ✗ FastAPI (port {server_port}): HTTP {response.status_code}")
            all_ok = False
    except Exception as e:
        logger.warning(f"  ✗ FastAPI (port {server_port}): {e}")
        all_ok = False
    
    if all_ok:
        logger.info("\n✅ All endpoints are responding!")
    else:
        logger.warning(
            "\n⚠ Some endpoints are not responding yet. "
            "Services may still be starting up (can take 30-60 seconds)."
        )
    
    return all_ok


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup and manage server deployment with configurable GPU count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with 1 GPU (for testing)
    python deploy/setup_server_compose.py --action start --gpu-count 1
    
    # Start with all 10 GPUs
    python deploy/setup_server_compose.py --action start --gpu-count 10
    
    # Generate config without starting
    python deploy/setup_server_compose.py --action generate --gpu-count 5
    
    # Stop all services
    python deploy/setup_server_compose.py --action stop
    
    # Restart with rebuild
    python deploy/setup_server_compose.py --action restart --build
    
    # Show status
    python deploy/setup_server_compose.py --action status
    
    # Test endpoints
    python deploy/setup_server_compose.py --action test --gpu-count 10
        """,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="server/config/server_config.yaml",
        help="Path to config file (default: server/config/server_config.yaml)",
    )
    parser.add_argument(
        "--action",
        choices=["start", "stop", "restart", "status", "test", "generate"],
        default="start",
        help="Action to perform (default: start)",
    )
    parser.add_argument(
        "--build", 
        action="store_true", 
        help="Rebuild Docker images when starting"
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Number of GPUs to use (overrides config file)",
    )
    parser.add_argument(
        "--compose-file",
        type=str,
        default="docker-compose.server.yml",
        help="Output docker-compose file (default: docker-compose.server.yml)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=7860,
        help="Base port for SD API (default: 7860)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Port for FastAPI server (default: 8000)",
    )
    
    args = parser.parse_args()
    
    # Paths
    config_path = Path(args.config)
    compose_file = Path(args.compose_file)
    
    # Load config
    config = load_config(config_path)
    
    # Override with CLI args
    gpu_count = args.gpu_count if args.gpu_count is not None else config["gpu"]["count"]
    base_port = args.base_port if args.base_port != 7860 else config["gpu"]["base_port"]
    server_port = args.server_port if args.server_port != 8000 else config["server"]["port"]
    
    # Actions that don't need GPU check
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
    
    # Check GPU availability for start/restart/generate
    if args.action in ["start", "restart", "generate"]:
        if not check_gpu_availability(gpu_count):
            logger.error("GPU check failed. Aborting.")
            sys.exit(1)
    
    # Generate docker-compose.yml
    logger.info(f"\nGenerating docker-compose configuration for {gpu_count} GPU(s)...")
    compose_dict = generate_docker_compose(gpu_count, base_port, server_port)
    write_docker_compose(compose_dict, compose_file)
    
    if args.action == "generate":
        logger.info(f"\n✓ Generated {compose_file}")
        logger.info(f"Run with --action start to launch services")
        return
    
    # Start/restart services
    if args.action == "start":
        start_services(compose_file, build=args.build, gpu_count=gpu_count)
        # Wait for startup
        time.sleep(30)
        test_endpoints(gpu_count, base_port, server_port)
    elif args.action == "restart":
        stop_services(compose_file)
        time.sleep(5)
        start_services(compose_file, build=args.build, gpu_count=gpu_count)
        time.sleep(30)
        test_endpoints(gpu_count, base_port, server_port)
    elif args.action == "test":
        test_endpoints(gpu_count, base_port, server_port)


if __name__ == "__main__":
    main()
