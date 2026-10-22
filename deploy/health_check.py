#!/usr/bin/env python3
"""Health check script for LDFA server deployment."""

import argparse
import sys
import requests
from typing import List


def check_endpoint(url: str, timeout: int = 10) -> dict:
    """Check a single endpoint."""
    try:
        response = requests.get(url, timeout=timeout)
        return {
            'url': url,
            'status': 'ok' if response.status_code == 200 else 'error',
            'status_code': response.status_code,
            'response_time_ms': response.elapsed.total_seconds() * 1000,
            'data': response.json() if response.status_code == 200 else None,
        }
    except requests.exceptions.Timeout:
        return {'url': url, 'status': 'timeout', 'status_code': None, 'response_time_ms': timeout * 1000}
    except requests.exceptions.ConnectionError as e:
        return {'url': url, 'status': 'connection_error', 'error': str(e)}
    except Exception as e:
        return {'url': url, 'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='LDFA Server Health Check')
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                       help='Base server URL')
    parser.add_argument('--timeout', type=int, default=10,
                       help='Request timeout in seconds')
    parser.add_argument('--check-sd-apis', action='store_true',
                       help='Also check individual SD API endpoints')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LDFA Anonymization Server Health Check")
    print("=" * 60)
    print(f"Server: {args.server}")
    print(f"Timeout: {args.timeout}s")
    print()
    
    results = []
    
    # Check main API
    print("Checking main API...")
    health_result = check_endpoint(f"{args.server}/api/health", args.timeout)
    results.append(('Main API Health', health_result))
    
    if health_result['status'] == 'ok':
        data = health_result.get('data', {})
        print(f"  Status: {data.get('status', 'unknown')}")
        print(f"  GPUs: {data.get('gpu_count', 0)}")
        print(f"  SD API Available: {data.get('sd_api_available', False)}")
        print(f"  Detectors Loaded: {data.get('detectors_loaded', False)}")
        print(f"  Content Filter: {data.get('content_filter_loaded', False)}")
    else:
        print(f"  ❌ {health_result.get('status', 'error')}")
        if 'error' in health_result:
            print(f"     Error: {health_result['error']}")
    
    print()
    
    # Check methods endpoint
    print("Checking methods endpoint...")
    methods_result = check_endpoint(f"{args.server}/api/methods", args.timeout)
    results.append(('Methods API', methods_result))
    
    if methods_result['status'] == 'ok':
        data = methods_result.get('data', {})
        print(f"  Available detectors: {data.get('detectors', [])}")
        for target, methods in data.get('targets', {}).items():
            print(f"  {target}: {methods}")
    else:
        print(f"  ❌ {methods_result['status']}")
    
    print()
    
    # Check SD API endpoints
    if args.check_sd_apis:
        print("Checking SD API endpoints...")
        sd_apis_ok = 0
        sd_apis_total = 10
        
        for i in range(10):
            port = 7860 + i
            url = f"http://localhost:{port}/sdapi/v1/options"
            result = check_endpoint(url, timeout=5)
            
            if result['status'] == 'ok':
                sd_apis_ok += 1
                status_icon = "✅"
            else:
                status_icon = "❌"
            
            print(f"  {status_icon} GPU {i} (port {port}): {result['status']}")
        
        print(f"\nSD APIs: {sd_apis_ok}/{sd_apis_total} healthy")
    
    print()
    print("=" * 60)
    
    # Summary
    all_healthy = all(r[1]['status'] == 'ok' for r in results[:2])
    
    if all_healthy:
        print("✅ All services healthy")
        return 0
    else:
        print("❌ Some services unhealthy")
        return 1


if __name__ == "__main__":
    sys.exit(main())
