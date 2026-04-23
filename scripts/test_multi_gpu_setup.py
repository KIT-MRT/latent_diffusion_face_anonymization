#!/usr/bin/env python3
"""
Test script to verify multi-GPU setup without requiring Docker containers.

This script tests:
1. Configuration loading
2. APIEndpointPool initialization
3. Request tracking
4. Statistics generation
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from diffusion_face_anonymisation.multi_gpu_utils import APIEndpointPool, BatchStatistics
import yaml

def test_config_loading():
    """Test configuration loading."""
    print("="*70)
    print("TEST 1: Configuration Loading")
    print("="*70)
    
    config_path = Path(__file__).parent.parent / 'config.yaml'
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Config loaded successfully")
    print(f"  GPUs: {config['gpu']['count']}")
    print(f"  Base port: {config['gpu']['base_port']}")
    print(f"  Worker threads: {config['api'].get('worker_threads', 'not set')}")
    
    return True


def test_endpoint_pool():
    """Test APIEndpointPool initialization."""
    print("\n" + "="*70)
    print("TEST 2: APIEndpointPool Initialization")
    print("="*70)
    
    try:
        pool = APIEndpointPool(
            base_port=7860,
            gpu_count=2,
            timeout=120,
            track_requests=True
        )
        
        print(f"✓ APIEndpointPool created successfully")
        print(f"  Endpoints: {pool.endpoints}")
        print(f"  Sessions: {len(pool.sessions)} connections")
        
        # Test round-robin
        print("\n  Testing round-robin distribution:")
        for i in range(5):
            endpoint = pool.get_next_endpoint()
            print(f"    Request {i+1} → {endpoint}")
        
        pool.close()
        print(f"\n✓ Pool closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_statistics():
    """Test BatchStatistics."""
    print("\n" + "="*70)
    print("TEST 3: Batch Statistics")
    print("="*70)
    
    try:
        stats = BatchStatistics()
        stats.total_images = 100
        stats.total_requests = 250
        stats.successful_requests = 245
        stats.failed_requests = 5
        stats.total_faces = 80
        stats.total_bodies = 120
        stats.total_plates = 50
        stats.detection_time_seconds = 30.5
        stats.lda_time_seconds = 450.2
        stats.composition_time_seconds = 15.3
        stats.total_time_seconds = 496.0
        
        print(f"✓ BatchStatistics created")
        print(f"  Total images: {stats.total_images}")
        print(f"  Total requests: {stats.total_requests}")
        print(f"  Success rate: {stats.successful_requests / stats.total_requests * 100:.1f}%")
        
        # Test dict conversion
        stats_dict = stats.to_dict()
        print(f"\n✓ Stats converted to dict")
        print(f"  Keys: {list(stats_dict.keys())}")
        
        # Test file save
        test_output = Path("/tmp/test_batch_stats.json")
        stats.save_to_file(test_output)
        print(f"\n✓ Stats saved to {test_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_request_tracking():
    """Test request tracking functionality."""
    print("\n" + "="*70)
    print("TEST 4: Request Tracking")
    print("="*70)
    
    try:
        pool = APIEndpointPool(
            base_port=7860,
            gpu_count=2,
            timeout=120,
            track_requests=True
        )
        
        # Simulate creating request records
        for i in range(5):
            endpoint = pool.get_next_endpoint()
            record = pool.create_request_record(
                image_file=f"test_image_{i}.png",
                object_type='body',
                object_index=i,
                endpoint=endpoint
            )
            
            # Simulate completion
            pool.update_request_record(record, 'success')
        
        print(f"✓ Created {len(pool.request_records)} request records")
        
        # Get stats
        endpoint_stats = pool.get_stats()
        print(f"\n✓ Endpoint statistics:")
        for endpoint, stats in endpoint_stats.items():
            print(f"    {endpoint}: {stats}")
        
        # Get batch stats
        batch_stats = pool.get_batch_statistics()
        print(f"\n✓ Batch statistics:")
        print(f"    Total requests: {batch_stats.total_requests}")
        print(f"    Successful: {batch_stats.successful_requests}")
        print(f"    Bodies tracked: {batch_stats.total_bodies}")
        
        pool.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "🧪 MULTI-GPU SETUP TESTS".center(70))
    print("="*70)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("APIEndpointPool", test_endpoint_pool),
        ("Batch Statistics", test_batch_statistics),
        ("Request Tracking", test_request_tracking),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8s} - {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
