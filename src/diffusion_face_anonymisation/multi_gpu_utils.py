"""
Multi-GPU utilities for distributed stable diffusion API calls.

This module provides load balancing and request distribution across multiple
GPU-backed Stable Diffusion API endpoints.
"""
import queue
import threading
import logging
import requests
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    """Record of a single API request for batch tracking."""
    request_id: int
    image_file: str
    object_type: str  # 'face', 'body', 'lp'
    object_index: int  # Which object in the image (0, 1, 2, ...)
    endpoint: str
    timestamp_start: float
    timestamp_end: Optional[float] = None
    duration_seconds: Optional[float] = None
    status: str = 'pending'  # 'pending', 'success', 'failed', 'retrying'
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class BatchStatistics:
    """Statistics for entire batch processing."""
    total_images: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retry_requests: int = 0
    
    total_faces: int = 0
    total_bodies: int = 0
    total_plates: int = 0
    
    detection_time_seconds: float = 0.0
    lda_time_seconds: float = 0.0
    composition_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    
    throughput_images_per_sec: float = 0.0
    throughput_requests_per_sec: float = 0.0
    
    endpoint_stats: Dict[str, Dict] = field(default_factory=dict)
    request_records: List[RequestRecord] = field(default_factory=list)
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            'summary': {
                'total_images': self.total_images,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'retry_requests': self.retry_requests,
                'success_rate': f"{self.successful_requests / max(self.total_requests, 1) * 100:.1f}%",
            },
            'detection_counts': {
                'faces': self.total_faces,
                'bodies': self.total_bodies,
                'license_plates': self.total_plates,
            },
            'timing': {
                'detection_seconds': round(self.detection_time_seconds, 2),
                'lda_seconds': round(self.lda_time_seconds, 2),
                'composition_seconds': round(self.composition_time_seconds, 2),
                'total_seconds': round(self.total_time_seconds, 2),
                'throughput_images_per_sec': round(self.throughput_images_per_sec, 2),
                'throughput_requests_per_sec': round(self.throughput_requests_per_sec, 2),
            },
            'endpoint_statistics': self.endpoint_stats,
            'request_log': [
                {
                    'request_id': r.request_id,
                    'image': Path(r.image_file).name,
                    'type': r.object_type,
                    'index': r.object_index,
                    'endpoint': r.endpoint,
                    'duration_ms': round(r.duration_seconds * 1000, 1) if r.duration_seconds else None,
                    'status': r.status,
                    'retries': r.retry_count,
                    'error': r.error_message,
                }
                for r in self.request_records
            ],
            'timestamps': {
                'start': self.start_time.isoformat() if self.start_time else None,
                'end': self.end_time.isoformat() if self.end_time else None,
            }
        }
    
    def save_to_file(self, output_path: Path):
        """Save statistics to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Batch statistics saved to {output_path}")


class APIEndpointPool:
    """
    Manages pool of SD API endpoints and distributes requests across them.
    Uses round-robin + queue-based retry strategy.
    
    Features:
    - Round-robin load balancing
    - Health tracking per endpoint
    - Automatic retry with exponential backoff
    - Session pooling for connection reuse
    - Detailed request logging
    """
    
    def __init__(
        self,
        base_port: int,
        gpu_count: int,
        timeout: int = 120,
        retry_attempts: int = 3,
        track_requests: bool = True
    ):
        """
        Initialize endpoint pool.
        
        Args:
            base_port: Starting port (e.g., 7860)
            gpu_count: Number of GPUs/endpoints
            timeout: Request timeout in seconds
            retry_attempts: Max retry attempts per request
            track_requests: Enable detailed request tracking
        """
        self.endpoints = [
            f"http://127.0.0.1:{base_port + i}/sdapi/v1/img2img"
            for i in range(gpu_count)
        ]
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.track_requests = track_requests
        
        self.current_index = 0
        self.lock = threading.Lock()
        
        # Retry queue for failed requests
        self.retry_queue = queue.Queue()
        
        # Session pool (one per endpoint for connection reuse)
        self.sessions = {
            endpoint: requests.Session()
            for endpoint in self.endpoints
        }
        
        # Health tracking
        self.endpoint_failures = {endpoint: 0 for endpoint in self.endpoints}
        self.endpoint_success = {endpoint: 0 for endpoint in self.endpoints}
        self.endpoint_total_time = {endpoint: 0.0 for endpoint in self.endpoints}
        
        # Request tracking
        self.request_counter = 0
        self.request_records: List[RequestRecord] = []
        self.request_lock = threading.Lock()
        
        logger.info(f"Initialized API pool with {gpu_count} endpoints: {self.endpoints}")
    
    def get_next_endpoint(self) -> str:
        """
        Get next endpoint using round-robin strategy.
        Skips endpoints with high failure rate.
        """
        with self.lock:
            # Try up to len(endpoints) times to find healthy endpoint
            for _ in range(len(self.endpoints)):
                endpoint = self.endpoints[self.current_index]
                self.current_index = (self.current_index + 1) % len(self.endpoints)
                
                # Skip if failure rate > 80%
                total = self.endpoint_failures[endpoint] + self.endpoint_success[endpoint]
                if total > 10:  # Only check after 10 requests
                    failure_rate = self.endpoint_failures[endpoint] / total
                    if failure_rate > 0.8:
                        logger.warning(
                            f"Skipping unhealthy endpoint {endpoint} "
                            f"(failure rate: {failure_rate:.1%})"
                        )
                        continue
                
                return endpoint
            
            # All endpoints unhealthy, return first anyway
            logger.error("All endpoints appear unhealthy, using first endpoint")
            return self.endpoints[0]
    
    def create_request_record(
        self,
        image_file: str,
        object_type: str,
        object_index: int,
        endpoint: str
    ) -> RequestRecord:
        """Create a new request record for tracking."""
        with self.request_lock:
            self.request_counter += 1
            record = RequestRecord(
                request_id=self.request_counter,
                image_file=image_file,
                object_type=object_type,
                object_index=object_index,
                endpoint=endpoint,
                timestamp_start=time.time()
            )
            self.request_records.append(record)
            return record
    
    def update_request_record(
        self,
        record: RequestRecord,
        status: str,
        error_message: Optional[str] = None
    ):
        """Update request record with completion status."""
        record.timestamp_end = time.time()
        record.duration_seconds = record.timestamp_end - record.timestamp_start
        record.status = status
        if error_message:
            record.error_message = error_message
    
    def send_request(
        self,
        payload: dict,
        image_file: str = "unknown",
        object_type: str = "unknown",
        object_index: int = 0
    ) -> str:
        """
        Send request to next available endpoint with retry logic.
        
        Args:
            payload: SD API payload dict
            image_file: Source image file (for tracking)
            object_type: Type of object being anonymized (for tracking)
            object_index: Index of object in image (for tracking)
            
        Returns:
            Base64-encoded image string
            
        Raises:
            RuntimeError: If all retries fail
        """
        last_error = None
        
        for attempt in range(self.retry_attempts):
            endpoint = self.get_next_endpoint()
            session = self.sessions[endpoint]
            
            # Create tracking record
            record = None
            if self.track_requests:
                record = self.create_request_record(
                    image_file, object_type, object_index, endpoint
                )
            
            try:
                logger.debug(
                    f"Sending {object_type} request to {endpoint} "
                    f"(attempt {attempt + 1}/{self.retry_attempts})"
                )
                
                start_time = time.time()
                response = session.post(
                    url=endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    # Success
                    with self.lock:
                        self.endpoint_success[endpoint] += 1
                        self.endpoint_total_time[endpoint] += elapsed
                    
                    if record:
                        self.update_request_record(record, 'success')
                        if attempt > 0:
                            record.retry_count = attempt
                    
                    logger.debug(f"Request completed in {elapsed:.2f}s on {endpoint}")
                    
                    response_json = response.json()
                    return response_json["images"][0]
                else:
                    # Non-200 status
                    with self.lock:
                        self.endpoint_failures[endpoint] += 1
                    
                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(f"Request to {endpoint} failed: {error_msg}")
                    last_error = error_msg
                    
                    if record:
                        record.retry_count = attempt
                        if attempt == self.retry_attempts - 1:
                            self.update_request_record(record, 'failed', error_msg)
                    
            except requests.exceptions.Timeout:
                with self.lock:
                    self.endpoint_failures[endpoint] += 1
                
                error_msg = f"Timeout after {self.timeout}s"
                logger.warning(f"Request to {endpoint} timed out")
                last_error = error_msg
                
                if record:
                    record.retry_count = attempt
                    if attempt == self.retry_attempts - 1:
                        self.update_request_record(record, 'failed', error_msg)
                
            except Exception as e:
                with self.lock:
                    self.endpoint_failures[endpoint] += 1
                
                error_msg = str(e)
                logger.warning(f"Request to {endpoint} failed: {error_msg}")
                last_error = error_msg
                
                if record:
                    record.retry_count = attempt
                    if attempt == self.retry_attempts - 1:
                        self.update_request_record(record, 'failed', error_msg)
            
            # Exponential backoff before retry
            if attempt < self.retry_attempts - 1:
                backoff = 2 ** attempt
                logger.debug(f"Retrying in {backoff}s...")
                time.sleep(backoff)
        
        # All retries exhausted
        raise RuntimeError(
            f"Failed to send request after {self.retry_attempts} attempts. "
            f"Last error: {last_error}"
        )
    
    def get_stats(self) -> dict:
        """Get endpoint usage statistics."""
        with self.lock:
            stats = {}
            for endpoint in self.endpoints:
                total = self.endpoint_failures[endpoint] + self.endpoint_success[endpoint]
                success_rate = (
                    self.endpoint_success[endpoint] / total if total > 0 else 0
                )
                avg_time = (
                    self.endpoint_total_time[endpoint] / self.endpoint_success[endpoint]
                    if self.endpoint_success[endpoint] > 0
                    else 0
                )
                stats[endpoint] = {
                    'total_requests': total,
                    'successes': self.endpoint_success[endpoint],
                    'failures': self.endpoint_failures[endpoint],
                    'success_rate': f"{success_rate * 100:.1f}%",
                    'avg_response_time_seconds': round(avg_time, 2),
                }
            return stats
    
    def get_batch_statistics(self) -> BatchStatistics:
        """Get comprehensive batch statistics."""
        batch_stats = BatchStatistics()
        
        # Populate endpoint stats
        batch_stats.endpoint_stats = self.get_stats()
        
        # Populate request records
        batch_stats.request_records = self.request_records.copy()
        
        # Calculate aggregates
        batch_stats.total_requests = len(self.request_records)
        batch_stats.successful_requests = sum(
            1 for r in self.request_records if r.status == 'success'
        )
        batch_stats.failed_requests = sum(
            1 for r in self.request_records if r.status == 'failed'
        )
        batch_stats.retry_requests = sum(
            1 for r in self.request_records if r.retry_count > 0
        )
        
        # Count object types
        batch_stats.total_faces = sum(
            1 for r in self.request_records if r.object_type == 'face'
        )
        batch_stats.total_bodies = sum(
            1 for r in self.request_records if r.object_type == 'body'
        )
        batch_stats.total_plates = sum(
            1 for r in self.request_records if r.object_type == 'lp'
        )
        
        return batch_stats
    
    def print_stats(self):
        """Print formatted statistics to console."""
        stats = self.get_stats()
        
        print("\n" + "="*70)
        print("MULTI-GPU ENDPOINT STATISTICS")
        print("="*70)
        
        for endpoint, endpoint_stats in stats.items():
            print(f"\n{endpoint}:")
            print(f"  Total requests:     {endpoint_stats['total_requests']}")
            print(f"  Success rate:       {endpoint_stats['success_rate']}")
            print(f"  Successes:          {endpoint_stats['successes']}")
            print(f"  Failures:           {endpoint_stats['failures']}")
            print(f"  Avg response time:  {endpoint_stats['avg_response_time_seconds']}s")
        
        print("\n" + "="*70)
    
    def close(self):
        """Close all sessions."""
        for session in self.sessions.values():
            session.close()
        logger.info("Closed all API endpoint sessions")
