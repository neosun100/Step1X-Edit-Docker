"""
GPU Resource Manager for Step1X-Edit
=====================================

Intelligent GPU memory management with lazy loading and instant offloading.

Features:
- Lazy loading: Load model to GPU on first request
- Instant offload: Move to CPU after each task (2-5s to reload)
- Auto-monitoring: Move to CPU after idle timeout
- Manual control: Force offload/release when needed

State transitions:
    Unloaded ──first_request(20-30s)──> GPU ──task_complete(2s)──> CPU
       ↑                                                            ↓
       └──────────────timeout/manual_release(1s)────────────────────┘
"""

import time
import threading
import logging
from typing import Callable, Optional, Any
import torch
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUResourceManager:
    """
    GPU Resource Manager with lazy loading and instant offloading.

    Usage:
        # Initialize
        gpu_manager = GPUResourceManager(idle_timeout=60)

        # Define model loading function
        def load_model():
            return YourModel.from_pretrained('model-name')

        # Use in processing
        def process_task(input_data):
            try:
                # Step 1: Lazy load
                model = gpu_manager.get_model(load_func=load_model)

                # Step 2: Process
                result = model(input_data)

                # Step 3: Instant offload (CRITICAL!)
                gpu_manager.force_offload()

                return result
            except Exception as e:
                gpu_manager.force_offload()
                raise e
    """

    def __init__(
        self,
        idle_timeout: int = 60,
        device: str = "cuda",
        offload_delay: float = 0.0,
        auto_monitor: bool = True
    ):
        """
        Initialize GPU Resource Manager.

        Args:
            idle_timeout: Idle timeout in seconds before auto-offload (default: 60)
            device: CUDA device to use (default: "cuda")
            offload_delay: Delay after offload in seconds (default: 0.0)
            auto_monitor: Enable automatic monitoring thread (default: True)
        """
        self.model_on_gpu = None      # Model on GPU
        self.model_on_cpu = None      # Model cached on CPU
        self.load_func = None          # Function to load model

        self.device = torch.device(device)
        self.idle_timeout = idle_timeout
        self.offload_delay = offload_delay
        self.last_use_time = time.time()

        self.lock = threading.Lock()
        self.monitor_thread = None
        self.running = False

        # Statistics
        self.stats = {
            'total_loads': 0,
            'gpu_to_cpu': 0,
            'cpu_to_gpu': 0,
            'full_releases': 0
        }

        if auto_monitor:
            self.start_monitor()

    def get_model(self, load_func: Callable[[], Any]) -> Any:
        """
        Get model with lazy loading.

        Behavior:
        1. If on GPU -> return directly
        2. If on CPU -> quickly move to GPU (2-5s)
        3. If unloaded -> load from disk (first time, 20-30s)

        Args:
            load_func: Function that returns the model (called only on first load)

        Returns:
            Model on GPU, ready for inference
        """
        with self.lock:
            self.last_use_time = time.time()

            # Case 1: Already on GPU
            if self.model_on_gpu is not None:
                logger.info("✓ Model already on GPU")
                return self.model_on_gpu

            # Case 2: Cached on CPU, quick transfer (2-5s)
            if self.model_on_cpu is not None:
                logger.info("⟳ Moving model from CPU to GPU (2-5s)...")
                start_time = time.time()

                self.model_on_gpu = self._move_to_gpu(self.model_on_cpu)

                elapsed = time.time() - start_time
                self.stats['cpu_to_gpu'] += 1
                logger.info(f"✓ Model moved to GPU in {elapsed:.2f}s")
                return self.model_on_gpu

            # Case 3: First load from disk (20-30s)
            logger.info("⟳ First load: Loading model from disk (20-30s)...")
            start_time = time.time()

            self.load_func = load_func
            model = load_func()

            # Move to GPU
            self.model_on_gpu = self._move_to_gpu(model)

            elapsed = time.time() - start_time
            self.stats['total_loads'] += 1
            logger.info(f"✓ Model loaded and moved to GPU in {elapsed:.2f}s")

            return self.model_on_gpu

    def force_offload(self):
        """
        Instant offload: Move model from GPU to CPU after task completion.

        This should be called immediately after each task to free GPU memory.
        Transfer time: ~2s
        """
        with self.lock:
            if self.model_on_gpu is None:
                logger.debug("Model not on GPU, skip offload")
                return

            logger.info("⟳ Offloading model to CPU...")
            start_time = time.time()

            # Move to CPU
            self.model_on_cpu = self._move_to_cpu(self.model_on_gpu)
            self.model_on_gpu = None

            # Clear GPU cache
            self._clear_gpu_cache()

            # Optional delay
            if self.offload_delay > 0:
                time.sleep(self.offload_delay)

            elapsed = time.time() - start_time
            self.stats['gpu_to_cpu'] += 1
            logger.info(f"✓ Model offloaded to CPU in {elapsed:.2f}s")

            # Log GPU memory
            self._log_gpu_memory()

    def force_release(self):
        """
        Complete release: Clear both GPU and CPU cache.

        Use when model won't be needed for a long time.
        Next request will require full reload from disk.
        """
        with self.lock:
            logger.info("⟳ Releasing all model resources...")
            start_time = time.time()

            # Clear both caches
            self.model_on_gpu = None
            self.model_on_cpu = None

            # Clear GPU cache
            self._clear_gpu_cache()

            elapsed = time.time() - start_time
            self.stats['full_releases'] += 1
            logger.info(f"✓ All resources released in {elapsed:.2f}s")

            # Log GPU memory
            self._log_gpu_memory()

    def get_status(self) -> dict:
        """
        Get current status of GPU manager.

        Returns:
            Dictionary with status information
        """
        with self.lock:
            idle_time = time.time() - self.last_use_time

            # Determine model location
            if self.model_on_gpu is not None:
                location = "GPU"
            elif self.model_on_cpu is not None:
                location = "CPU"
            else:
                location = "Unloaded"

            # Get GPU memory usage
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            else:
                gpu_memory_allocated = 0
                gpu_memory_reserved = 0

            return {
                'model_location': location,
                'idle_time': idle_time,
                'idle_timeout': self.idle_timeout,
                'gpu_memory_allocated_gb': round(gpu_memory_allocated, 2),
                'gpu_memory_reserved_gb': round(gpu_memory_reserved, 2),
                'auto_monitor_running': self.running,
                'statistics': self.stats.copy()
            }

    def start_monitor(self):
        """Start automatic monitoring thread."""
        if self.running:
            logger.warning("Monitor already running")
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"✓ Auto-monitor started (timeout: {self.idle_timeout}s)")

    def stop_monitor(self):
        """Stop automatic monitoring thread."""
        if not self.running:
            return

        self.running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=5)
        logger.info("✓ Auto-monitor stopped")

    def _monitor_loop(self):
        """Monitoring loop that runs in background thread."""
        while self.running:
            time.sleep(30)  # Check every 30 seconds

            with self.lock:
                if self.model_on_gpu is not None:
                    idle_time = time.time() - self.last_use_time

                    # Auto-offload if idle timeout reached
                    if idle_time > self.idle_timeout:
                        logger.info(f"Idle for {idle_time:.0f}s, auto-offloading...")
                        self._offload_internal()

    def _offload_internal(self):
        """Internal offload without acquiring lock (lock must be held by caller)."""
        if self.model_on_gpu is None:
            return

        # Move to CPU
        self.model_on_cpu = self._move_to_cpu(self.model_on_gpu)
        self.model_on_gpu = None

        # Clear GPU cache
        self._clear_gpu_cache()

        self.stats['gpu_to_cpu'] += 1
        logger.info("✓ Auto-offload completed")

    def _move_to_gpu(self, model: Any) -> Any:
        """Move model to GPU."""
        if hasattr(model, 'to'):
            return model.to(device=self.device)
        return model

    def _move_to_cpu(self, model: Any) -> Any:
        """Move model to CPU."""
        if hasattr(model, 'to'):
            return model.to(device='cpu')
        return model

    def _clear_gpu_cache(self):
        """Clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()

    def _log_gpu_memory(self):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_monitor()


# Global singleton instance (optional)
_global_manager: Optional[GPUResourceManager] = None


def get_global_gpu_manager(
    idle_timeout: int = 60,
    device: str = "cuda",
    **kwargs
) -> GPUResourceManager:
    """
    Get or create global GPU manager instance.

    Args:
        idle_timeout: Idle timeout in seconds
        device: CUDA device
        **kwargs: Additional arguments for GPUResourceManager

    Returns:
        Global GPUResourceManager instance
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = GPUResourceManager(
            idle_timeout=idle_timeout,
            device=device,
            **kwargs
        )

    return _global_manager


# Example usage
if __name__ == "__main__":
    # Example with mock model
    class MockModel:
        def __init__(self):
            self.data = torch.randn(1000, 1000)

        def to(self, device):
            print(f"Moving to {device}")
            self.data = self.data.to(device)
            return self

        def __call__(self, x):
            return self.data @ x

    # Initialize manager
    manager = GPUResourceManager(idle_timeout=10)

    # Define load function
    def load_model():
        print("Loading model...")
        time.sleep(2)  # Simulate loading
        return MockModel()

    # Simulate processing
    print("\n=== First request (full load) ===")
    model = manager.get_model(load_func=load_model)
    result = model(torch.randn(1000, 1))
    manager.force_offload()

    print("\n=== Second request (quick reload from CPU) ===")
    time.sleep(1)
    model = manager.get_model(load_func=load_model)
    result = model(torch.randn(1000, 1))
    manager.force_offload()

    print("\n=== Status ===")
    print(manager.get_status())

    print("\n=== Full release ===")
    manager.force_release()
    print(manager.get_status())
