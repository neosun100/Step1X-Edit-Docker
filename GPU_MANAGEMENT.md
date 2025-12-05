# GPU Memory Management Guide

## Overview

Step1X-Edit implements intelligent GPU memory management with **lazy loading** and **instant offloading** to optimize resource usage.

## State Transitions

```
Unloaded ──first_request(20-30s)──> GPU ──task_complete(2s)──> CPU ──new_request(2-5s)──> GPU
   ↑                                                              ↓
   └──────────────timeout/manual_release(1s)──────────────────────┘
```

## Features

### 1. Lazy Loading
- Model loads to GPU only on first request
- Subsequent requests reload from CPU cache (2-5s)
- Minimizes idle GPU memory usage

### 2. Instant Offload
- Automatically moves model to CPU after each task
- Frees GPU memory immediately
- Keeps model in RAM for quick reload

### 3. Auto-Monitoring
- Background thread monitors idle time
- Auto-offloads after configurable timeout (default: 60s)
- Prevents memory leaks

### 4. Manual Control
- Force offload: Move to CPU, keep in RAM
- Force release: Clear both GPU and CPU cache

## Configuration

### Environment Variables

```bash
# GPU idle timeout (seconds)
GPU_IDLE_TIMEOUT=60

# GPU device ID (auto-selected by default)
NVIDIA_VISIBLE_DEVICES=0
```

### Programmatic Configuration

```python
from gpu_manager import GPUResourceManager

manager = GPUResourceManager(
    idle_timeout=60,        # Auto-offload after 60s idle
    device="cuda",          # CUDA device
    offload_delay=0.0,      # Delay after offload
    auto_monitor=True       # Enable auto-monitoring
)
```

## Usage Patterns

### Standard Processing Flow

```python
def process_task(input_data):
    try:
        # Step 1: Lazy load (auto-managed)
        model = gpu_manager.get_model(load_func=load_model)
        
        # Step 2: Process
        result = model(input_data)
        
        # Step 3: Instant offload (CRITICAL!)
        gpu_manager.force_offload()
        
        return result
        
    except Exception as e:
        # Always offload on error
        gpu_manager.force_offload()
        raise e
```

### Batch Processing

```python
def batch_process(items):
    results = []
    
    for item in items:
        # Model loads once, stays on GPU during batch
        model = gpu_manager.get_model(load_func=load_model)
        result = model(item)
        results.append(result)
    
    # Offload after entire batch
    gpu_manager.force_offload()
    
    return results
```

### Long-Term Idle

```python
# When done for extended period
gpu_manager.force_release()  # Clear both GPU and CPU
```

## API Endpoints

### Get GPU Status

```bash
GET /api/gpu/status
```

Response:
```json
{
  "model_location": "CPU",
  "idle_time": 45.2,
  "gpu_memory_allocated_gb": 0.12,
  "gpu_memory_reserved_gb": 0.5,
  "statistics": {
    "total_loads": 5,
    "gpu_to_cpu": 5,
    "cpu_to_gpu": 4,
    "full_releases": 1
  }
}
```

### Manual Offload

```bash
POST /api/gpu/offload
```

Moves model from GPU to CPU, freeing GPU memory.

### Complete Release

```bash
POST /api/gpu/release
```

Clears both GPU and CPU cache.

## Performance Metrics

| Operation | Time | GPU Memory |
|-----------|------|------------|
| First load (disk → GPU) | 20-30s | ~40GB |
| Reload (CPU → GPU) | 2-5s | ~40GB |
| Offload (GPU → CPU) | ~2s | <1GB |
| Release (clear all) | ~1s | <1GB |

## Memory Usage

### Without Management
- Idle: 40-50GB GPU memory
- Processing: 40-50GB GPU memory

### With Management
- Idle: <1GB GPU memory
- Processing: 40-50GB GPU memory (only during task)
- Between tasks: <1GB GPU memory

## Monitoring

### Check GPU Memory

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Memory usage
nvidia-smi --query-gpu=memory.used --format=csv
```

### Check Manager Status

```python
status = gpu_manager.get_status()
print(f"Model location: {status['model_location']}")
print(f"Idle time: {status['idle_time']}s")
print(f"GPU memory: {status['gpu_memory_allocated_gb']}GB")
```

## Troubleshooting

### High GPU Memory When Idle

**Symptom**: GPU memory stays high after processing

**Solution**:
1. Check if `force_offload()` is called after each task
2. Verify auto-monitor is running: `manager.running == True`
3. Check idle timeout: `manager.idle_timeout`

### Slow Reload

**Symptom**: Reload takes >10s

**Solution**:
1. Ensure model is cached in CPU: `manager.model_on_cpu is not None`
2. Check system RAM usage
3. Verify no other processes are using GPU

### Out of Memory

**Symptom**: CUDA out of memory error

**Solution**:
1. Call `force_release()` to clear all cache
2. Reduce batch size
3. Lower resolution (`size_level`)

## Best Practices

1. **Always offload after tasks**: Call `force_offload()` immediately after processing
2. **Handle exceptions**: Offload in exception handlers
3. **Monitor idle time**: Set appropriate timeout for your use case
4. **Batch wisely**: Keep model on GPU during batch, offload after
5. **Release when done**: Call `force_release()` for long idle periods

## Advanced Configuration

### Custom Timeout Per Request

```python
# Temporarily extend timeout for long-running tasks
original_timeout = manager.idle_timeout
manager.idle_timeout = 300  # 5 minutes

# Process
result = process_long_task()

# Restore
manager.idle_timeout = original_timeout
```

### Disable Auto-Monitor

```python
manager = GPUResourceManager(auto_monitor=False)

# Manual control only
manager.get_model(load_func)
# ... process ...
manager.force_offload()
```

### Multiple GPU Support

```python
# GPU 0
manager_0 = GPUResourceManager(device="cuda:0")

# GPU 1
manager_1 = GPUResourceManager(device="cuda:1")
```

## Statistics

Track usage patterns:

```python
stats = manager.get_status()['statistics']

print(f"Total loads: {stats['total_loads']}")
print(f"GPU→CPU transfers: {stats['gpu_to_cpu']}")
print(f"CPU→GPU transfers: {stats['cpu_to_gpu']}")
print(f"Full releases: {stats['full_releases']}")
```

## Integration Examples

### FastAPI

```python
from fastapi import FastAPI
from gpu_manager import get_global_gpu_manager

app = FastAPI()
gpu_manager = get_global_gpu_manager(idle_timeout=60)

@app.post("/process")
async def process(data: dict):
    try:
        model = gpu_manager.get_model(load_func=load_model)
        result = model(data)
        gpu_manager.force_offload()
        return result
    except Exception as e:
        gpu_manager.force_offload()
        raise e
```

### Gradio

```python
import gradio as gr
from gpu_manager import get_global_gpu_manager

gpu_manager = get_global_gpu_manager()

def process_fn(image, prompt):
    try:
        model = gpu_manager.get_model(load_func=load_model)
        result = model(image, prompt)
        gpu_manager.force_offload()
        return result
    except Exception as e:
        gpu_manager.force_offload()
        raise e

gr.Interface(fn=process_fn, ...).launch()
```
