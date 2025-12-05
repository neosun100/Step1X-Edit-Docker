# Step1X-Edit Docker Deployment Guide

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/stepfun-ai/Step1X-Edit.git
cd Step1X-Edit

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Start service
bash start.sh
```

Access:
- **UI**: http://0.0.0.0:8000
- **API Docs**: http://0.0.0.0:8000/docs
- **Health**: http://0.0.0.0:8000/health

## Features

✅ **Auto GPU Selection**: Automatically selects GPU with least memory usage  
✅ **GPU Memory Management**: Lazy loading + instant offloading  
✅ **Three Access Modes**: UI + API + MCP in single container  
✅ **Multi-language UI**: English, 简体中文, 繁體中文, 日本語  
✅ **Docker Optimized**: Runs on 0.0.0.0 for external access  

## Prerequisites

- NVIDIA GPU with 24GB+ VRAM
- NVIDIA Driver 525+
- Docker 20.10+
- nvidia-docker2

### Install nvidia-docker

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Configuration

### Environment Variables (.env)

```bash
# Server
PORT=8000                    # Service port
HOST=0.0.0.0                # Bind to all interfaces

# GPU
NVIDIA_VISIBLE_DEVICES=0    # Auto-selected by start.sh
GPU_IDLE_TIMEOUT=60         # Offload after 60s idle

# Model
MODEL_PATH=/path/to/model   # Path to Step1X-Edit model

# Features
ENABLE_UI=true              # Enable web UI
ENABLE_API=true             # Enable REST API
ENABLE_MCP=true             # Enable MCP server

# Defaults
DEFAULT_NUM_STEPS=28
DEFAULT_GUIDANCE_SCALE=6.0
DEFAULT_SIZE_LEVEL=1024
```

## Usage

### Start Service

```bash
bash start.sh
```

Output:
```
=========================================
  Step1X-Edit Docker Startup Script
=========================================
✓ NVIDIA Docker environment detected
✓ Selected GPU 0: NVIDIA H800 (1024MB used)
✓ Loading .env file
Starting Step1X-Edit container...
✓ Step1X-Edit Started Successfully!

Access URLs:
  • UI:      http://0.0.0.0:8000
  • API:     http://0.0.0.0:8000/docs
  • Health:  http://0.0.0.0:8000/health
```

### Stop Service

```bash
docker-compose down
```

### View Logs

```bash
docker-compose logs -f
```

### Restart Service

```bash
docker-compose restart
```

## Access Modes

### 1. Web UI

Open browser: `http://0.0.0.0:8000`

Features:
- Drag & drop image upload
- All parameters configurable
- Real-time progress
- Side-by-side comparison
- GPU status monitoring
- Multi-language support

### 2. REST API

**Edit Image:**
```bash
curl -X POST http://0.0.0.0:8000/api/edit \
  -F "file=@input.jpg" \
  -F "prompt=add a red hat" \
  -F "num_steps=28" \
  -F "guidance_scale=6.0" \
  -F "size_level=1024" \
  --output result.png
```

**GPU Status:**
```bash
curl http://0.0.0.0:8000/api/gpu/status
```

**Offload GPU:**
```bash
curl -X POST http://0.0.0.0:8000/api/gpu/offload
```

**API Documentation:**  
Visit `http://0.0.0.0:8000/docs` for interactive Swagger UI

### 3. MCP (Model Context Protocol)

**Python Client:**
```python
from mcp import ClientSession

async with ClientSession() as session:
    result = await session.call_tool(
        "edit_image",
        {
            "image_path": "input.jpg",
            "prompt": "add a red hat"
        }
    )
```

See [MCP_GUIDE.md](MCP_GUIDE.md) for details.

## GPU Management

### Memory States

| State | GPU Memory | Description |
|-------|------------|-------------|
| Unloaded | <1GB | Model not loaded |
| CPU Cache | <1GB | Model in RAM, quick reload |
| GPU Active | ~40GB | Model on GPU, processing |

### Automatic Management

1. **First request**: Load model to GPU (20-30s)
2. **Processing**: Model stays on GPU
3. **Task complete**: Auto-offload to CPU (2s)
4. **Next request**: Quick reload from CPU (2-5s)
5. **Idle timeout**: Auto-offload after 60s

### Manual Control

**Via API:**
```bash
# Offload to CPU (keep in RAM)
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# Complete release (clear all)
curl -X POST http://0.0.0.0:8000/api/gpu/release
```

**Via UI:**  
Click "Free GPU" button in GPU status panel

**Via MCP:**
```python
await mcp_client.call_tool("offload_gpu", {})
await mcp_client.call_tool("release_gpu", {})
```

See [GPU_MANAGEMENT.md](GPU_MANAGEMENT.md) for details.

## Testing

### Health Check

```bash
curl http://0.0.0.0:8000/health
```

Expected:
```json
{
  "status": "healthy",
  "timestamp": 1733500000.0,
  "gpu_available": true
}
```

### Test Image Edit

```bash
# Download test image
wget https://example.com/test.jpg -O test.jpg

# Edit via API
curl -X POST http://0.0.0.0:8000/api/edit \
  -F "file=@test.jpg" \
  -F "prompt=add a red hat" \
  --output result.png

# Check result
file result.png
```

### GPU Status

```bash
curl http://0.0.0.0:8000/api/gpu/status | jq
```

Expected:
```json
{
  "model_location": "CPU",
  "idle_time": 45.2,
  "gpu_memory_allocated_gb": 0.12,
  "gpu_memory_reserved_gb": 0.5,
  "statistics": {
    "total_loads": 1,
    "gpu_to_cpu": 1,
    "cpu_to_gpu": 0,
    "full_releases": 0
  }
}
```

## Troubleshooting

### Container Won't Start

**Check Docker:**
```bash
docker ps -a
docker logs step1x-edit
```

**Check GPU:**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### High GPU Memory

**Symptom**: GPU memory stays high when idle

**Solution**:
```bash
# Check status
curl http://0.0.0.0:8000/api/gpu/status

# Manual offload
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# Check again
nvidia-smi
```

### Slow Processing

**Check**:
1. GPU utilization: `nvidia-smi`
2. Model location: `curl http://0.0.0.0:8000/api/gpu/status`
3. Container resources: `docker stats step1x-edit`

**Optimize**:
- Reduce `num_steps` (e.g., 20 instead of 28)
- Lower `size_level` (e.g., 768 instead of 1024)
- Increase `GPU_IDLE_TIMEOUT` to keep model on GPU longer

### Port Already in Use

**Change port in .env:**
```bash
PORT=8001
```

**Restart:**
```bash
docker-compose down
bash start.sh
```

## Performance

### Benchmarks (H800 GPU)

| Operation | Time | GPU Memory |
|-----------|------|------------|
| First load | 20-30s | 40GB |
| Edit (1024px, 28 steps) | 15-20s | 40GB |
| Reload from CPU | 2-5s | 40GB |
| Offload to CPU | 2s | <1GB |

### Optimization Tips

1. **Batch processing**: Keep model on GPU during batch
2. **Adjust timeout**: Increase for frequent use, decrease for occasional use
3. **Resolution**: Use 768px for faster processing
4. **Steps**: 20-25 steps often sufficient

## Multi-GPU Setup

### Manual GPU Selection

```bash
# Use GPU 1
export NVIDIA_VISIBLE_DEVICES=1
docker-compose up -d
```

### Multiple Instances

```bash
# Instance 1 on GPU 0
NVIDIA_VISIBLE_DEVICES=0 PORT=8000 docker-compose up -d

# Instance 2 on GPU 1
NVIDIA_VISIBLE_DEVICES=1 PORT=8001 docker-compose up -d
```

## Security

### Production Deployment

1. **Use reverse proxy** (nginx/traefik)
2. **Enable HTTPS**
3. **Add authentication**
4. **Limit file upload size**
5. **Rate limiting**

### Example nginx config

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Container Stats

```bash
docker stats step1x-edit
```

### GPU Monitoring

```bash
watch -n 1 nvidia-smi
```

### Logs

```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker logs step1x-edit
```

## Backup & Restore

### Backup Configuration

```bash
tar -czf step1x-config-backup.tar.gz .env docker-compose.yml
```

### Backup Outputs

```bash
tar -czf step1x-outputs-backup.tar.gz outputs/
```

### Restore

```bash
tar -xzf step1x-config-backup.tar.gz
bash start.sh
```

## Updates

### Update Container

```bash
# Pull latest code
git pull

# Rebuild image
docker-compose build

# Restart
docker-compose down
bash start.sh
```

### Update Model

```bash
# Download new model
# Update MODEL_PATH in .env
# Restart container
docker-compose restart
```

## Support

- **Documentation**: [README.md](README.md)
- **GPU Management**: [GPU_MANAGEMENT.md](GPU_MANAGEMENT.md)
- **MCP Guide**: [MCP_GUIDE.md](MCP_GUIDE.md)
- **GitHub Issues**: https://github.com/stepfun-ai/Step1X-Edit/issues
- **Discord**: https://discord.gg/j3qzuAyn
