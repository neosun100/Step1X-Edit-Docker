# Docker Deployment Guide

## 🚀 Quick Start

### Prerequisites

1. **NVIDIA GPU** with CUDA support
2. **NVIDIA Docker Runtime** installed
3. **Docker** and **Docker Compose**

### One-Command Startup

```bash
./start.sh
```

This script will:
- ✅ Check all prerequisites
- ✅ Auto-select GPU with least memory usage
- ✅ Create necessary directories
- ✅ Build and start Docker containers
- ✅ Display access information

## 📋 Manual Setup

### 1. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

### 2. Build Image

```bash
docker-compose build
```

### 3. Start Services

```bash
docker-compose up -d
```

### 4. Check Status

```bash
docker-compose ps
docker-compose logs -f
```

## 🎯 Access Points

After starting, services are available at:

| Service | URL | Description |
|---------|-----|-------------|
| **UI** | http://localhost:7860 | Web interface |
| **API** | http://localhost:8000 | REST API |
| **Swagger** | http://localhost:8000/docs | API documentation |
| **MCP** | tcp://localhost:8001 | MCP server |

## ⚙️ Configuration

### GPU Selection

The start script automatically selects the GPU with least memory usage.

Manual selection:
```bash
export NVIDIA_VISIBLE_DEVICES=1  # Use GPU 1
./start.sh
```

### Memory Optimization

#### Option 1: FP8 Quantization (~11GB savings)
```bash
# In .env
QUANTIZED=true
```

#### Option 2: CPU Offload (~17GB savings, slower)
```bash
# In .env
OFFLOAD=true
```

#### Option 3: Both (~27GB savings)
```bash
# In .env
QUANTIZED=true
OFFLOAD=true
```

### GPU Manager Settings

```bash
# In .env
GPU_IDLE_TIMEOUT=60      # Auto-offload after 60s
AUTO_MONITOR=true        # Enable auto-monitoring
GPU_OFFLOAD_DELAY=0.0    # Delay after offload
```

## 🔧 Common Commands

### View Logs
```bash
docker-compose logs -f
docker-compose logs -f step1x-edit
```

### Restart Services
```bash
docker-compose restart
```

### Stop Services
```bash
docker-compose down
```

### Rebuild After Code Changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Access Container Shell
```bash
docker exec -it step1x-edit bash
```

### Monitor GPU Usage
```bash
watch -n 1 nvidia-smi
```

## 📁 Directory Structure

```
Step1X-Edit/
├── models/          # Model weights (mounted volume)
├── inputs/          # Input images
├── outputs/         # Generated images
├── config/          # Configuration files
└── examples/        # Example images
```

### Volume Mounts

Edit `docker-compose.yml` to customize:

```yaml
volumes:
  - /path/to/your/models:/app/models
  - /path/to/your/inputs:/app/inputs
  - /path/to/your/outputs:/app/outputs
```

## 🧪 Testing

### API Tests
```bash
./test_api.sh examples/0000.jpg
```

### Manual Testing

#### UI Test
```bash
# Open in browser
open http://localhost:7860
```

#### API Test
```bash
curl -X POST http://localhost:8000/api/edit \
  -F "file=@examples/0000.jpg" \
  -F "prompt=add a hat" \
  -F "num_steps=10" \
  -o output.png
```

#### GPU Status
```bash
curl http://localhost:8000/gpu/status | python3 -m json.tool
```

## 🔍 Troubleshooting

### Container won't start

**Check logs:**
```bash
docker-compose logs step1x-edit
```

**Common issues:**
1. NVIDIA Docker not installed
2. GPU already in use
3. Port conflicts

### GPU not detected

**Check NVIDIA Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Check GPU visibility:**
```bash
docker exec step1x-edit nvidia-smi
```

### Out of memory errors

**Solutions:**
1. Enable quantization: `QUANTIZED=true`
2. Enable offload: `OFFLOAD=true`
3. Lower resolution: `DEFAULT_SIZE_LEVEL=768`
4. Use smaller batch size

### Port already in use

**Find process:**
```bash
lsof -i :7860  # UI port
lsof -i :8000  # API port
```

**Change ports in .env:**
```bash
PORT=7861
API_PORT=8001
```

### Model loading errors

**Check paths:**
```bash
docker exec step1x-edit ls -la /app/models
```

**Verify mounts:**
```bash
docker inspect step1x-edit | grep -A 10 Mounts
```

## 🚀 Production Deployment

### Multi-GPU Setup

#### Option 1: Multiple Containers
```bash
# GPU 0
NVIDIA_VISIBLE_DEVICES=0 PORT=7860 API_PORT=8000 docker-compose up -d

# GPU 1
NVIDIA_VISIBLE_DEVICES=1 PORT=7861 API_PORT=8001 docker-compose up -d
```

#### Option 2: Load Balancer
```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf

  step1x-edit-gpu0:
    ...
    environment:
      - NVIDIA_VISIBLE_DEVICES=0

  step1x-edit-gpu1:
    ...
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
```

### Monitoring

#### Prometheus Metrics (optional)
```python
# Add to your app
from prometheus_client import Counter, Histogram

edit_requests = Counter('edit_requests_total', 'Total edit requests')
edit_duration = Histogram('edit_duration_seconds', 'Edit duration')
```

#### Health Check Endpoint
```bash
curl http://localhost:8000/health
```

### Security

#### API Authentication (TODO)
```bash
# In .env
API_AUTH_ENABLED=true
API_KEY=your-secret-key
```

#### Firewall Rules
```bash
# Allow only specific IPs
sudo ufw allow from 192.168.1.0/24 to any port 8000
```

## 📊 Performance Tuning

### GPU Memory vs Speed

| Configuration | Memory | Speed |
|--------------|--------|-------|
| Default | 49.8GB | 22s |
| + Auto-offload | <1GB* | 22s + 2-5s |
| + FP8 | 34GB | 25s |
| + Offload | 29GB | 63s |
| + FP8 + Offload | 18GB | 51s |

*After offload

### Recommendations

**Interactive use (single user):**
```bash
GPU_IDLE_TIMEOUT=60
AUTO_MONITOR=true
QUANTIZED=false
OFFLOAD=false
```

**Production (multi-user):**
```bash
GPU_IDLE_TIMEOUT=30
AUTO_MONITOR=true
QUANTIZED=true    # If memory is tight
OFFLOAD=false
```

**Batch processing:**
```bash
AUTO_MONITOR=false  # Disable auto-offload
QUANTIZED=false
OFFLOAD=false
# Manually control via API
```

## 🔗 Related Documentation

- [GPU Management](GPU_MANAGEMENT.md)
- [MCP Guide](MCP_GUIDE.md)
- [Main README](README.md)

## 📞 Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify GPU: `nvidia-smi`
3. Test API: `./test_api.sh`
4. Check documentation above
