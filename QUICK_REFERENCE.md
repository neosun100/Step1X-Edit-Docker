# Step1X-Edit Quick Reference

## 🚀 Quick Start

```bash
cp .env.example .env          # Configure
bash start.sh                 # Start
bash test_deployment.sh       # Test
```

**Access:**
- UI: http://0.0.0.0:8000
- API: http://0.0.0.0:8000/docs
- Health: http://0.0.0.0:8000/health

## 📋 Common Commands

### Container Management
```bash
docker-compose up -d          # Start
docker-compose down           # Stop
docker-compose restart        # Restart
docker-compose logs -f        # View logs
docker ps                     # Check status
```

### GPU Monitoring
```bash
nvidia-smi                    # GPU status
watch -n 1 nvidia-smi         # Real-time monitoring
docker exec step1x-edit nvidia-smi  # GPU in container
```

### API Calls
```bash
# Health check
curl http://0.0.0.0:8000/health

# GPU status
curl http://0.0.0.0:8000/api/gpu/status

# Offload GPU
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# Release GPU
curl -X POST http://0.0.0.0:8000/api/gpu/release

# Edit image
curl -X POST http://0.0.0.0:8000/api/edit \
  -F "file=@input.jpg" \
  -F "prompt=add a red hat" \
  -F "num_steps=28" \
  --output result.png
```

## ⚙️ Configuration (.env)

```bash
PORT=8000                     # Service port
GPU_IDLE_TIMEOUT=60          # Auto-offload timeout (seconds)
MODEL_PATH=/path/to/model    # Model directory
ENABLE_UI=true               # Enable web UI
ENABLE_API=true              # Enable REST API
ENABLE_MCP=true              # Enable MCP server
```

## 🎨 Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| num_steps | 10-50 | 28 | Inference steps (higher = better quality) |
| guidance_scale | 1.0-15.0 | 6.0 | CFG scale (higher = stronger prompt) |
| size_level | 512/768/1024 | 1024 | Output resolution |
| seed | any int | random | Random seed for reproducibility |

## 📊 GPU States

| State | GPU Memory | Description |
|-------|------------|-------------|
| Unloaded | <1GB | Model not loaded |
| CPU Cache | <1GB | Model in RAM, quick reload (2-5s) |
| GPU Active | ~40GB | Model on GPU, processing |

## 🔄 State Transitions

```
Unloaded ──first(20-30s)──> GPU ──complete(2s)──> CPU
   ↑                                                ↓
   └──────────timeout/release(1s)──────────────────┘
```

## 🐛 Troubleshooting

### Container won't start
```bash
docker logs step1x-edit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### High GPU memory
```bash
curl -X POST http://0.0.0.0:8000/api/gpu/offload
nvidia-smi
```

### Port in use
```bash
# Edit .env
PORT=8001
bash start.sh
```

### Slow processing
- Reduce num_steps (e.g., 20)
- Lower size_level (e.g., 768)
- Increase GPU_IDLE_TIMEOUT

## 📱 Access Modes

### 1. Web UI
- Open: http://0.0.0.0:8000
- Drag & drop image
- Enter prompt
- Adjust parameters
- Click "Edit Image"

### 2. REST API
```bash
curl -X POST http://0.0.0.0:8000/api/edit \
  -F "file=@image.jpg" \
  -F "prompt=your instruction" \
  --output result.png
```

### 3. MCP
```python
from mcp import ClientSession

async with ClientSession() as session:
    result = await session.call_tool(
        "edit_image",
        {"image_path": "input.jpg", "prompt": "add a hat"}
    )
```

## 🔧 GPU Management

### Auto Management
- First request: Load to GPU (20-30s)
- After task: Auto-offload to CPU (2s)
- Next request: Quick reload (2-5s)
- Idle timeout: Auto-offload after 60s

### Manual Control

**Via API:**
```bash
curl -X POST http://0.0.0.0:8000/api/gpu/offload  # To CPU
curl -X POST http://0.0.0.0:8000/api/gpu/release  # Clear all
```

**Via UI:**
- Click "Free GPU" button

**Via MCP:**
```python
await mcp_client.call_tool("offload_gpu", {})
await mcp_client.call_tool("release_gpu", {})
```

## 📈 Performance

| Operation | Time | GPU Memory |
|-----------|------|------------|
| First load | 20-30s | ~40GB |
| Edit (1024px, 28 steps) | 15-20s | ~40GB |
| Reload from CPU | 2-5s | ~40GB |
| Offload to CPU | ~2s | <1GB |

## 🔐 Security (Production)

```nginx
# nginx reverse proxy
server {
    listen 80;
    server_name your-domain.com;
    client_max_body_size 50M;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
    }
}
```

## 📚 Documentation

- **DEPLOYMENT.md** - Complete deployment guide
- **GPU_MANAGEMENT.md** - GPU memory management
- **MCP_GUIDE.md** - MCP usage guide
- **DEPLOYMENT_SUMMARY.md** - Implementation summary

## 🆘 Support

- GitHub: https://github.com/stepfun-ai/Step1X-Edit/issues
- Discord: https://discord.gg/j3qzuAyn

## ✅ Testing

```bash
bash test_deployment.sh
```

Tests:
- ✓ Container running
- ✓ GPU accessible
- ✓ Health endpoint
- ✓ GPU status API
- ✓ UI accessible
- ✓ API docs available
- ✓ GPU management
- ✓ Image editing (optional)

## 🎯 Best Practices

1. **Always offload after tasks** - Frees GPU memory
2. **Monitor GPU usage** - `watch -n 1 nvidia-smi`
3. **Adjust timeout** - Match your usage pattern
4. **Batch processing** - Keep model on GPU during batch
5. **Use appropriate resolution** - 768px often sufficient

## 🔄 Updates

```bash
git pull                      # Update code
docker-compose build          # Rebuild image
docker-compose down           # Stop
bash start.sh                 # Start
```

## 💡 Tips

- **Faster processing**: Lower num_steps (20-25) or size_level (768)
- **Better quality**: Higher num_steps (35-40) and guidance_scale (7-8)
- **Reproducible results**: Set seed parameter
- **Save GPU memory**: Lower GPU_IDLE_TIMEOUT for occasional use
- **Frequent use**: Increase GPU_IDLE_TIMEOUT to keep model ready

## 📞 Quick Help

**Can't access UI?**
- Check: `curl http://0.0.0.0:8000/health`
- Logs: `docker-compose logs -f`

**GPU memory high?**
- Offload: `curl -X POST http://0.0.0.0:8000/api/gpu/offload`
- Check: `nvidia-smi`

**Slow processing?**
- Reduce num_steps or size_level
- Check GPU utilization: `nvidia-smi`

**Need help?**
- Read: DEPLOYMENT.md
- Ask: GitHub Issues or Discord
