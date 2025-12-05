<div align="center">
  <img src="assets/logo.png" height=100>
  <h1>Step1X-Edit Docker Deployment</h1>
  <p>🎨 AI-Powered Image Editing with Intelligent GPU Management</p>
  
  [English](README_NEW.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](Dockerfile)
  [![GPU](https://img.shields.io/badge/GPU-CUDA%2012.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
</div>

## 📖 Overview

Production-ready Docker deployment for Step1X-Edit with intelligent GPU memory management. Features lazy loading, instant offloading, and three access modes (UI + API + MCP) in a single container.

### ✨ Key Features

- 🚀 **One-Click Deployment** - Auto GPU selection and startup
- 🧠 **Smart GPU Management** - Lazy loading + instant offloading (<1GB idle)
- 🎨 **Modern Web UI** - Drag & drop interface with real-time preview
- 🔌 **REST API** - Full-featured API with Swagger documentation
- 🤖 **MCP Support** - Model Context Protocol for AI assistants
- 🌍 **Multi-Language** - English, 简体中文, 繁體中文, 日本語
- 🐳 **Docker Optimized** - Single container, external access ready
- 📊 **GPU Monitoring** - Real-time status and manual control

## 🚀 Quick Start

### Prerequisites

- NVIDIA GPU with 24GB+ VRAM
- NVIDIA Driver 525+
- Docker 20.10+
- nvidia-docker2

### 3-Step Launch

```bash
# 1. Configure
cp .env.example .env
# Edit MODEL_PATH in .env

# 2. Start (auto-selects best GPU)
bash start.sh

# 3. Access
# UI:  http://0.0.0.0:8000
# API: http://0.0.0.0:8000/docs
```

## 📦 Installation

### Method 1: Docker (Recommended)

#### Install nvidia-docker

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

#### Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env
```

Required settings:
```bash
MODEL_PATH=/path/to/Step1X-Edit-model
PORT=8000
GPU_IDLE_TIMEOUT=60
```

#### Start Service

```bash
# One-click start with auto GPU selection
bash start.sh

# Or manually with docker-compose
docker-compose up -d
```

#### Verify Deployment

```bash
# Run test suite
bash test_deployment.sh

# Check health
curl http://0.0.0.0:8000/health

# View logs
docker-compose logs -f
```

### Method 2: Direct Run

#### Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install flash-attention
python scripts/get_flash_attn.py
# Download and install the wheel from the output
```

#### Start Server

```bash
# Set environment variables
export MODEL_PATH=/path/to/model
export PORT=8000
export GPU_IDLE_TIMEOUT=60

# Start unified server
python unified_server.py
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Service port |
| `HOST` | 0.0.0.0 | Bind address (all interfaces) |
| `NVIDIA_VISIBLE_DEVICES` | 0 | GPU ID (auto-selected by start.sh) |
| `GPU_IDLE_TIMEOUT` | 60 | Auto-offload timeout (seconds) |
| `MODEL_PATH` | - | Path to Step1X-Edit model (required) |
| `ENABLE_UI` | true | Enable web UI |
| `ENABLE_API` | true | Enable REST API |
| `ENABLE_MCP` | true | Enable MCP server |
| `DEFAULT_NUM_STEPS` | 28 | Default inference steps |
| `DEFAULT_GUIDANCE_SCALE` | 6.0 | Default CFG scale |
| `DEFAULT_SIZE_LEVEL` | 1024 | Default resolution |

### Docker Compose Configuration

```yaml
version: '3.8'
services:
  step1x-edit:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-0}
      - PORT=${PORT:-8000}
      - GPU_IDLE_TIMEOUT=${GPU_IDLE_TIMEOUT:-60}
      - MODEL_PATH=${MODEL_PATH}
    ports:
      - "${PORT:-8000}:8000"
    volumes:
      - ${MODEL_PATH}:/models
      - ./outputs:/app/outputs
    restart: unless-stopped
```

## 💻 Usage

### Web UI

1. Open browser: `http://0.0.0.0:8000`
2. Drag & drop image or click to upload
3. Enter editing instruction (e.g., "add a red hat")
4. Adjust parameters:
   - **Steps** (10-50): Higher = better quality
   - **Guidance Scale** (1-15): Higher = stronger prompt
   - **Resolution** (512/768/1024): Output size
   - **Seed**: For reproducible results
5. Click "Edit Image"
6. View side-by-side comparison

### REST API

#### Edit Image

```bash
curl -X POST http://0.0.0.0:8000/api/edit \
  -F "file=@input.jpg" \
  -F "prompt=add a red hat on the person" \
  -F "num_steps=28" \
  -F "guidance_scale=6.0" \
  -F "size_level=1024" \
  --output result.png
```

#### Check GPU Status

```bash
curl http://0.0.0.0:8000/api/gpu/status
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
    "cpu_to_gpu": 4
  }
}
```

#### Manual GPU Control

```bash
# Offload to CPU (keep in RAM)
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# Complete release (clear all)
curl -X POST http://0.0.0.0:8000/api/gpu/release
```

#### API Documentation

Interactive Swagger UI: `http://0.0.0.0:8000/docs`

### MCP (Model Context Protocol)

#### Python Client

```python
from mcp import ClientSession

async with ClientSession() as session:
    result = await session.call_tool(
        "edit_image",
        {
            "image_path": "input.jpg",
            "prompt": "add a red hat",
            "num_steps": 28,
            "guidance_scale": 6.0
        }
    )
    print(f"Saved to: {result['output_path']}")
```

#### Available Tools

- `edit_image` - Edit single image
- `batch_edit_images` - Edit multiple images
- `get_gpu_status` - Get GPU status
- `offload_gpu` - Offload to CPU
- `release_gpu` - Complete release

See [MCP_GUIDE.md](MCP_GUIDE.md) for details.

## 🧠 GPU Memory Management

### Intelligent Resource Management

```
Unloaded ──first(20-30s)──> GPU ──complete(2s)──> CPU ──next(2-5s)──> GPU
   ↑                                                 ↓
   └──────────────timeout/release(1s)───────────────┘
```

### Memory States

| State | GPU Memory | Description |
|-------|------------|-------------|
| Unloaded | <1GB | Model not loaded |
| CPU Cache | <1GB | Model in RAM, quick reload (2-5s) |
| GPU Active | ~40GB | Model on GPU, processing |

### Features

- **Lazy Loading**: Model loads only on first request
- **Instant Offload**: Auto-moves to CPU after each task (2s)
- **Quick Reload**: CPU→GPU in 2-5 seconds
- **Auto-Monitoring**: Background thread with configurable timeout
- **Manual Control**: Force offload/release via API or UI

See [GPU_MANAGEMENT.md](GPU_MANAGEMENT.md) for details.

## 📊 Performance

### Benchmarks (H800 GPU)

| Operation | Time | GPU Memory |
|-----------|------|------------|
| First load (disk→GPU) | 20-30s | ~40GB |
| Edit (1024px, 28 steps) | 15-20s | ~40GB |
| Reload (CPU→GPU) | 2-5s | ~40GB |
| Offload (GPU→CPU) | ~2s | <1GB |
| Release (clear all) | ~1s | <1GB |

### Optimization Tips

- **Faster**: Lower `num_steps` (20) or `size_level` (768)
- **Better Quality**: Higher `num_steps` (35-40) and `guidance_scale` (7-8)
- **Reproducible**: Set `seed` parameter
- **Frequent Use**: Increase `GPU_IDLE_TIMEOUT`

## 📁 Project Structure

```
Step1X-Edit/
├── Dockerfile                      # Docker image definition
├── docker-compose.yml              # Container orchestration
├── start.sh                        # One-click startup script
├── test_deployment.sh              # Test suite
│
├── unified_server.py               # UI + API server
├── mcp_server.py                   # MCP server
├── gpu_manager.py                  # GPU resource manager
├── step1x_manager.py               # Step1X-Edit wrapper
│
├── DEPLOYMENT.md                   # Deployment guide
├── GPU_MANAGEMENT.md               # GPU management docs
├── MCP_GUIDE.md                    # MCP usage guide
├── QUICK_REFERENCE.md              # Quick reference
│
├── modules/                        # Model modules
├── scripts/                        # Utility scripts
└── examples/                       # Example images
```

## 🛠️ Tech Stack

- **Framework**: FastAPI, Gradio
- **AI/ML**: PyTorch, Transformers, Diffusers
- **GPU**: CUDA 12.1, Flash Attention
- **Container**: Docker, nvidia-docker2
- **Protocol**: MCP (Model Context Protocol)
- **API**: REST, WebSocket, Swagger/OpenAPI

## 🧪 Testing

```bash
# Run full test suite
bash test_deployment.sh
```

Tests include:
- ✓ Container health
- ✓ GPU accessibility
- ✓ API endpoints
- ✓ UI accessibility
- ✓ GPU management
- ✓ Image editing (optional)

## 🐛 Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs step1x-edit

# Check GPU
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### High GPU Memory

```bash
# Check status
curl http://0.0.0.0:8000/api/gpu/status

# Manual offload
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# Verify
nvidia-smi
```

### Port Already in Use

```bash
# Change port in .env
PORT=8001

# Restart
docker-compose down
bash start.sh
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete troubleshooting guide.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📝 Changelog

### v1.2.0 (2025-12-06)
- ✨ Added unified server with UI + API + MCP
- 🧠 Implemented intelligent GPU memory management
- 🐳 Docker deployment with auto GPU selection
- 📚 Comprehensive documentation
- 🧪 Test suite for validation

### v1.1.0 (2025-07-09)
- ✨ Added T2I generation support
- 🎨 Improved editing quality
- 📊 Better instruction following

### v1.0.0 (2025-04-25)
- 🎉 Initial release
- 🎨 Image editing with natural language
- 📊 GEdit-Bench evaluation

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Special thanks to:
- [Step1X-Edit Team](https://github.com/stepfun-ai/Step1X-Edit) - Original model
- [Kohya](https://github.com/kohya-ss/sd-scripts) - Training scripts
- [xDiT](https://github.com/xdit-project/xDiT) - Parallel inference
- [TeaCache](https://github.com/ali-vilab/TeaCache) - Acceleration
- [HuggingFace](https://huggingface.co) - Model hosting

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/neosun100/Step1X-Edit/issues)
- **Discord**: [Join community](https://discord.gg/j3qzuAyn)
- **Documentation**: [Full docs](DEPLOYMENT.md)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/Step1X-Edit&type=Date)](https://star-history.com/#neosun100/Step1X-Edit)

## 📱 Follow Us

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

<div align="center">
  Made with ❤️ by the Step1X-Edit Community
  <br>
  <sub>If this project helps you, please give it a ⭐️</sub>
</div>
