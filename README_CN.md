<div align="center">
  <img src="assets/logo.png" height=100>
  <h1>Step1X-Edit Docker 部署版</h1>
  <p>🎨 智能 GPU 管理的 AI 图像编辑系统</p>
  
  [English](README_NEW.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)
  
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
  [![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](Dockerfile)
  [![GPU](https://img.shields.io/badge/GPU-CUDA%2012.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
  [![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
</div>

## 📖 项目简介

Step1X-Edit 的生产级 Docker 部署方案，具备智能 GPU 显存管理功能。支持懒加载、即用即卸，单容器提供三种访问方式（UI + API + MCP）。

### ✨ 核心特性

- 🚀 **一键部署** - 自动选择最优 GPU 并启动
- 🧠 **智能显存管理** - 懒加载 + 即用即卸（空闲 <1GB）
- 🎨 **现代化 Web UI** - 拖拽上传，实时预览
- 🔌 **REST API** - 完整 API 接口，Swagger 文档
- 🤖 **MCP 支持** - 模型上下文协议，对接 AI 助手
- 🌍 **多语言** - 英文、简体中文、繁体中文、日文
- 🐳 **Docker 优化** - 单容器，支持外部访问
- 📊 **GPU 监控** - 实时状态显示和手动控制

## 🚀 快速开始

### 前置要求

- NVIDIA GPU（24GB+ 显存）
- NVIDIA 驱动 525+
- Docker 20.10+
- nvidia-docker2

### 三步启动

```bash
# 1. 配置环境
cp .env.example .env
# 编辑 .env 中的 MODEL_PATH

# 2. 启动服务（自动选择最优 GPU）
bash start.sh

# 3. 访问服务
# UI:  http://0.0.0.0:8000
# API: http://0.0.0.0:8000/docs
```

## 📦 安装部署

### 方式一：Docker 部署（推荐）

#### 安装 nvidia-docker

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

#### 配置环境变量

```bash
# 复制模板
cp .env.example .env

# 编辑配置
nano .env
```

必需配置：
```bash
MODEL_PATH=/path/to/Step1X-Edit-model  # 模型路径
PORT=8000                               # 服务端口
GPU_IDLE_TIMEOUT=60                     # GPU 空闲超时（秒）
```

#### 启动服务

```bash
# 一键启动（自动选择 GPU）
bash start.sh

# 或手动启动
docker-compose up -d
```

#### 验证部署

```bash
# 运行测试套件
bash test_deployment.sh

# 检查健康状态
curl http://0.0.0.0:8000/health

# 查看日志
docker-compose logs -f
```

### 方式二：直接运行

#### 安装依赖

```bash
# 安装 Python 包
pip install -r requirements.txt

# 安装 flash-attention
python scripts/get_flash_attn.py
# 根据输出下载并安装对应的 wheel 文件
```

#### 启动服务器

```bash
# 设置环境变量
export MODEL_PATH=/path/to/model
export PORT=8000
export GPU_IDLE_TIMEOUT=60

# 启动统一服务器
python unified_server.py
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `PORT` | 8000 | 服务端口 |
| `HOST` | 0.0.0.0 | 绑定地址（所有网卡） |
| `NVIDIA_VISIBLE_DEVICES` | 0 | GPU ID（start.sh 自动选择） |
| `GPU_IDLE_TIMEOUT` | 60 | 自动卸载超时（秒） |
| `MODEL_PATH` | - | Step1X-Edit 模型路径（必需） |
| `ENABLE_UI` | true | 启用 Web UI |
| `ENABLE_API` | true | 启用 REST API |
| `ENABLE_MCP` | true | 启用 MCP 服务器 |
| `DEFAULT_NUM_STEPS` | 28 | 默认推理步数 |
| `DEFAULT_GUIDANCE_SCALE` | 6.0 | 默认 CFG 系数 |
| `DEFAULT_SIZE_LEVEL` | 1024 | 默认分辨率 |

### Docker Compose 配置

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

## 💻 使用方法

### Web UI

1. 打开浏览器：`http://0.0.0.0:8000`
2. 拖拽图片或点击上传
3. 输入编辑指令（如："给人物添加一顶红色帽子"）
4. 调整参数：
   - **步数** (10-50)：越高质量越好
   - **引导系数** (1-15)：越高提示词影响越强
   - **分辨率** (512/768/1024)：输出尺寸
   - **随机种子**：用于可复现结果
5. 点击"编辑图片"
6. 查看对比结果

### REST API

#### 编辑图片

```bash
curl -X POST http://0.0.0.0:8000/api/edit \
  -F "file=@input.jpg" \
  -F "prompt=给人物添加一顶红色帽子" \
  -F "num_steps=28" \
  -F "guidance_scale=6.0" \
  -F "size_level=1024" \
  --output result.png
```

#### 查询 GPU 状态

```bash
curl http://0.0.0.0:8000/api/gpu/status
```

响应示例：
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

#### 手动 GPU 控制

```bash
# 卸载到 CPU（保留在内存）
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# 完全释放（清空所有缓存）
curl -X POST http://0.0.0.0:8000/api/gpu/release
```

#### API 文档

交互式 Swagger UI：`http://0.0.0.0:8000/docs`

### MCP（模型上下文协议）

#### Python 客户端

```python
from mcp import ClientSession

async with ClientSession() as session:
    result = await session.call_tool(
        "edit_image",
        {
            "image_path": "input.jpg",
            "prompt": "添加红色帽子",
            "num_steps": 28,
            "guidance_scale": 6.0
        }
    )
    print(f"保存至: {result['output_path']}")
```

#### 可用工具

- `edit_image` - 编辑单张图片
- `batch_edit_images` - 批量编辑图片
- `get_gpu_status` - 获取 GPU 状态
- `offload_gpu` - 卸载到 CPU
- `release_gpu` - 完全释放

详见 [MCP_GUIDE.md](MCP_GUIDE.md)

## 🧠 GPU 显存管理

### 智能资源管理

```
未加载 ──首次(20-30s)──> GPU ──完成(2s)──> CPU ──下次(2-5s)──> GPU
   ↑                                          ↓
   └──────────────超时/释放(1s)────────────────┘
```

### 显存状态

| 状态 | GPU 显存 | 说明 |
|------|----------|------|
| 未加载 | <1GB | 模型未加载 |
| CPU 缓存 | <1GB | 模型在内存，快速重载（2-5秒） |
| GPU 活跃 | ~40GB | 模型在 GPU，处理中 |

### 功能特性

- **懒加载**：仅在首次请求时加载模型
- **即用即卸**：任务完成后自动转移到 CPU（2秒）
- **快速重载**：CPU→GPU 仅需 2-5 秒
- **自动监控**：后台线程，可配置超时时间
- **手动控制**：通过 API 或 UI 强制卸载/释放

详见 [GPU_MANAGEMENT.md](GPU_MANAGEMENT.md)

## 📊 性能指标

### 基准测试（H800 GPU）

| 操作 | 耗时 | GPU 显存 |
|------|------|----------|
| 首次加载（磁盘→GPU） | 20-30秒 | ~40GB |
| 编辑（1024px, 28步） | 15-20秒 | ~40GB |
| 重载（CPU→GPU） | 2-5秒 | ~40GB |
| 卸载（GPU→CPU） | ~2秒 | <1GB |
| 释放（清空所有） | ~1秒 | <1GB |

### 优化建议

- **更快速度**：降低 `num_steps`（20）或 `size_level`（768）
- **更高质量**：提高 `num_steps`（35-40）和 `guidance_scale`（7-8）
- **可复现**：设置 `seed` 参数
- **频繁使用**：增加 `GPU_IDLE_TIMEOUT`

## 📁 项目结构

```
Step1X-Edit/
├── Dockerfile                      # Docker 镜像定义
├── docker-compose.yml              # 容器编排配置
├── start.sh                        # 一键启动脚本
├── test_deployment.sh              # 测试套件
│
├── unified_server.py               # UI + API 服务器
├── mcp_server.py                   # MCP 服务器
├── gpu_manager.py                  # GPU 资源管理器
├── step1x_manager.py               # Step1X-Edit 封装
│
├── DEPLOYMENT.md                   # 部署指南
├── GPU_MANAGEMENT.md               # GPU 管理文档
├── MCP_GUIDE.md                    # MCP 使用指南
├── QUICK_REFERENCE.md              # 快速参考
│
├── modules/                        # 模型模块
├── scripts/                        # 工具脚本
└── examples/                       # 示例图片
```

## 🛠️ 技术栈

- **框架**：FastAPI, Gradio
- **AI/ML**：PyTorch, Transformers, Diffusers
- **GPU**：CUDA 12.1, Flash Attention
- **容器**：Docker, nvidia-docker2
- **协议**：MCP（模型上下文协议）
- **API**：REST, WebSocket, Swagger/OpenAPI

## 🧪 测试

```bash
# 运行完整测试套件
bash test_deployment.sh
```

测试项目：
- ✓ 容器健康检查
- ✓ GPU 可访问性
- ✓ API 端点
- ✓ UI 可访问性
- ✓ GPU 管理功能
- ✓ 图片编辑（可选）

## 🐛 故障排查

### 容器无法启动

```bash
# 查看日志
docker logs step1x-edit

# 检查 GPU
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### GPU 显存占用高

```bash
# 检查状态
curl http://0.0.0.0:8000/api/gpu/status

# 手动卸载
curl -X POST http://0.0.0.0:8000/api/gpu/offload

# 验证
nvidia-smi
```

### 端口被占用

```bash
# 修改 .env 中的端口
PORT=8001

# 重启
docker-compose down
bash start.sh
```

完整故障排查指南见 [DEPLOYMENT.md](DEPLOYMENT.md)

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feature/amazing`）
3. 提交更改（`git commit -m '添加某某功能'`）
4. 推送到分支（`git push origin feature/amazing`）
5. 提交 Pull Request

## 📝 更新日志

### v1.2.0 (2025-12-06)
- ✨ 新增统一服务器（UI + API + MCP）
- 🧠 实现智能 GPU 显存管理
- 🐳 Docker 部署，自动 GPU 选择
- 📚 完整文档
- 🧪 测试套件

### v1.1.0 (2025-07-09)
- ✨ 支持文生图（T2I）
- 🎨 提升编辑质量
- 📊 更好的指令遵循

### v1.0.0 (2025-04-25)
- 🎉 首次发布
- 🎨 自然语言图像编辑
- 📊 GEdit-Bench 评测

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

特别感谢：
- [Step1X-Edit 团队](https://github.com/stepfun-ai/Step1X-Edit) - 原始模型
- [Kohya](https://github.com/kohya-ss/sd-scripts) - 训练脚本
- [xDiT](https://github.com/xdit-project/xDiT) - 并行推理
- [TeaCache](https://github.com/ali-vilab/TeaCache) - 加速方案
- [HuggingFace](https://huggingface.co) - 模型托管

## 📞 联系与支持

- **GitHub Issues**：[报告问题或功能请求](https://github.com/neosun100/Step1X-Edit/issues)
- **Discord**：[加入社区](https://discord.gg/j3qzuAyn)
- **文档**：[完整文档](DEPLOYMENT.md)

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=neosun100/Step1X-Edit&type=Date)](https://star-history.com/#neosun100/Step1X-Edit)

## 📱 关注公众号

![公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

---

<div align="center">
  由 Step1X-Edit 社区用 ❤️ 制作
  <br>
  <sub>如果这个项目对你有帮助，请给它一个 ⭐️</sub>
</div>
