# MCP (Model Context Protocol) Guide

## Overview

Step1X-Edit provides a Model Context Protocol (MCP) server for programmatic access to image editing capabilities. MCP enables seamless integration with AI assistants and automation tools.

## What is MCP?

Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to LLMs. It enables:

- **Tool-based interaction**: Call functions programmatically
- **Type safety**: Strongly typed parameters and returns
- **Error handling**: Structured error responses
- **Resource sharing**: Shared GPU manager across all tools

## Available Tools

### 1. edit_image

Edit a single image with natural language instruction.

**Parameters:**
- `image_path` (str, required): Path to input image
- `prompt` (str, required): Editing instruction
- `output_path` (str, optional): Path to save result
- `num_steps` (int, default: 28): Inference steps (10-50)
- `guidance_scale` (float, default: 6.0): CFG scale (1.0-15.0)
- `size_level` (int, default: 1024): Resolution (512/768/1024)
- `seed` (int, optional): Random seed

**Returns:**
```json
{
  "status": "success",
  "message": "Image edited successfully",
  "output_path": "/path/to/output.jpg",
  "processing_time": 15.3,
  "metadata": {
    "prompt": "add a red hat",
    "num_steps": 28,
    "guidance_scale": 6.0,
    "size_level": 1024,
    "seed": 42,
    "gpu_location": "CPU",
    "gpu_memory_gb": 0.5
  }
}
```

**Example:**
```python
result = await mcp_client.call_tool(
    "edit_image",
    {
        "image_path": "portrait.jpg",
        "prompt": "add a red hat on the person",
        "num_steps": 28,
        "guidance_scale": 6.0
    }
)
print(f"Saved to: {result['output_path']}")
```

### 2. batch_edit_images

Edit multiple images in batch.

**Parameters:**
- `image_paths` (list[str], required): List of input image paths
- `prompts` (str | list[str], required): Single prompt or list of prompts
- `output_dir` (str, optional): Output directory
- `num_steps` (int, default: 28): Inference steps
- `guidance_scale` (float, default: 6.0): CFG scale
- `size_level` (int, default: 1024): Resolution
- `seed` (int, optional): Random seed

**Returns:**
```json
{
  "status": "success",
  "message": "Processed 3 images: 3 succeeded, 0 failed",
  "results": [
    {
      "input": "img1.jpg",
      "result": { "status": "success", ... }
    },
    ...
  ],
  "summary": {
    "total": 3,
    "successes": 3,
    "failures": 0
  }
}
```

**Example:**
```python
result = await mcp_client.call_tool(
    "batch_edit_images",
    {
        "image_paths": ["img1.jpg", "img2.jpg", "img3.jpg"],
        "prompts": "add a hat",
        "output_dir": "./outputs"
    }
)
```

### 3. get_gpu_status

Get current GPU resource status.

**Parameters:** None

**Returns:**
```json
{
  "status": "success",
  "gpu_status": {
    "model_location": "CPU",
    "idle_time": 45.2,
    "idle_timeout": 60,
    "gpu_memory_allocated_gb": 0.12,
    "gpu_memory_reserved_gb": 0.5,
    "auto_monitor_running": true,
    "statistics": {
      "total_loads": 5,
      "gpu_to_cpu": 5,
      "cpu_to_gpu": 4,
      "full_releases": 1
    }
  }
}
```

**Example:**
```python
status = await mcp_client.call_tool("get_gpu_status", {})
print(f"Model location: {status['gpu_status']['model_location']}")
print(f"GPU memory: {status['gpu_status']['gpu_memory_allocated_gb']}GB")
```

### 4. offload_gpu

Manually offload model from GPU to CPU.

**Parameters:** None

**Returns:**
```json
{
  "status": "success",
  "message": "GPU offloaded. Model now on CPU, GPU memory: 0.12GB",
  "gpu_status": { ... }
}
```

**Example:**
```python
result = await mcp_client.call_tool("offload_gpu", {})
print(result['message'])
```

**Use when:**
- Want to free GPU memory temporarily
- Planning to use model again soon
- Need to run other GPU tasks

### 5. release_gpu

Completely release model from both GPU and CPU.

**Parameters:** None

**Returns:**
```json
{
  "status": "success",
  "message": "GPU released. Model location: Unloaded, GPU memory: 0.05GB",
  "gpu_status": { ... }
}
```

**Example:**
```python
result = await mcp_client.call_tool("release_gpu", {})
print(result['message'])
```

**Use when:**
- Done with editing for extended period
- Need maximum available memory
- Switching to different model/task

## Setup

### 1. Install MCP Client

```bash
pip install fastmcp
```

### 2. Configure MCP Server

Create `mcp_config.json`:

```json
{
  "mcpServers": {
    "step1x-edit": {
      "command": "python3",
      "args": ["mcp_server.py"],
      "env": {
        "MODEL_PATH": "/path/to/Step1X-Edit",
        "GPU_IDLE_TIMEOUT": "60"
      }
    }
  }
}
```

### 3. Start MCP Server

```bash
# Standalone
python3 mcp_server.py

# Or via unified server (auto-starts)
python3 unified_server.py
```

## Usage Examples

### Python Client

```python
from mcp import ClientSession
import asyncio

async def main():
    async with ClientSession() as session:
        # Edit image
        result = await session.call_tool(
            "edit_image",
            {
                "image_path": "input.jpg",
                "prompt": "add a red hat",
                "num_steps": 28
            }
        )
        print(f"Result: {result['output_path']}")
        
        # Check GPU status
        status = await session.call_tool("get_gpu_status", {})
        print(f"GPU: {status['gpu_status']['model_location']}")
        
        # Offload when done
        await session.call_tool("offload_gpu", {})

asyncio.run(main())
```

### CLI Usage

```bash
# Edit image
mcp call edit_image '{
  "image_path": "input.jpg",
  "prompt": "add a red hat",
  "num_steps": 28
}'

# Check status
mcp call get_gpu_status '{}'

# Offload GPU
mcp call offload_gpu '{}'
```

### Batch Processing

```python
async def batch_process():
    async with ClientSession() as session:
        # Process multiple images
        result = await session.call_tool(
            "batch_edit_images",
            {
                "image_paths": [
                    "img1.jpg",
                    "img2.jpg",
                    "img3.jpg"
                ],
                "prompts": [
                    "add a hat",
                    "change to night scene",
                    "add sunglasses"
                ],
                "num_steps": 28
            }
        )
        
        print(f"Processed: {result['summary']['total']}")
        print(f"Success: {result['summary']['successes']}")
        print(f"Failed: {result['summary']['failures']}")
```

## Integration with AI Assistants

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "step1x-edit": {
      "command": "python3",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "MODEL_PATH": "/path/to/Step1X-Edit"
      }
    }
  }
}
```

### Custom Integration

```python
class ImageEditingAssistant:
    def __init__(self):
        self.mcp_client = MCPClient()
    
    async def edit_image(self, image_path: str, instruction: str):
        """Edit image based on natural language instruction."""
        result = await self.mcp_client.call_tool(
            "edit_image",
            {
                "image_path": image_path,
                "prompt": instruction
            }
        )
        return result['output_path']
    
    async def cleanup(self):
        """Free GPU resources."""
        await self.mcp_client.call_tool("offload_gpu", {})
```

## Error Handling

All tools return structured error responses:

```json
{
  "status": "error",
  "error": "Image file not found: /path/to/image.jpg"
}
```

**Common errors:**
- `Image file not found`: Invalid image path
- `Manager not initialized`: Server not ready
- `CUDA out of memory`: Insufficient GPU memory
- `Invalid parameter`: Parameter validation failed

**Handling:**
```python
result = await mcp_client.call_tool("edit_image", params)

if result['status'] == 'error':
    print(f"Error: {result['error']}")
    # Handle error
else:
    print(f"Success: {result['output_path']}")
```

## Performance Tips

1. **Batch processing**: Use `batch_edit_images` for multiple images
2. **Reuse connections**: Keep MCP session open for multiple calls
3. **Manual offload**: Call `offload_gpu` between batches
4. **Monitor status**: Check `get_gpu_status` to track resource usage

## Comparison: MCP vs API

| Feature | MCP | REST API |
|---------|-----|----------|
| **Access** | Programmatic | HTTP |
| **Type Safety** | Strong | Weak |
| **Integration** | AI assistants | Web apps |
| **Overhead** | Low | Medium |
| **Use Case** | Automation | Web services |

**When to use MCP:**
- Integrating with AI assistants
- Building automation workflows
- Need type-safe interface
- Local/programmatic access

**When to use API:**
- Web applications
- Remote access
- HTTP-based integrations
- Need Swagger docs

## Advanced Usage

### Custom Timeout

```python
# Set custom timeout via environment
os.environ['GPU_IDLE_TIMEOUT'] = '300'  # 5 minutes

# Or in mcp_config.json
{
  "env": {
    "GPU_IDLE_TIMEOUT": "300"
  }
}
```

### Monitoring

```python
async def monitor_gpu():
    while True:
        status = await mcp_client.call_tool("get_gpu_status", {})
        print(f"GPU: {status['gpu_status']['model_location']}")
        print(f"Memory: {status['gpu_status']['gpu_memory_allocated_gb']}GB")
        await asyncio.sleep(10)
```

### Workflow Automation

```python
async def automated_workflow(image_dir: str):
    """Process all images in directory."""
    images = list(Path(image_dir).glob("*.jpg"))
    
    for img in images:
        # Edit
        result = await mcp_client.call_tool(
            "edit_image",
            {
                "image_path": str(img),
                "prompt": "enhance quality"
            }
        )
        
        if result['status'] == 'success':
            print(f"✓ {img.name}")
        else:
            print(f"✗ {img.name}: {result['error']}")
    
    # Cleanup
    await mcp_client.call_tool("offload_gpu", {})
```

## Troubleshooting

### MCP Server Not Starting

**Check:**
1. Python version: `python3 --version` (need 3.10+)
2. Dependencies: `pip install fastmcp`
3. Model path: Verify `MODEL_PATH` environment variable

### Connection Refused

**Solutions:**
1. Ensure server is running: `ps aux | grep mcp_server`
2. Check port availability
3. Verify firewall settings

### Slow Response

**Optimize:**
1. Keep model on GPU during batch: Don't offload between items
2. Use `batch_edit_images` for multiple images
3. Increase `GPU_IDLE_TIMEOUT` for frequent use

## Resources

- [MCP Specification](https://modelcontextprotocol.io)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Step1X-Edit GitHub](https://github.com/stepfun-ai/Step1X-Edit)

## Support

For issues or questions:
- GitHub Issues: https://github.com/stepfun-ai/Step1X-Edit/issues
- Discord: https://discord.gg/j3qzuAyn
