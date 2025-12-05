"""
Step1X-Edit MCP Server (Test Mode)
====================================

Test version with mock models for MCP validation.
"""

import os
import sys
import time
from typing import Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

try:
    from fastmcp import FastMCP
except ImportError:
    print("Error: fastmcp not installed. Install it with: pip install fastmcp")
    sys.exit(1)

from gpu_manager import GPUResourceManager

# Initialize MCP server
mcp = FastMCP("Step1X-Edit-Test")

# Global GPU manager (shared across all tools)
gpu_manager: Optional[GPUResourceManager] = None


# Mock model for testing
class MockImageProcessor:
    def __init__(self):
        self.data = torch.randn(100, 100)

    def to(self, device):
        self.data = self.data.to(device)
        return self

    def process(self, image, prompt, size_level):
        """Mock image processing - adds text overlay and resizes."""
        # Resize image to target size
        image = image.resize((size_level, size_level))

        # Add text overlay to show processing
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        except:
            font = None

        # Add semi-transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 128))
        draw_overlay = ImageDraw.Draw(overlay)

        # Multiple lines of text
        lines = [
            f"MCP TEST MODE",
            f"Prompt: {prompt[:40]}",
            f"Size: {size_level}x{size_level}",
        ]

        y_offset = 10
        for line in lines:
            draw_overlay.text((10, y_offset), line, fill=(255, 255, 255, 255), font=font)
            y_offset += 40

        # Composite
        image = image.convert('RGBA')
        image = Image.alpha_composite(image, overlay)
        return image.convert('RGB')


def load_mock_model():
    """Load mock model."""
    print("[MCP] Loading mock model...")
    time.sleep(0.5)  # Simulate loading
    return MockImageProcessor()


def get_manager() -> GPUResourceManager:
    """Get or create global manager instance."""
    global gpu_manager
    if gpu_manager is None:
        gpu_manager = GPUResourceManager(
            idle_timeout=60,
            device="cuda:1"  # Use GPU 1
        )
        print("[MCP] ✓ GPU Manager initialized")
    return gpu_manager


@mcp.tool()
def edit_image(
    image_path: str,
    prompt: str,
    output_path: Optional[str] = None,
    num_steps: int = 28,
    guidance_scale: float = 6.0,
    size_level: int = 1024,
    seed: Optional[int] = None
) -> dict:
    """
    Edit an image using Step1X-Edit (TEST MODE).

    This is a test version that uses mock models to validate the MCP interface.
    The output will have a text overlay showing "MCP TEST MODE".

    Args:
        image_path: Path to input image file
        prompt: Editing instruction
        output_path: Path to save edited image (optional)
        num_steps: Number of inference steps (mock, not used)
        guidance_scale: CFG guidance scale (mock, not used)
        size_level: Output resolution - 512, 768, or 1024
        seed: Random seed (mock, not used)

    Returns:
        Dictionary with status, output path, and metadata
    """
    try:
        # Validate image path
        if not os.path.exists(image_path):
            return {
                'status': 'error',
                'error': f"Image file not found: {image_path}"
            }

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Get manager and model
        mgr = get_manager()
        model = mgr.get_model(load_func=load_mock_model)

        # Process (mock)
        start_time = time.time()
        result_image = model.process(image, prompt, size_level)

        # Offload GPU
        mgr.force_offload()

        processing_time = time.time() - start_time

        # Determine output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"output_mcp_{input_path.name}")

        # Save result
        result_image.save(output_path)

        # Get GPU status
        gpu_status = mgr.get_status()

        return {
            'status': 'success',
            'message': 'Image edited successfully (test mode)',
            'output_path': output_path,
            'processing_time': processing_time,
            'metadata': {
                'prompt': prompt,
                'num_steps': num_steps,
                'guidance_scale': guidance_scale,
                'size_level': size_level,
                'seed': seed,
                'gpu_location': gpu_status['model_location'],
                'gpu_memory_gb': gpu_status['gpu_memory_allocated_gb'],
                'mode': 'test'
            }
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }


@mcp.tool()
def get_gpu_status() -> dict:
    """
    Get current GPU resource status.

    Returns detailed information about GPU memory usage, model location,
    and statistics for monitoring and debugging.

    Returns:
        Dictionary with GPU status information including:
        - model_location: Where the model is currently located (GPU/CPU/Unloaded)
        - idle_time: Seconds since last use
        - gpu_memory_allocated_gb: Current GPU memory usage
        - statistics: Usage statistics

    Example:
        ```python
        status = await mcp_client.call_tool("get_gpu_status", {})
        print(f"Model location: {status['model_location']}")
        print(f"GPU memory: {status['gpu_memory_allocated_gb']:.2f} GB")
        ```
    """
    try:
        mgr = get_manager()
        status = mgr.get_status()
        status['status'] = 'success'
        return status

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def offload_gpu() -> dict:
    """
    Manually offload model from GPU to CPU.

    Use this to free GPU memory while keeping the model cached in CPU memory
    for fast reload. This is useful for sharing GPU resources with other tasks.

    Returns:
        Dictionary with status and new GPU state

    Example:
        ```python
        result = await mcp_client.call_tool("offload_gpu", {})
        print(result['message'])
        ```
    """
    try:
        mgr = get_manager()
        mgr.force_offload()
        status = mgr.get_status()

        return {
            'status': 'success',
            'message': f"GPU offloaded. Model now on {status['model_location']}, GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB",
            'gpu_status': status
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def release_gpu() -> dict:
    """
    Completely release model from both GPU and CPU memory.

    Use this to fully unload the model and free all memory. The model will need
    to be reloaded from disk on the next edit request.

    Returns:
        Dictionary with status and new GPU state

    Example:
        ```python
        result = await mcp_client.call_tool("release_gpu", {})
        print(result['message'])
        ```
    """
    try:
        mgr = get_manager()
        mgr.force_release()
        status = mgr.get_status()

        return {
            'status': 'success',
            'message': f"GPU released. Model location: {status['model_location']}, GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB",
            'gpu_status': status
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def server_info() -> dict:
    """
    Get information about the MCP server.

    Returns:
        Dictionary with server information
    """
    return {
        'name': 'Step1X-Edit-Test',
        'version': '1.0.0-test',
        'mode': 'test',
        'tools': ['edit_image', 'get_gpu_status', 'offload_gpu', 'release_gpu', 'server_info'],
        'description': 'Test version of Step1X-Edit MCP server using mock models'
    }


if __name__ == "__main__":
    print("="*60)
    print("Step1X-Edit MCP Server (Test Mode)")
    print("="*60)
    print("Starting MCP server...")
    print("This is a TEST version using mock models")
    print("")
    print("Available tools:")
    print("  - edit_image: Edit images with text overlays (test mode)")
    print("  - get_gpu_status: Get GPU resource status")
    print("  - offload_gpu: Offload model to CPU")
    print("  - release_gpu: Release model completely")
    print("  - server_info: Get server information")
    print("="*60)
    print("")

    # Run MCP server
    mcp.run()
