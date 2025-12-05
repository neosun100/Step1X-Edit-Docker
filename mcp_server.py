"""
Step1X-Edit MCP Server
======================

Model Context Protocol server for programmatic access to Step1X-Edit.

Features:
- Image editing tool
- GPU status and control tools
- Type-safe interface
- Error handling
- Shared GPU manager across all tools

Usage (from MCP client):
    result = await mcp_client.call_tool(
        "edit_image",
        {
            "image_path": "/path/to/image.jpg",
            "prompt": "add a hat",
            "num_steps": 28
        }
    )
"""

import os
import sys
from typing import Optional
from pathlib import Path

from PIL import Image

# Try to import fastmcp
try:
    from fastmcp import FastMCP
except ImportError:
    print("Error: fastmcp not installed. Install it with: pip install fastmcp")
    sys.exit(1)

from step1x_manager import create_manager_from_env, Step1XEditManager

# Initialize MCP server
mcp = FastMCP("Step1X-Edit")

# Global manager (shared across all tools)
manager: Optional[Step1XEditManager] = None


def get_manager() -> Step1XEditManager:
    """Get or create global manager instance."""
    global manager
    if manager is None:
        manager = create_manager_from_env()
    return manager


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
    Edit an image using Step1X-Edit with natural language instruction.

    This tool loads an image, applies the specified editing instruction,
    and saves the result. The GPU is automatically managed for optimal
    memory usage.

    Args:
        image_path: Path to input image file (PNG, JPG, etc.)
        prompt: Editing instruction in natural language
        output_path: Path to save edited image (optional, defaults to 'output_<filename>')
        num_steps: Number of inference steps (10-50, default: 28, higher = better quality)
        guidance_scale: CFG guidance scale (1.0-15.0, default: 6.0, higher = stronger prompt)
        size_level: Output resolution - 512, 768, or 1024 (default: 1024)
        seed: Random seed for reproducibility (null for random)

    Returns:
        Dictionary with:
        - status: 'success' or 'error'
        - message: Status message
        - output_path: Path to saved edited image
        - processing_time: Time taken in seconds
        - metadata: Additional information

    Example:
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
        print(f"Edited image saved to: {result['output_path']}")
        ```
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

        # Get manager
        mgr = get_manager()

        # Edit image
        import time
        start_time = time.time()

        result_image = mgr.edit_image(
            image=image,
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            size_level=size_level,
            seed=seed,
            show_progress=False
        )

        processing_time = time.time() - start_time

        # Determine output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"output_{input_path.name}")

        # Save result
        result_image.save(output_path)

        # Get GPU status
        gpu_status = mgr.get_gpu_status()

        return {
            'status': 'success',
            'message': 'Image edited successfully',
            'output_path': output_path,
            'processing_time': processing_time,
            'metadata': {
                'prompt': prompt,
                'num_steps': num_steps,
                'guidance_scale': guidance_scale,
                'size_level': size_level,
                'seed': seed,
                'gpu_location': gpu_status['model_location'],
                'gpu_memory_gb': gpu_status['gpu_memory_allocated_gb']
            }
        }

    except Exception as e:
        # Always offload on error
        try:
            if manager is not None:
                manager.manual_offload()
        except:
            pass

        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def batch_edit_images(
    image_paths: list[str],
    prompts: str | list[str],
    output_dir: Optional[str] = None,
    num_steps: int = 28,
    guidance_scale: float = 6.0,
    size_level: int = 1024,
    seed: Optional[int] = None
) -> dict:
    """
    Edit multiple images in batch.

    Args:
        image_paths: List of input image file paths
        prompts: Single prompt for all images, or list of prompts (one per image)
        output_dir: Directory to save edited images (optional, defaults to same dir as input)
        num_steps: Number of inference steps (default: 28)
        guidance_scale: CFG guidance scale (default: 6.0)
        size_level: Output resolution (default: 1024)
        seed: Random seed (null for random)

    Returns:
        Dictionary with status and list of results for each image

    Example:
        ```python
        result = await mcp_client.call_tool(
            "batch_edit_images",
            {
                "image_paths": ["img1.jpg", "img2.jpg"],
                "prompts": "add a hat",
                "num_steps": 28
            }
        )
        ```
    """
    try:
        # Validate inputs
        if isinstance(prompts, str):
            prompts_list = [prompts] * len(image_paths)
        else:
            if len(prompts) != len(image_paths):
                return {
                    'status': 'error',
                    'error': 'Number of prompts must match number of images'
                }
            prompts_list = prompts

        # Process each image
        results = []
        for i, (img_path, prompt) in enumerate(zip(image_paths, prompts_list)):
            print(f"Processing image {i+1}/{len(image_paths)}: {img_path}")

            # Determine output path
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"output_{os.path.basename(img_path)}")
            else:
                output_path = None

            # Edit image
            result = edit_image(
                image_path=img_path,
                prompt=prompt,
                output_path=output_path,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                size_level=size_level,
                seed=seed
            )

            results.append({
                'input': img_path,
                'result': result
            })

        # Count successes and failures
        successes = sum(1 for r in results if r['result']['status'] == 'success')
        failures = len(results) - successes

        return {
            'status': 'success' if failures == 0 else 'partial',
            'message': f"Processed {len(results)} images: {successes} succeeded, {failures} failed",
            'results': results,
            'summary': {
                'total': len(results),
                'successes': successes,
                'failures': failures
            }
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def get_gpu_status() -> dict:
    """
    Get current GPU resource status.

    Returns detailed information about the model and GPU state,
    including:
    - Model location (GPU, CPU, or Unloaded)
    - Idle time since last use
    - GPU memory usage
    - Statistics (total loads, offloads, etc.)

    Returns:
        Dictionary with GPU status information

    Example:
        ```python
        status = await mcp_client.call_tool("get_gpu_status", {})
        print(f"Model is on: {status['model_location']}")
        print(f"GPU memory: {status['gpu_memory_allocated_gb']}GB")
        ```
    """
    try:
        mgr = get_manager()
        status = mgr.get_gpu_status()

        return {
            'status': 'success',
            'gpu_status': status
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


@mcp.tool()
def offload_gpu() -> dict:
    """
    Manually offload model from GPU to CPU.

    This frees GPU memory while keeping the model cached in RAM.
    Next request will reload from CPU to GPU quickly (2-5 seconds).

    Use this when:
    - You want to free GPU memory temporarily
    - Planning to use the model again soon
    - Need to run other GPU tasks

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
        mgr.manual_offload()

        status = mgr.get_gpu_status()

        return {
            'status': 'success',
            'message': f"GPU offloaded. Model now on {status['model_location']}, "
                      f"GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB",
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
    Completely release model from both GPU and CPU.

    This frees all memory but requires full reload on next request (20-30 seconds).

    Use this when:
    - Done with image editing for an extended period
    - Need maximum available memory
    - Switching to a different model/task

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
        mgr.manual_release()

        status = mgr.get_gpu_status()

        return {
            'status': 'success',
            'message': f"GPU released. Model location: {status['model_location']}, "
                      f"GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB",
            'gpu_status': status
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


# ============================================================================
# Run MCP Server
# ============================================================================

if __name__ == "__main__":
    # Run MCP server
    mcp.run()
