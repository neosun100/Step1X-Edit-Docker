"""
Step1X-Edit API Server
======================

RESTful API for Step1X-Edit with Swagger documentation.

Features:
- Image editing endpoint
- GPU status and control
- Async processing
- Swagger UI documentation
- Error handling
"""

import os
import io
import time
import base64
from typing import Optional
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import torch
import uvicorn

from step1x_manager import create_manager_from_env, Step1XEditManager

# Initialize FastAPI app
app = FastAPI(
    title="Step1X-Edit API",
    description="AI-powered image editing with natural language instructions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global manager instance
manager: Optional[Step1XEditManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize manager on startup."""
    global manager
    manager = create_manager_from_env()


# ============================================================================
# Models
# ============================================================================

class SizeLevel(int, Enum):
    """Resolution options."""
    SIZE_512 = 512
    SIZE_768 = 768
    SIZE_1024 = 1024


class EditImageRequest(BaseModel):
    """Request model for image editing."""
    prompt: str = Field(..., description="Editing instruction in natural language")
    num_steps: int = Field(28, ge=10, le=50, description="Number of inference steps")
    guidance_scale: float = Field(6.0, ge=1.0, le=15.0, description="Guidance scale (CFG)")
    size_level: SizeLevel = Field(1024, description="Output resolution")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility (null for random)")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "add a red hat on the person",
                "num_steps": 28,
                "guidance_scale": 6.0,
                "size_level": 1024,
                "seed": 42
            }
        }


class EditImageResponse(BaseModel):
    """Response model for image editing."""
    success: bool
    message: str
    processing_time: float
    image_base64: Optional[str] = None
    metadata: dict


class GPUStatusResponse(BaseModel):
    """Response model for GPU status."""
    model_location: str
    idle_time: float
    idle_timeout: int
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    auto_monitor_running: bool
    statistics: dict


class MessageResponse(BaseModel):
    """Generic message response."""
    success: bool
    message: str


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "name": "Step1X-Edit API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check if GPU is available
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0

        return {
            "status": "healthy",
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "manager_initialized": manager is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/edit", response_model=EditImageResponse, tags=["Image Editing"])
async def edit_image(
    file: UploadFile = File(..., description="Input image file"),
    prompt: str = Form(..., description="Editing instruction"),
    num_steps: int = Form(28, description="Number of inference steps"),
    guidance_scale: float = Form(6.0, description="Guidance scale"),
    size_level: int = Form(1024, description="Output resolution"),
    seed: Optional[int] = Form(None, description="Random seed (null for random)"),
):
    """
    Edit an image with natural language instruction.

    Args:
        file: Input image file (PNG, JPG, etc.)
        prompt: Editing instruction in natural language
        num_steps: Number of inference steps (10-50, default: 28)
        guidance_scale: CFG guidance scale (1.0-15.0, default: 6.0)
        size_level: Output resolution (512, 768, or 1024, default: 1024)
        seed: Random seed for reproducibility (null for random)

    Returns:
        Edited image as base64 string with metadata
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Edit image
        start_time = time.time()
        result_image = manager.edit_image(
            image=image,
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            size_level=size_level,
            seed=seed,
            show_progress=False
        )
        processing_time = time.time() - start_time

        # Convert to base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Get GPU status
        gpu_status = manager.get_gpu_status()

        return EditImageResponse(
            success=True,
            message="Image edited successfully",
            processing_time=processing_time,
            image_base64=img_base64,
            metadata={
                "prompt": prompt,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "size_level": size_level,
                "seed": seed,
                "gpu_location": gpu_status["model_location"]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/edit-file", tags=["Image Editing"])
async def edit_image_file(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    num_steps: int = Form(28),
    guidance_scale: float = Form(6.0),
    size_level: int = Form(1024),
    seed: Optional[int] = Form(None),
):
    """
    Edit an image and return the result as a file.

    Same parameters as /api/edit but returns PNG file directly.
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Edit image
        result_image = manager.edit_image(
            image=image,
            prompt=prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            size_level=size_level,
            seed=seed,
            show_progress=False
        )

        # Convert to bytes
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        buffered.seek(0)

        return StreamingResponse(
            buffered,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=edited_{file.filename}"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/status", response_model=GPUStatusResponse, tags=["GPU Management"])
async def get_gpu_status():
    """
    Get current GPU resource status.

    Returns detailed information about:
    - Model location (GPU/CPU/Unloaded)
    - Idle time
    - GPU memory usage
    - Statistics (loads, offloads, etc.)
    """
    try:
        status = manager.get_gpu_status()
        return GPUStatusResponse(**status)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpu/offload", response_model=MessageResponse, tags=["GPU Management"])
async def offload_gpu():
    """
    Manually offload model from GPU to CPU.

    This frees GPU memory while keeping the model cached in RAM
    for quick reload (2-5 seconds).
    """
    try:
        manager.manual_offload()
        status = manager.get_gpu_status()

        return MessageResponse(
            success=True,
            message=f"GPU offloaded. Model now on {status['model_location']}, "
                    f"GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpu/release", response_model=MessageResponse, tags=["GPU Management"])
async def release_gpu():
    """
    Completely release model from both GPU and CPU.

    This frees all memory but requires full reload on next request (20-30 seconds).
    """
    try:
        manager.manual_release()
        status = manager.get_gpu_status()

        return MessageResponse(
            success=True,
            message=f"GPU released. Model location: {status['model_location']}, "
                    f"GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
