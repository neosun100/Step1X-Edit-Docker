"""
Test version of API server - works without actual models
"""

import os
import io
import time
from typing import Optional
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
import torch

from gpu_manager import GPUResourceManager

# Initialize FastAPI app
app = FastAPI(
    title="Step1X-Edit API (Test Mode)",
    description="Test version without actual models",
    version="1.0.0-test",
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

# Global GPU manager
gpu_manager: Optional[GPUResourceManager] = None


@app.on_event("startup")
async def startup_event():
    """Initialize GPU manager on startup."""
    global gpu_manager
    gpu_manager = GPUResourceManager(
        idle_timeout=60,
        device="cuda:1"  # Use GPU 1
    )
    print("✓ GPU Manager initialized")


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


# Mock model for testing
class MockImageProcessor:
    def __init__(self):
        self.data = torch.randn(100, 100)

    def to(self, device):
        self.data = self.data.to(device)
        return self

    def process(self, image, prompt):
        """Mock image processing - adds text overlay."""
        # Add text overlay to show processing
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = None

        # Add semi-transparent overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 128))
        draw_overlay = ImageDraw.Draw(overlay)

        text = f"Processed: {prompt[:30]}"
        draw_overlay.text((10, 10), text, fill=(255, 255, 255, 255), font=font)

        # Composite
        image = image.convert('RGBA')
        image = Image.alpha_composite(image, overlay)
        return image.convert('RGB')


def load_mock_model():
    """Load mock model."""
    print("Loading mock model...")
    time.sleep(0.5)  # Simulate loading
    return MockImageProcessor()


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "name": "Step1X-Edit API (Test Mode)",
        "version": "1.0.0-test",
        "mode": "testing",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0

        return {
            "status": "healthy",
            "mode": "test",
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "gpu_manager_initialized": gpu_manager is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/edit", response_model=EditImageResponse, tags=["Image Editing"])
async def edit_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    num_steps: int = Form(28),
    guidance_scale: float = Form(6.0),
    size_level: int = Form(1024),
    seed: Optional[int] = Form(None),
):
    """
    Edit an image (test mode - adds text overlay).
    """
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get model with GPU manager
        start_time = time.time()
        model = gpu_manager.get_model(load_func=load_mock_model)

        # Process (mock)
        result_image = model.process(image, prompt)

        # Offload GPU
        gpu_manager.force_offload()

        processing_time = time.time() - start_time

        # Convert to base64
        import base64
        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Get GPU status
        gpu_status = gpu_manager.get_status()

        return EditImageResponse(
            success=True,
            message="Image edited successfully (test mode)",
            processing_time=processing_time,
            image_base64=img_base64,
            metadata={
                "prompt": prompt,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "size_level": size_level,
                "seed": seed,
                "gpu_location": gpu_status["model_location"],
                "mode": "test"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu/status", response_model=GPUStatusResponse, tags=["GPU Management"])
async def get_gpu_status():
    """Get current GPU resource status."""
    try:
        status = gpu_manager.get_status()
        return GPUStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpu/offload", response_model=MessageResponse, tags=["GPU Management"])
async def offload_gpu():
    """Manually offload model from GPU to CPU."""
    try:
        gpu_manager.force_offload()
        status = gpu_manager.get_status()

        return MessageResponse(
            success=True,
            message=f"GPU offloaded. Model now on {status['model_location']}, "
                    f"GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gpu/release", response_model=MessageResponse, tags=["GPU Management"])
async def release_gpu():
    """Completely release model from both GPU and CPU."""
    try:
        gpu_manager.force_release()
        status = gpu_manager.get_status()

        return MessageResponse(
            success=True,
            message=f"GPU released. Model location: {status['model_location']}, "
                    f"GPU memory: {status['gpu_memory_allocated_gb']:.2f}GB"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
