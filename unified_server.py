"""
Step1X-Edit Unified Server
===========================

Single server providing:
- UI: Web interface at /
- API: RESTful endpoints at /api/*
- MCP: Model Context Protocol (separate process)
- Swagger: API docs at /docs

All modes share the same GPU manager for optimal resource usage.
"""

import os
import io
import time
import base64
import asyncio
import subprocess
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

from step1x_manager import create_manager_from_env, Step1XEditManager

# Initialize FastAPI
app = FastAPI(
    title="Step1X-Edit Unified Server",
    description="AI Image Editing: UI + API + MCP",
    version="1.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global manager
manager: Optional[Step1XEditManager] = None
mcp_process: Optional[subprocess.Popen] = None


@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    global manager, mcp_process
    
    print("🚀 Starting Step1X-Edit Unified Server...")
    
    # Initialize manager
    manager = create_manager_from_env()
    print("✓ GPU Manager initialized")
    
    # Start MCP server if enabled
    if os.getenv("ENABLE_MCP", "true").lower() == "true":
        try:
            mcp_process = subprocess.Popen(
                ["python3", "mcp_server.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print("✓ MCP Server started")
        except Exception as e:
            print(f"⚠ MCP Server failed to start: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global mcp_process
    
    if mcp_process:
        mcp_process.terminate()
        mcp_process.wait()
        print("✓ MCP Server stopped")


# ============================================================================
# Models
# ============================================================================

class EditRequest(BaseModel):
    prompt: str = Field(..., description="Editing instruction")
    num_steps: int = Field(28, ge=10, le=50)
    guidance_scale: float = Field(6.0, ge=1.0, le=15.0)
    size_level: int = Field(1024, description="512, 768, or 1024")
    seed: Optional[int] = None


class GPUStatus(BaseModel):
    model_location: str
    idle_time: float
    gpu_memory_allocated_gb: float
    gpu_memory_reserved_gb: float
    statistics: dict


# ============================================================================
# Health & Status
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "gpu_available": manager is not None
    }


@app.get("/api/gpu/status", response_model=GPUStatus)
async def get_gpu_status():
    """Get GPU status."""
    if not manager:
        raise HTTPException(500, "Manager not initialized")
    
    status = manager.get_gpu_status()
    return status


@app.post("/api/gpu/offload")
async def offload_gpu():
    """Manually offload GPU to CPU."""
    if not manager:
        raise HTTPException(500, "Manager not initialized")
    
    manager.manual_offload()
    status = manager.get_gpu_status()
    
    return {
        "success": True,
        "message": f"GPU offloaded. Model now on {status['model_location']}",
        "gpu_status": status
    }


@app.post("/api/gpu/release")
async def release_gpu():
    """Completely release GPU and CPU cache."""
    if not manager:
        raise HTTPException(500, "Manager not initialized")
    
    manager.manual_release()
    status = manager.get_gpu_status()
    
    return {
        "success": True,
        "message": "GPU fully released",
        "gpu_status": status
    }


# ============================================================================
# Image Editing API
# ============================================================================

@app.post("/api/edit")
async def edit_image_api(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    num_steps: int = Form(28),
    guidance_scale: float = Form(6.0),
    size_level: int = Form(1024),
    seed: Optional[int] = Form(None)
):
    """
    Edit image via API.
    
    Returns edited image as base64 or binary stream.
    """
    if not manager:
        raise HTTPException(500, "Manager not initialized")
    
    try:
        # Load image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Edit
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
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return as image
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Prompt": prompt
            }
        )
        
    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================================
# UI
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve web UI."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Step1X-Edit - AI Image Editor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        
        .content { padding: 40px; }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 30px;
        }
        
        .upload-area:hover { background: #f8f9ff; border-color: #764ba2; }
        .upload-area.dragover { background: #e8ebff; transform: scale(1.02); }
        
        .params {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .param-group {
            background: #f8f9ff;
            padding: 20px;
            border-radius: 10px;
        }
        
        .param-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        
        .param-group input, .param-group select, .param-group textarea {
            width: 100%;
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border 0.3s;
        }
        
        .param-group input:focus, .param-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .param-group textarea { min-height: 80px; resize: vertical; }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
        }
        
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        .progress {
            display: none;
            margin: 20px 0;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .progress-bar {
            height: 30px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .results {
            display: none;
            margin-top: 30px;
        }
        
        .image-compare {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        
        .image-box {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .image-box img { width: 100%; display: block; }
        .image-box .label {
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: center;
            font-weight: 600;
        }
        
        .gpu-status {
            background: #f8f9ff;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .gpu-status .info { font-size: 14px; color: #666; }
        .gpu-status .btn-small {
            padding: 8px 16px;
            font-size: 14px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        
        .github-buttons {
            background: linear-gradient(135deg, #f8f9ff 0%, #e8ebff 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
        }
        
        .github-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 600;
            font-size: 15px;
            transition: all 0.3s;
            border: 2px solid transparent;
        }
        
        .github-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }
        
        .github-btn-star {
            background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%);
            color: #333;
            border-color: #ffc107;
            font-size: 16px;
            padding: 14px 28px;
            animation: pulse 2s infinite;
        }
        
        .github-btn-star:hover {
            background: linear-gradient(135deg, #ffed4e 0%, #ffd700 100%);
            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
        }
        
        .github-btn-bug {
            background: #fff;
            color: #e74c3c;
            border-color: #e74c3c;
        }
        
        .github-btn-bug:hover {
            background: #e74c3c;
            color: #fff;
        }
        
        .github-btn-feature {
            background: #fff;
            color: #667eea;
            border-color: #667eea;
        }
        
        .github-btn-feature:hover {
            background: #667eea;
            color: #fff;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        @media (max-width: 768px) {
            .image-compare { grid-template-columns: 1fr; }
            .params { grid-template-columns: 1fr; }
            .github-buttons { flex-direction: column; }
            .github-btn { width: 100%; justify-content: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Step1X-Edit</h1>
            <p>AI-Powered Image Editing with Natural Language</p>
        </div>
        
        <div class="content">
            <div class="github-buttons">
                <a href="https://github.com/neosun100/Step1X-Edit-Docker" target="_blank" class="github-btn github-btn-star">
                    ⭐ Star on GitHub
                </a>
                <a href="https://github.com/neosun100/Step1X-Edit-Docker/issues/new?labels=bug&template=bug_report.md" target="_blank" class="github-btn github-btn-bug">
                    🐛 Report Bug
                </a>
                <a href="https://github.com/neosun100/Step1X-Edit-Docker/issues/new?labels=enhancement&template=feature_request.md" target="_blank" class="github-btn github-btn-feature">
                    💡 Request Feature
                </a>
            </div>
            
            <div class="upload-area" id="uploadArea">
                <input type="file" id="fileInput" accept="image/*" style="display:none">
                <h3>📁 Click or Drag Image Here</h3>
                <p>Supports: JPG, PNG, WebP</p>
            </div>
            
            <div class="params">
                <div class="param-group">
                    <label>✏️ Editing Instruction</label>
                    <textarea id="prompt" placeholder="e.g., add a red hat on the person"></textarea>
                </div>
                
                <div class="param-group">
                    <label>🎯 Steps (10-50)</label>
                    <input type="number" id="numSteps" value="28" min="10" max="50">
                </div>
                
                <div class="param-group">
                    <label>⚡ Guidance Scale (1-15)</label>
                    <input type="number" id="guidanceScale" value="6.0" min="1" max="15" step="0.1">
                </div>
                
                <div class="param-group">
                    <label>📐 Resolution</label>
                    <select id="sizeLevel">
                        <option value="512">512px</option>
                        <option value="768">768px</option>
                        <option value="1024" selected>1024px</option>
                    </select>
                </div>
                
                <div class="param-group">
                    <label>🎲 Seed (optional)</label>
                    <input type="number" id="seed" placeholder="Random">
                </div>
            </div>
            
            <button class="btn" id="editBtn" disabled>🚀 Edit Image</button>
            
            <div class="progress" id="progress">
                <div class="progress-bar" id="progressBar">Processing...</div>
            </div>
            
            <div class="results" id="results">
                <h2>Results</h2>
                <div class="image-compare">
                    <div class="image-box">
                        <div class="label">Original</div>
                        <img id="originalImg" alt="Original">
                    </div>
                    <div class="image-box">
                        <div class="label">Edited</div>
                        <img id="editedImg" alt="Edited">
                    </div>
                </div>
            </div>
            
            <div class="gpu-status">
                <div class="info" id="gpuInfo">GPU: Loading...</div>
                <button class="btn-small" onclick="offloadGPU()">Free GPU</button>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        // Upload area
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const editBtn = document.getElementById('editBtn');
        
        uploadArea.onclick = () => fileInput.click();
        
        fileInput.onchange = (e) => {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                uploadArea.innerHTML = `<h3>✓ ${selectedFile.name}</h3><p>Click to change</p>`;
                editBtn.disabled = false;
            }
        };
        
        // Drag & drop
        uploadArea.ondragover = (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        };
        
        uploadArea.ondragleave = () => uploadArea.classList.remove('dragover');
        
        uploadArea.ondrop = (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            selectedFile = e.dataTransfer.files[0];
            if (selectedFile) {
                uploadArea.innerHTML = `<h3>✓ ${selectedFile.name}</h3><p>Click to change</p>`;
                editBtn.disabled = false;
            }
        };
        
        // Edit image
        editBtn.onclick = async () => {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('prompt', document.getElementById('prompt').value);
            formData.append('num_steps', document.getElementById('numSteps').value);
            formData.append('guidance_scale', document.getElementById('guidanceScale').value);
            formData.append('size_level', document.getElementById('sizeLevel').value);
            
            const seed = document.getElementById('seed').value;
            if (seed) formData.append('seed', seed);
            
            // Show progress
            document.getElementById('progress').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            editBtn.disabled = true;
            
            try {
                const response = await fetch('/api/edit', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error('Edit failed');
                
                const blob = await response.blob();
                const editedUrl = URL.createObjectURL(blob);
                const originalUrl = URL.createObjectURL(selectedFile);
                
                document.getElementById('originalImg').src = originalUrl;
                document.getElementById('editedImg').src = editedUrl;
                document.getElementById('results').style.display = 'block';
                
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('progress').style.display = 'none';
                editBtn.disabled = false;
                updateGPUStatus();
            }
        };
        
        // GPU status
        async function updateGPUStatus() {
            try {
                const response = await fetch('/api/gpu/status');
                const data = await response.json();
                document.getElementById('gpuInfo').textContent = 
                    `GPU: ${data.model_location} | Memory: ${data.gpu_memory_allocated_gb}GB`;
            } catch (error) {
                console.error('Failed to update GPU status:', error);
            }
        }
        
        async function offloadGPU() {
            try {
                await fetch('/api/gpu/offload', { method: 'POST' });
                updateGPUStatus();
            } catch (error) {
                alert('Failed to offload GPU');
            }
        }
        
        // Update GPU status every 10s
        setInterval(updateGPUStatus, 10000);
        updateGPUStatus();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         Step1X-Edit Unified Server Starting...          ║
╚══════════════════════════════════════════════════════════╝

  UI:      http://{host}:{port}
  API:     http://{host}:{port}/docs
  Health:  http://{host}:{port}/health

""")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
