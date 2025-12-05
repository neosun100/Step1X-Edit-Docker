"""
Step1X-Edit Model Manager with GPU Resource Management
=======================================================

Integrates GPU Resource Manager with Step1X-Edit ImageGenerator.
"""

import os
import torch
from typing import Optional, Union, List
from PIL import Image
import numpy as np

from gpu_manager import GPUResourceManager
from inference import ImageGenerator
import logging

logger = logging.getLogger(__name__)


class Step1XEditManager:
    """
    Managed wrapper for Step1X-Edit ImageGenerator with GPU resource management.

    This class integrates the GPU manager with the image generation pipeline,
    ensuring efficient GPU memory usage through lazy loading and instant offloading.

    Usage:
        # Initialize
        manager = Step1XEditManager(
            dit_path="path/to/dit",
            ae_path="path/to/ae",
            qwen2vl_model_path="path/to/qwen2vl",
            gpu_idle_timeout=60
        )

        # Edit image
        result = manager.edit_image(
            image="input.jpg",
            prompt="add a hat",
            num_steps=28,
            guidance_scale=6.0
        )

        # GPU is automatically offloaded after processing
    """

    def __init__(
        self,
        dit_path: Optional[str] = None,
        ae_path: Optional[str] = None,
        qwen2vl_model_path: Optional[str] = None,
        device: str = "cuda",
        max_length: int = 640,
        dtype: torch.dtype = torch.bfloat16,
        quantized: bool = False,
        offload: bool = False,
        lora: Optional[str] = None,
        mode: str = "flash",
        version: str = "v1.0",
        gpu_idle_timeout: int = 60,
        auto_offload: bool = True,
    ):
        """
        Initialize Step1X-Edit Manager.

        Args:
            dit_path: Path to DiT model weights
            ae_path: Path to AutoEncoder weights
            qwen2vl_model_path: Path to Qwen2VL model
            device: CUDA device (default: "cuda")
            max_length: Maximum text length (default: 640)
            dtype: Model dtype (default: torch.bfloat16)
            quantized: Use FP8 quantization (default: False)
            offload: Enable CPU offload for modules (default: False)
            lora: Path to LoRA weights (optional)
            mode: Attention mode (default: "flash")
            version: Model version (default: "v1.0")
            gpu_idle_timeout: GPU idle timeout in seconds (default: 60)
            auto_offload: Automatically offload after each task (default: True)
        """
        self.dit_path = dit_path
        self.ae_path = ae_path
        self.qwen2vl_model_path = qwen2vl_model_path
        self.device = device
        self.max_length = max_length
        self.dtype = dtype
        self.quantized = quantized
        self.offload = offload
        self.lora = lora
        self.mode = mode
        self.version = version
        self.auto_offload = auto_offload

        # Initialize GPU manager
        self.gpu_manager = GPUResourceManager(
            idle_timeout=gpu_idle_timeout,
            device=device
        )

        # Model will be lazily loaded
        self.generator = None

        logger.info("Step1X-Edit Manager initialized")
        logger.info(f"  Device: {device}")
        logger.info(f"  Quantized: {quantized}")
        logger.info(f"  Offload: {offload}")
        logger.info(f"  GPU idle timeout: {gpu_idle_timeout}s")
        logger.info(f"  Auto-offload: {auto_offload}")

    def _load_generator(self) -> ImageGenerator:
        """
        Load ImageGenerator (called once on first request).

        Returns:
            Initialized ImageGenerator
        """
        logger.info("Loading Step1X-Edit models...")

        generator = ImageGenerator(
            dit_path=self.dit_path,
            ae_path=self.ae_path,
            qwen2vl_model_path=self.qwen2vl_model_path,
            device=self.device,
            max_length=self.max_length,
            dtype=self.dtype,
            quantized=self.quantized,
            offload=self.offload,
            lora=self.lora,
            mode=self.mode,
            version=self.version
        )

        logger.info("✓ Step1X-Edit models loaded successfully")
        return generator

    def _get_generator(self) -> ImageGenerator:
        """
        Get generator with lazy loading via GPU manager.

        Returns:
            ImageGenerator on GPU
        """
        # Use GPU manager for lazy loading
        generator = self.gpu_manager.get_model(load_func=self._load_generator)
        return generator

    def edit_image(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor],
        prompt: str,
        num_steps: int = 28,
        guidance_scale: float = 6.0,
        size_level: int = 1024,
        seed: Optional[int] = None,
        show_progress: bool = True,
    ) -> Image.Image:
        """
        Edit an image using Step1X-Edit.

        Args:
            image: Input image (file path, PIL Image, numpy array, or torch tensor)
            prompt: Editing instruction
            num_steps: Number of inference steps (default: 28)
            guidance_scale: CFG guidance scale (default: 6.0)
            size_level: Resolution level - 512, 768, or 1024 (default: 1024)
            seed: Random seed (optional)
            show_progress: Show progress bar (default: True)

        Returns:
            Edited image as PIL Image
        """
        try:
            # Step 1: Get generator (lazy load if needed)
            logger.info("Starting image editing...")
            generator = self._get_generator()

            # Step 2: Load and preprocess image
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                from PIL import Image
                image = Image.fromarray(image)
            elif isinstance(image, torch.Tensor):
                from torchvision.transforms import functional as F
                image = F.to_pil_image(image)

            # Step 3: Generate
            logger.info(f"Generating with prompt: {prompt}")
            logger.info(f"  Steps: {num_steps}, CFG: {guidance_scale}, Size: {size_level}")

            if seed is not None:
                import torch
                torch.manual_seed(seed)

            # Call generator
            result = generator.generate(
                image=image,
                prompt=prompt,
                num_steps=num_steps,
                size_level=size_level,
                guidance_scale=guidance_scale,
                show_progress=show_progress
            )

            logger.info("✓ Image editing completed")

            # Step 4: Auto-offload GPU (CRITICAL!)
            if self.auto_offload:
                self.gpu_manager.force_offload()

            return result

        except Exception as e:
            logger.error(f"Error during image editing: {e}")
            # Always offload on error
            if self.auto_offload:
                self.gpu_manager.force_offload()
            raise

    def batch_edit(
        self,
        images: List[Union[str, Image.Image]],
        prompts: Union[str, List[str]],
        **kwargs
    ) -> List[Image.Image]:
        """
        Edit multiple images.

        Args:
            images: List of input images
            prompts: Single prompt or list of prompts (one per image)
            **kwargs: Additional arguments for edit_image

        Returns:
            List of edited images
        """
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)

        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")

        results = []
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.edit_image(image, prompt, **kwargs)
            results.append(result)

        return results

    def get_gpu_status(self) -> dict:
        """
        Get current GPU status.

        Returns:
            Dictionary with GPU status information
        """
        return self.gpu_manager.get_status()

    def manual_offload(self):
        """Manually offload GPU memory to CPU."""
        self.gpu_manager.force_offload()
        logger.info("Manual GPU offload completed")

    def manual_release(self):
        """Manually release all GPU and CPU memory."""
        self.gpu_manager.force_release()
        logger.info("Manual GPU release completed")

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'gpu_manager'):
            self.gpu_manager.stop_monitor()


def create_manager_from_env() -> Step1XEditManager:
    """
    Create Step1XEditManager from environment variables.

    Environment variables:
        MODEL_PATH: Base model directory
        DIT_PATH: DiT model path
        AE_PATH: AutoEncoder path
        QWEN2VL_PATH: Qwen2VL model path
        GPU_IDLE_TIMEOUT: GPU idle timeout (default: 60)
        DEFAULT_NUM_STEPS: Default inference steps (default: 28)
        DEFAULT_GUIDANCE_SCALE: Default CFG scale (default: 6.0)
        QUANTIZED: Use FP8 quantization (default: false)
        OFFLOAD: Enable CPU offload (default: false)
        LORA_PATH: LoRA weights path (optional)

    Returns:
        Configured Step1XEditManager
    """
    # Model paths
    model_path = os.getenv("MODEL_PATH", "./models")
    dit_path = os.getenv("DIT_PATH", "")
    ae_path = os.getenv("AE_PATH", "")
    qwen2vl_path = os.getenv("QWEN2VL_PATH", "")

    # Use model_path as fallback
    if not dit_path:
        dit_path = os.path.join(model_path, "dit")
    if not ae_path:
        ae_path = os.path.join(model_path, "ae")
    if not qwen2vl_path:
        qwen2vl_path = os.path.join(model_path, "qwen2vl")

    # GPU settings
    gpu_idle_timeout = int(os.getenv("GPU_IDLE_TIMEOUT", "60"))

    # Model settings
    quantized = os.getenv("QUANTIZED", "false").lower() == "true"
    offload = os.getenv("OFFLOAD", "false").lower() == "true"
    lora = os.getenv("LORA_PATH", None)

    # Get device
    device = f"cuda:{os.getenv('NVIDIA_VISIBLE_DEVICES', '0')}"

    logger.info("Creating Step1XEditManager from environment variables")

    return Step1XEditManager(
        dit_path=dit_path if os.path.exists(dit_path) else None,
        ae_path=ae_path if os.path.exists(ae_path) else None,
        qwen2vl_model_path=qwen2vl_path if os.path.exists(qwen2vl_path) else None,
        device=device,
        quantized=quantized,
        offload=offload,
        lora=lora,
        gpu_idle_timeout=gpu_idle_timeout,
        auto_offload=True
    )


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python step1x_manager.py <image_path> <prompt>")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]

    # Create manager from environment
    manager = create_manager_from_env()

    # Edit image
    print(f"Editing image: {image_path}")
    print(f"Prompt: {prompt}")

    result = manager.edit_image(
        image=image_path,
        prompt=prompt,
        num_steps=int(os.getenv("DEFAULT_NUM_STEPS", "28")),
        guidance_scale=float(os.getenv("DEFAULT_GUIDANCE_SCALE", "6.0")),
        size_level=int(os.getenv("DEFAULT_SIZE_LEVEL", "1024"))
    )

    # Save result
    output_path = "output.png"
    result.save(output_path)
    print(f"✓ Result saved to: {output_path}")

    # Show GPU status
    print("\nGPU Status:")
    status = manager.get_gpu_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
