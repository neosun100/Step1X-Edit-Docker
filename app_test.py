"""
Step1X-Edit Web Application (Test Mode)
=========================================

Test version with mock models for UI validation.
"""

import os
import time
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import torch

from gpu_manager import GPUResourceManager

# Initialize GPU manager
gpu_manager = None

# Translations (same as app.py)
TRANSLATIONS = {
    "en": {
        "title": "Step1X-Edit - AI Image Editor (Test Mode)",
        "description": "Test version without actual models - adds text overlay to show processing",
        "tab_single": "Single Image",
        "tab_gpu": "GPU Status",
        "input_image": "Input Image",
        "input_prompt": "Editing Instruction",
        "prompt_placeholder": "Describe how you want to edit the image...",
        "params_basic": "Basic Parameters",
        "params_advanced": "Advanced Parameters",
        "num_steps": "Inference Steps",
        "guidance_scale": "Guidance Scale",
        "size_level": "Resolution",
        "seed": "Random Seed",
        "btn_edit": "Edit Image",
        "btn_clear": "Clear",
        "btn_offload": "Offload GPU",
        "btn_release": "Release GPU",
        "btn_refresh": "Refresh Status",
        "output_image": "Output Image",
        "output_info": "Processing Information",
        "msg_processing": "Processing...",
        "msg_success": "✓ Image editing completed (test mode)",
        "msg_offload": "✓ GPU offloaded to CPU",
        "msg_release": "✓ GPU memory released",
        "msg_error": "❌ Error:",
    },
    "zh-CN": {
        "title": "Step1X-Edit - AI图像编辑器（测试模式）",
        "description": "测试版本无实际模型 - 添加文本覆盖层显示处理",
        "tab_single": "单图编辑",
        "tab_gpu": "GPU状态",
        "input_image": "输入图像",
        "input_prompt": "编辑指令",
        "prompt_placeholder": "描述您想如何编辑图像...",
        "params_basic": "基础参数",
        "params_advanced": "高级参数",
        "num_steps": "推理步数",
        "guidance_scale": "引导强度",
        "size_level": "分辨率",
        "seed": "随机种子",
        "btn_edit": "编辑图像",
        "btn_clear": "清空",
        "btn_offload": "卸载GPU",
        "btn_release": "释放GPU",
        "btn_refresh": "刷新状态",
        "output_image": "输出图像",
        "output_info": "处理信息",
        "msg_processing": "处理中...",
        "msg_success": "✓ 图像编辑完成（测试模式）",
        "msg_offload": "✓ GPU已卸载到CPU",
        "msg_release": "✓ GPU显存已释放",
        "msg_error": "❌ 错误：",
    },
}


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
            f"TEST MODE",
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
    print("Loading mock model...")
    time.sleep(0.5)  # Simulate loading
    return MockImageProcessor()


def init_manager():
    """Initialize GPU manager."""
    global gpu_manager
    if gpu_manager is None:
        gpu_manager = GPUResourceManager(
            idle_timeout=60,
            device="cuda:1"  # Use GPU 1
        )
        print("✓ GPU Manager initialized")
    return gpu_manager


def t(key, lang="en"):
    """Get translation for key."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def edit_image_ui(image, prompt, num_steps, guidance_scale, size_level, seed, lang):
    """UI function for editing image."""
    try:
        if image is None:
            return None, f"{t('msg_error', lang)} No image provided"

        if not prompt:
            return None, f"{t('msg_error', lang)} No prompt provided"

        # Initialize manager
        mgr = init_manager()

        # Get model with GPU manager
        start_time = time.time()
        model = mgr.get_model(load_func=load_mock_model)

        # Process (mock)
        result_image = model.process(image, prompt, size_level)

        # Offload GPU
        mgr.force_offload()

        processing_time = time.time() - start_time

        # Get GPU status
        status = mgr.get_status()

        # Format info
        info = f"""{t('msg_success', lang)}

⏱️ Time: {processing_time:.2f}s
🖼️ Resolution: {size_level}x{size_level}
📊 Steps: {num_steps} (mock)
🎯 CFG Scale: {guidance_scale} (mock)
🎲 Seed: {seed if seed != -1 else 'Random'}

GPU Status:
- Location: {status['model_location']}
- Memory: {status['gpu_memory_allocated_gb']:.2f}GB allocated
- Idle: {status['idle_time']:.1f}s
"""

        return result_image, info

    except Exception as e:
        error_msg = f"{t('msg_error', lang)} {str(e)}"
        import traceback
        traceback.print_exc()
        return None, error_msg


def get_gpu_status_ui(lang):
    """Get GPU status for UI."""
    try:
        mgr = init_manager()
        status = mgr.get_status()

        info = f"""## GPU Resource Status

**Model Location:** {status['model_location']}
**Idle Time:** {status['idle_time']:.1f}s
**Idle Timeout:** {status['idle_timeout']}s

### GPU Memory
- Allocated: {status['gpu_memory_allocated_gb']:.2f} GB
- Reserved: {status['gpu_memory_reserved_gb']:.2f} GB

### Statistics
- Total Loads: {status['statistics']['total_loads']}
- GPU→CPU: {status['statistics']['gpu_to_cpu']}
- CPU→GPU: {status['statistics']['cpu_to_gpu']}
- Full Releases: {status['statistics']['full_releases']}
"""
        return info

    except Exception as e:
        return f"{t('msg_error', lang)} {str(e)}"


def offload_gpu_ui(lang):
    """Offload GPU manually."""
    try:
        mgr = init_manager()
        mgr.force_offload()
        return f"{t('msg_offload', lang)}\n\n{get_gpu_status_ui(lang)}"
    except Exception as e:
        return f"{t('msg_error', lang)} {str(e)}"


def release_gpu_ui(lang):
    """Release GPU manually."""
    try:
        mgr = init_manager()
        mgr.force_release()
        return f"{t('msg_release', lang)}\n\n{get_gpu_status_ui(lang)}"
    except Exception as e:
        return f"{t('msg_error', lang)} {str(e)}"


def create_ui():
    """Create Gradio UI."""

    with gr.Blocks(title="Step1X-Edit Test") as app:

        # Language selector
        lang_state = gr.State("en")

        with gr.Row():
            gr.Markdown("# 🎨 Step1X-Edit - Test Mode")
            language = gr.Dropdown(
                choices=[
                    ("English", "en"),
                    ("简体中文", "zh-CN"),
                ],
                value="en",
                label="Language / 语言",
                scale=1
            )

        gr.Markdown("**Test version** - Uses mock models for UI validation. Adds text overlay to show processing.")

        with gr.Tabs() as tabs:

            # Tab 1: Single Image Editing
            with gr.Tab("Single Image") as tab_single:
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Input Image",
                            type="pil",
                            height=400
                        )

                        prompt = gr.Textbox(
                            label="Editing Instruction",
                            placeholder="Describe how you want to edit the image...",
                            lines=3
                        )

                        with gr.Accordion("Basic Parameters", open=True):
                            num_steps = gr.Slider(
                                minimum=10,
                                maximum=50,
                                value=28,
                                step=1,
                                label="Inference Steps (mock)",
                                info="Not used in test mode"
                            )

                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=15.0,
                                value=6.0,
                                step=0.5,
                                label="Guidance Scale (mock)",
                                info="Not used in test mode"
                            )

                            size_level = gr.Radio(
                                choices=[512, 768, 1024],
                                value=1024,
                                label="Resolution",
                                info="Output image will be resized to this"
                            )

                        with gr.Accordion("Advanced Parameters", open=False):
                            seed = gr.Number(
                                value=-1,
                                label="Random Seed (mock)",
                                info="Not used in test mode"
                            )

                        with gr.Row():
                            edit_btn = gr.Button("🎨 Edit Image (Test)", variant="primary", scale=2)
                            clear_btn = gr.ClearButton(scale=1)

                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="Output Image",
                            type="pil",
                            height=400
                        )

                        output_info = gr.Markdown("Processing information will appear here")

            # Tab 2: GPU Status
            with gr.Tab("GPU Status") as tab_gpu:
                gpu_status_display = gr.Markdown("Click 'Refresh Status' to view GPU status")

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    offload_btn = gr.Button("💾 Offload GPU", variant="secondary")
                    release_btn = gr.Button("🗑️ Release GPU", variant="stop")

        # Event handlers
        edit_btn.click(
            fn=edit_image_ui,
            inputs=[input_image, prompt, num_steps, guidance_scale, size_level, seed, lang_state],
            outputs=[output_image, output_info]
        )

        refresh_btn.click(
            fn=get_gpu_status_ui,
            inputs=[lang_state],
            outputs=[gpu_status_display]
        )

        offload_btn.click(
            fn=offload_gpu_ui,
            inputs=[lang_state],
            outputs=[gpu_status_display]
        )

        release_btn.click(
            fn=release_gpu_ui,
            inputs=[lang_state],
            outputs=[gpu_status_display]
        )

        language.change(
            fn=lambda x: x,
            inputs=[language],
            outputs=[lang_state]
        )

        # Clear button targets
        clear_btn.add([input_image, prompt, output_image, output_info])

    return app


if __name__ == "__main__":
    # Read port from environment
    port = int(os.getenv("UI_PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")

    # Create and launch app
    print("="*60)
    print("Step1X-Edit Test UI")
    print("="*60)
    print(f"Starting UI on {host}:{port}")
    print("This is a TEST version using mock models")
    print("="*60)

    app = create_ui()
    app.launch(
        server_name=host,
        server_port=port,
        share=False
    )
