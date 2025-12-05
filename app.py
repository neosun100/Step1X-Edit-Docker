"""
Step1X-Edit Web Application
============================

Modern, multilingual UI for Step1X-Edit with GPU resource management.

Features:
- Multi-language support (English, 简体中文, 繁體中文, 日本語)
- All adjustable parameters with grouping
- Real-time progress display
- GPU status monitoring
- Manual GPU control
- Dark mode support
"""

import os
import time
import gradio as gr
from PIL import Image
import torch

from step1x_manager import create_manager_from_env, Step1XEditManager

# Initialize manager
manager = None

# Translations
TRANSLATIONS = {
    "en": {
        "title": "Step1X-Edit - AI Image Editor",
        "description": "Edit images with natural language instructions using Step1X-Edit model",

        # Tabs
        "tab_single": "Single Image",
        "tab_batch": "Batch Processing",
        "tab_settings": "Settings",
        "tab_gpu": "GPU Status",

        # Input section
        "input_image": "Input Image",
        "input_prompt": "Editing Instruction",
        "prompt_placeholder": "Describe how you want to edit the image...",

        # Parameters
        "params_basic": "Basic Parameters",
        "params_advanced": "Advanced Parameters",
        "num_steps": "Inference Steps",
        "num_steps_info": "Higher = better quality but slower (default: 28)",
        "guidance_scale": "Guidance Scale",
        "guidance_scale_info": "Higher = stronger prompt adherence (default: 6.0)",
        "size_level": "Resolution",
        "size_level_info": "Output image resolution",
        "seed": "Random Seed",
        "seed_info": "Use -1 for random (for reproducible results, use a fixed number)",

        # Buttons
        "btn_edit": "Edit Image",
        "btn_clear": "Clear",
        "btn_offload": "Offload GPU",
        "btn_release": "Release GPU",
        "btn_refresh": "Refresh Status",

        # Output
        "output_image": "Output Image",
        "output_info": "Processing Information",

        # GPU Status
        "gpu_status_title": "GPU Resource Status",
        "model_location": "Model Location",
        "idle_time": "Idle Time",
        "idle_timeout": "Idle Timeout",
        "gpu_memory": "GPU Memory Usage",
        "statistics": "Statistics",

        # Messages
        "msg_processing": "Processing...",
        "msg_success": "✓ Image editing completed",
        "msg_offload": "✓ GPU offloaded to CPU",
        "msg_release": "✓ GPU memory released",
        "msg_error": "❌ Error:",

        # Settings
        "settings_gpu": "GPU Settings",
        "settings_model": "Model Settings",
        "gpu_timeout_label": "GPU Idle Timeout (seconds)",
        "auto_offload_label": "Auto-offload after processing",
        "quantized_label": "Use FP8 quantization (saves ~11GB VRAM)",
        "offload_label": "Enable CPU offload (saves ~17GB VRAM, slower)",
    },
    "zh-CN": {
        "title": "Step1X-Edit - AI图像编辑器",
        "description": "使用Step1X-Edit模型通过自然语言指令编辑图像",

        "tab_single": "单图编辑",
        "tab_batch": "批量处理",
        "tab_settings": "设置",
        "tab_gpu": "GPU状态",

        "input_image": "输入图像",
        "input_prompt": "编辑指令",
        "prompt_placeholder": "描述您想如何编辑图像...",

        "params_basic": "基础参数",
        "params_advanced": "高级参数",
        "num_steps": "推理步数",
        "num_steps_info": "越高质量越好但速度越慢（默认：28）",
        "guidance_scale": "引导强度",
        "guidance_scale_info": "越高越遵循提示词（默认：6.0）",
        "size_level": "分辨率",
        "size_level_info": "输出图像分辨率",
        "seed": "随机种子",
        "seed_info": "使用-1表示随机（若需可复现结果，使用固定数字）",

        "btn_edit": "编辑图像",
        "btn_clear": "清空",
        "btn_offload": "卸载GPU",
        "btn_release": "释放GPU",
        "btn_refresh": "刷新状态",

        "output_image": "输出图像",
        "output_info": "处理信息",

        "gpu_status_title": "GPU资源状态",
        "model_location": "模型位置",
        "idle_time": "空闲时间",
        "idle_timeout": "空闲超时",
        "gpu_memory": "GPU显存使用",
        "statistics": "统计信息",

        "msg_processing": "处理中...",
        "msg_success": "✓ 图像编辑完成",
        "msg_offload": "✓ GPU已卸载到CPU",
        "msg_release": "✓ GPU显存已释放",
        "msg_error": "❌ 错误：",

        "settings_gpu": "GPU设置",
        "settings_model": "模型设置",
        "gpu_timeout_label": "GPU空闲超时（秒）",
        "auto_offload_label": "处理后自动卸载",
        "quantized_label": "使用FP8量化（节省约11GB显存）",
        "offload_label": "启用CPU卸载（节省约17GB显存，但较慢）",
    },
    "zh-TW": {
        "title": "Step1X-Edit - AI圖像編輯器",
        "description": "使用Step1X-Edit模型通過自然語言指令編輯圖像",

        "tab_single": "單圖編輯",
        "tab_batch": "批次處理",
        "tab_settings": "設定",
        "tab_gpu": "GPU狀態",

        "input_image": "輸入圖像",
        "input_prompt": "編輯指令",
        "prompt_placeholder": "描述您想如何編輯圖像...",

        "params_basic": "基礎參數",
        "params_advanced": "進階參數",
        "num_steps": "推理步數",
        "num_steps_info": "越高品質越好但速度越慢（預設：28）",
        "guidance_scale": "引導強度",
        "guidance_scale_info": "越高越遵循提示詞（預設：6.0）",
        "size_level": "解析度",
        "size_level_info": "輸出圖像解析度",
        "seed": "隨機種子",
        "seed_info": "使用-1表示隨機（若需可複現結果，使用固定數字）",

        "btn_edit": "編輯圖像",
        "btn_clear": "清空",
        "btn_offload": "卸載GPU",
        "btn_release": "釋放GPU",
        "btn_refresh": "刷新狀態",

        "output_image": "輸出圖像",
        "output_info": "處理資訊",

        "gpu_status_title": "GPU資源狀態",
        "model_location": "模型位置",
        "idle_time": "閒置時間",
        "idle_timeout": "閒置超時",
        "gpu_memory": "GPU顯存使用",
        "statistics": "統計資訊",

        "msg_processing": "處理中...",
        "msg_success": "✓ 圖像編輯完成",
        "msg_offload": "✓ GPU已卸載到CPU",
        "msg_release": "✓ GPU顯存已釋放",
        "msg_error": "❌ 錯誤：",

        "settings_gpu": "GPU設定",
        "settings_model": "模型設定",
        "gpu_timeout_label": "GPU閒置超時（秒）",
        "auto_offload_label": "處理後自動卸載",
        "quantized_label": "使用FP8量化（節省約11GB顯存）",
        "offload_label": "啟用CPU卸載（節省約17GB顯存，但較慢）",
    },
    "ja": {
        "title": "Step1X-Edit - AI画像エディター",
        "description": "Step1X-Editモデルを使用して自然言語で画像を編集",

        "tab_single": "単一画像",
        "tab_batch": "バッチ処理",
        "tab_settings": "設定",
        "tab_gpu": "GPUステータス",

        "input_image": "入力画像",
        "input_prompt": "編集指示",
        "prompt_placeholder": "画像の編集方法を説明してください...",

        "params_basic": "基本パラメータ",
        "params_advanced": "詳細パラメータ",
        "num_steps": "推論ステップ数",
        "num_steps_info": "高いほど品質が向上しますが遅くなります（デフォルト：28）",
        "guidance_scale": "ガイダンススケール",
        "guidance_scale_info": "高いほどプロンプトに忠実になります（デフォルト：6.0）",
        "size_level": "解像度",
        "size_level_info": "出力画像の解像度",
        "seed": "ランダムシード",
        "seed_info": "-1でランダム（再現可能な結果には固定値を使用）",

        "btn_edit": "画像を編集",
        "btn_clear": "クリア",
        "btn_offload": "GPUをオフロード",
        "btn_release": "GPUを解放",
        "btn_refresh": "ステータスを更新",

        "output_image": "出力画像",
        "output_info": "処理情報",

        "gpu_status_title": "GPUリソースステータス",
        "model_location": "モデルの場所",
        "idle_time": "アイドル時間",
        "idle_timeout": "アイドルタイムアウト",
        "gpu_memory": "GPUメモリ使用量",
        "statistics": "統計情報",

        "msg_processing": "処理中...",
        "msg_success": "✓ 画像編集が完了しました",
        "msg_offload": "✓ GPUがCPUにオフロードされました",
        "msg_release": "✓ GPUメモリが解放されました",
        "msg_error": "❌ エラー：",

        "settings_gpu": "GPU設定",
        "settings_model": "モデル設定",
        "gpu_timeout_label": "GPUアイドルタイムアウト（秒）",
        "auto_offload_label": "処理後に自動オフロード",
        "quantized_label": "FP8量子化を使用（約11GBのVRAMを節約）",
        "offload_label": "CPUオフロードを有効化（約17GBのVRAMを節約、遅い）",
    }
}


def init_manager():
    """Initialize manager from environment."""
    global manager
    if manager is None:
        manager = create_manager_from_env()
    return manager


def t(key, lang="en"):
    """Get translation for key."""
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, key)


def edit_image_ui(image, prompt, num_steps, guidance_scale, size_level, seed, lang):
    """UI function for editing image."""
    try:
        # Initialize manager
        mgr = init_manager()

        # Process seed
        if seed == -1:
            seed = None

        # Edit image
        start_time = time.time()
        result = mgr.edit_image(
            image=image,
            prompt=prompt,
            num_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            size_level=int(size_level),
            seed=seed,
            show_progress=True
        )
        elapsed_time = time.time() - start_time

        # Get GPU status
        status = mgr.get_gpu_status()

        # Format info
        info = f"""{t('msg_success', lang)}

⏱️ Time: {elapsed_time:.2f}s
🖼️ Resolution: {size_level}x{size_level}
📊 Steps: {num_steps}
🎯 CFG Scale: {guidance_scale}
🎲 Seed: {seed if seed is not None else 'Random'}

GPU Status:
- Location: {status['model_location']}
- Memory: {status['gpu_memory_allocated_gb']:.2f}GB allocated
- Idle: {status['idle_time']:.1f}s
"""

        return result, info

    except Exception as e:
        error_msg = f"{t('msg_error', lang)} {str(e)}"
        return None, error_msg


def get_gpu_status_ui(lang):
    """Get GPU status for UI."""
    try:
        mgr = init_manager()
        status = mgr.get_gpu_status()

        info = f"""## {t('gpu_status_title', lang)}

**{t('model_location', lang)}:** {status['model_location']}
**{t('idle_time', lang)}:** {status['idle_time']:.1f}s
**{t('idle_timeout', lang)}:** {status['idle_timeout']}s

### {t('gpu_memory', lang)}
- Allocated: {status['gpu_memory_allocated_gb']:.2f} GB
- Reserved: {status['gpu_memory_reserved_gb']:.2f} GB

### {t('statistics', lang)}
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
        mgr.manual_offload()
        return f"{t('msg_offload', lang)}\n\n{get_gpu_status_ui(lang)}"
    except Exception as e:
        return f"{t('msg_error', lang)} {str(e)}"


def release_gpu_ui(lang):
    """Release GPU manually."""
    try:
        mgr = init_manager()
        mgr.manual_release()
        return f"{t('msg_release', lang)}\n\n{get_gpu_status_ui(lang)}"
    except Exception as e:
        return f"{t('msg_error', lang)} {str(e)}"


def create_ui():
    """Create Gradio UI."""

    # Custom CSS for better styling
    css = """
    .container {max-width: 1200px; margin: auto;}
    .gradio-container {font-family: 'Arial', sans-serif;}
    .gr-button-primary {background: linear-gradient(90deg, #4CAF50, #45a049);}
    .gr-form {gap: 0.5rem;}
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Step1X-Edit") as app:

        # Language selector
        lang_state = gr.State("en")

        with gr.Row():
            gr.Markdown("# 🎨 Step1X-Edit - AI Image Editor")
            language = gr.Dropdown(
                choices=[
                    ("English", "en"),
                    ("简体中文", "zh-CN"),
                    ("繁體中文", "zh-TW"),
                    ("日本語", "ja")
                ],
                value="en",
                label="Language / 语言",
                scale=1
            )

        gr.Markdown("Edit images with natural language instructions")

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
                                label="Inference Steps",
                                info="Higher = better quality but slower"
                            )

                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=15.0,
                                value=6.0,
                                step=0.5,
                                label="Guidance Scale",
                                info="Higher = stronger prompt adherence"
                            )

                            size_level = gr.Radio(
                                choices=[512, 768, 1024],
                                value=1024,
                                label="Resolution",
                                info="Output image resolution"
                            )

                        with gr.Accordion("Advanced Parameters", open=False):
                            seed = gr.Number(
                                value=-1,
                                label="Random Seed",
                                info="Use -1 for random"
                            )

                        with gr.Row():
                            edit_btn = gr.Button("🎨 Edit Image", variant="primary", scale=2)
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
    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")

    # Create and launch app
    app = create_ui()
    app.launch(
        server_name=host,
        server_port=port,
        share=False
    )
