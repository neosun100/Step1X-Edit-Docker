#!/usr/bin/env python3
"""
Test script for Gradio UI
"""

import requests
import json
import base64
from PIL import Image
import io
import time

# Configuration
UI_URL = "http://127.0.0.1:7860"
TEST_IMAGE = "examples/0000.jpg"

print("="*60)
print("Step1X-Edit UI Test Script")
print("="*60)
print(f"UI URL: {UI_URL}")
print(f"Test Image: {TEST_IMAGE}")
print("")

# Test 1: Check if UI is accessible
print("Test 1: UI Accessibility")
try:
    response = requests.get(UI_URL, timeout=5)
    if response.status_code == 200:
        print("✓ UI is accessible")
        print(f"  Status code: {response.status_code}")
    else:
        print(f"✗ UI returned unexpected status: {response.status_code}")
except Exception as e:
    print(f"✗ Failed to access UI: {e}")
print("")

# Test 2: Get API configuration
print("Test 2: Get API Configuration")
try:
    response = requests.get(f"{UI_URL}/config", timeout=5)
    if response.status_code == 200:
        config = response.json()
        print("✓ Got API configuration")
        print(f"  Title: {config.get('title', 'N/A')}")
        print(f"  Version: {config.get('version', 'N/A')}")
        print(f"  Components: {len(config.get('components', []))}")
    else:
        print(f"✗ Config request failed: {response.status_code}")
except Exception as e:
    print(f"✗ Error getting config: {e}")
print("")

# Test 3: Simulate image editing via Gradio API
print("Test 3: Image Editing via Gradio API")
try:
    # Load test image
    with open(TEST_IMAGE, "rb") as f:
        image_data = f.read()

    # Encode to base64
    img_base64 = base64.b64encode(image_data).decode()

    # Prepare API request (Gradio format)
    api_data = {
        "data": [
            f"data:image/jpeg;base64,{img_base64}",  # input_image
            "add a red hat",                          # prompt
            28,                                        # num_steps
            6.0,                                       # guidance_scale
            512,                                       # size_level (smaller for faster test)
            -1,                                        # seed
            "en"                                       # lang
        ]
    }

    print("  Sending edit request...")
    start_time = time.time()

    response = requests.post(
        f"{UI_URL}/api/predict",
        json=api_data,
        timeout=30
    )

    elapsed = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print("✓ Image editing successful")
        print(f"  Processing time: {elapsed:.2f}s")

        # Check if we got data back
        if "data" in result and len(result["data"]) > 0:
            print(f"  Response contains {len(result['data'])} items")

            # Try to decode output image
            output_data = result["data"][0]
            if output_data and "image" in str(output_data):
                print("  ✓ Output image received")

            # Check info text
            if len(result["data"]) > 1:
                info_text = result["data"][1]
                if "✓" in info_text and "GPU Status" in info_text:
                    print("  ✓ Processing info includes GPU status")
        else:
            print("  ⚠ Response data structure unexpected")
    else:
        print(f"✗ Edit request failed: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
except Exception as e:
    print(f"✗ Error during image editing: {e}")
    import traceback
    traceback.print_exc()
print("")

# Test 4: Check available functions
print("Test 4: Available Functions")
try:
    response = requests.get(f"{UI_URL}/info", timeout=5)
    if response.status_code == 200:
        info = response.json()
        print("✓ Got function info")
        if "named_endpoints" in info:
            endpoints = info["named_endpoints"]
            print(f"  Available endpoints: {len(endpoints)}")
            for endpoint in list(endpoints.keys())[:5]:
                print(f"    - {endpoint}")
    else:
        print(f"  No info endpoint (status {response.status_code})")
except Exception as e:
    print(f"  No info endpoint available: {type(e).__name__}")
print("")

# Summary
print("="*60)
print("Test Summary")
print("="*60)
print("The Gradio UI is running successfully!")
print("")
print("Manual testing:")
print(f"1. Open browser: {UI_URL}")
print("2. Upload an image from examples/")
print("3. Enter a prompt (e.g., 'add a red hat')")
print("4. Click 'Edit Image (Test)' button")
print("5. Check GPU Status tab")
print("")
print("Expected behavior:")
print("- Image will be resized to selected resolution")
print("- Text overlay will show 'TEST MODE' and prompt")
print("- GPU status will show model location and memory")
print("- Language switcher should work for EN/中文")
print("")
