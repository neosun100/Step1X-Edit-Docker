#!/bin/bash

# =============================================================================
# Step1X-Edit API Test Script
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
TEST_IMAGE="${1:-examples/0000.jpg}"

echo "========================================"
echo "Step1X-Edit API Test Script"
echo "========================================"
echo "API URL: $API_URL"
echo "Test Image: $TEST_IMAGE"
echo ""

# Check if test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${RED}Error: Test image not found: $TEST_IMAGE${NC}"
    echo "Usage: $0 [path_to_test_image]"
    exit 1
fi

# Function to print test result
print_test() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ $2${NC}"
    else
        echo -e "${RED}✗ $2${NC}"
    fi
}

echo "Running API tests..."
echo ""

# =============================================================================
# Test 1: Health Check
# =============================================================================
echo "Test 1: Health Check"
response=$(curl -s "$API_URL/health")
if echo "$response" | grep -q '"status":"healthy"'; then
    print_test 0 "Health check passed"
    echo "  Response: $response"
else
    print_test 1 "Health check failed"
    echo "  Response: $response"
fi
echo ""

# =============================================================================
# Test 2: GPU Status
# =============================================================================
echo "Test 2: GPU Status"
response=$(curl -s "$API_URL/gpu/status")
if echo "$response" | grep -q 'model_location'; then
    print_test 0 "GPU status endpoint working"
    echo "  Response: $response" | python3 -m json.tool 2>/dev/null || echo "  $response"
else
    print_test 1 "GPU status endpoint failed"
fi
echo ""

# =============================================================================
# Test 3: Image Editing (Base64 Response)
# =============================================================================
echo "Test 3: Image Editing (Base64 Response)"
response=$(curl -s -X POST "$API_URL/api/edit" \
    -F "file=@$TEST_IMAGE" \
    -F "prompt=add a red hat" \
    -F "num_steps=10" \
    -F "guidance_scale=6.0" \
    -F "size_level=512")

if echo "$response" | grep -q '"success":true'; then
    print_test 0 "Image editing successful"
    # Extract processing time
    processing_time=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['processing_time'])" 2>/dev/null || echo "N/A")
    echo "  Processing time: ${processing_time}s"

    # Save image if base64 is present
    if echo "$response" | grep -q 'image_base64'; then
        echo "$response" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
with open('test_output.png', 'wb') as f:
    f.write(base64.b64decode(data['image_base64']))
print('  Output saved to: test_output.png')
" 2>/dev/null || echo "  Could not save output image"
    fi
else
    print_test 1 "Image editing failed"
    echo "  Response: $response"
fi
echo ""

# =============================================================================
# Test 4: Image Editing (File Response)
# =============================================================================
echo "Test 4: Image Editing (File Response)"
curl -s -X POST "$API_URL/api/edit-file" \
    -F "file=@$TEST_IMAGE" \
    -F "prompt=make it brighter" \
    -F "num_steps=10" \
    -F "size_level=512" \
    -o test_output_file.png

if [ -f "test_output_file.png" ] && [ -s "test_output_file.png" ]; then
    print_test 0 "File response working"
    echo "  Output saved to: test_output_file.png"
else
    print_test 1 "File response failed"
fi
echo ""

# =============================================================================
# Test 5: GPU Offload
# =============================================================================
echo "Test 5: GPU Manual Offload"
response=$(curl -s -X POST "$API_URL/gpu/offload")
if echo "$response" | grep -q '"success":true'; then
    print_test 0 "GPU offload successful"
    echo "  Response: $response"
else
    print_test 1 "GPU offload failed"
fi
echo ""

# =============================================================================
# Test 6: Swagger Documentation
# =============================================================================
echo "Test 6: Swagger Documentation"
response=$(curl -s "$API_URL/docs")
if echo "$response" | grep -q "swagger"; then
    print_test 0 "Swagger docs accessible"
    echo "  URL: $API_URL/docs"
else
    print_test 1 "Swagger docs not accessible"
fi
echo ""

# =============================================================================
# Summary
# =============================================================================
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "All basic tests completed!"
echo ""
echo "Generated files:"
if [ -f "test_output.png" ]; then
    echo "  - test_output.png (base64 response)"
fi
if [ -f "test_output_file.png" ]; then
    echo "  - test_output_file.png (file response)"
fi
echo ""
echo "API Endpoints:"
echo "  - Health:     $API_URL/health"
echo "  - GPU Status: $API_URL/gpu/status"
echo "  - Edit:       $API_URL/api/edit"
echo "  - Edit File:  $API_URL/api/edit-file"
echo "  - GPU Offload: $API_URL/gpu/offload"
echo "  - GPU Release: $API_URL/gpu/release"
echo "  - Swagger:    $API_URL/docs"
echo ""
