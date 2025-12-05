#!/bin/bash

set -e

echo "========================================="
echo "  Step1X-Edit Deployment Test Suite"
echo "========================================="

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
BASE_URL="http://${HOST}:${PORT}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Test function
test_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local expected_code="$4"
    
    echo -n "Testing $name... "
    
    response_code=$(curl -s -o /dev/null -w "%{http_code}" -X "$method" "${BASE_URL}${endpoint}")
    
    if [ "$response_code" -eq "$expected_code" ]; then
        echo -e "${GREEN}✓ PASS${NC} (HTTP $response_code)"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC} (Expected $expected_code, got $response_code)"
        ((FAILED++))
        return 1
    fi
}

echo ""
echo "=== 1. Health Checks ==="

# Health endpoint
test_endpoint "Health Check" "GET" "/health" 200

# GPU status
test_endpoint "GPU Status" "GET" "/api/gpu/status" 200

echo ""
echo "=== 2. UI Access ==="

# UI homepage
test_endpoint "UI Homepage" "GET" "/" 200

echo ""
echo "=== 3. API Documentation ==="

# Swagger docs
test_endpoint "Swagger UI" "GET" "/docs" 200

# ReDoc
test_endpoint "ReDoc" "GET" "/redoc" 200

echo ""
echo "=== 4. GPU Management ==="

# Offload GPU
test_endpoint "GPU Offload" "POST" "/api/gpu/offload" 200

# Release GPU
test_endpoint "GPU Release" "POST" "/api/gpu/release" 200

echo ""
echo "=== 5. Detailed Status Check ==="

# Get detailed GPU status
echo -n "Fetching GPU status... "
gpu_status=$(curl -s "${BASE_URL}/api/gpu/status")

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    echo "$gpu_status" | jq '.' 2>/dev/null || echo "$gpu_status"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC}"
    ((FAILED++))
fi

echo ""
echo "=== 6. Container Status ==="

# Check if container is running
echo -n "Checking container... "
if docker ps | grep -q step1x-edit; then
    echo -e "${GREEN}✓ PASS${NC} (Container running)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (Container not found)"
    ((FAILED++))
fi

# Check GPU access in container
echo -n "Checking GPU access... "
gpu_check=$(docker exec step1x-edit nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASS${NC} (GPU: $gpu_check)"
    ((PASSED++))
else
    echo -e "${RED}✗ FAIL${NC} (No GPU access)"
    ((FAILED++))
fi

echo ""
echo "=== 7. Image Edit Test (Optional) ==="

# Check if test image exists
if [ -f "examples/0000.jpg" ]; then
    echo -n "Testing image edit... "
    
    response=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/api/edit" \
        -F "file=@examples/0000.jpg" \
        -F "prompt=add a red hat" \
        -F "num_steps=20" \
        -F "guidance_scale=6.0" \
        -F "size_level=512" \
        -o /tmp/test_result.png)
    
    http_code=$(echo "$response" | tail -n1)
    
    if [ "$http_code" -eq 200 ] && [ -f "/tmp/test_result.png" ]; then
        file_size=$(stat -f%z "/tmp/test_result.png" 2>/dev/null || stat -c%s "/tmp/test_result.png" 2>/dev/null)
        if [ "$file_size" -gt 1000 ]; then
            echo -e "${GREEN}✓ PASS${NC} (Generated ${file_size} bytes)"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAIL${NC} (File too small: ${file_size} bytes)"
            ((FAILED++))
        fi
    else
        echo -e "${YELLOW}⊘ SKIP${NC} (HTTP $http_code or file not created)"
    fi
else
    echo -e "${YELLOW}⊘ SKIP${NC} (No test image found at examples/0000.jpg)"
fi

echo ""
echo "========================================="
echo "  Test Results"
echo "========================================="
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo "Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "Access your deployment:"
    echo "  • UI:      ${BASE_URL}"
    echo "  • API:     ${BASE_URL}/docs"
    echo "  • Health:  ${BASE_URL}/health"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  • Check logs: docker-compose logs -f"
    echo "  • Check GPU: nvidia-smi"
    echo "  • Check container: docker ps -a"
    exit 1
fi
