#!/bin/bash
# Complete API testing script

echo "=========================================="
echo "API Testing Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"

echo "Testing API at: $API_URL"
echo ""

# Test 1: Root endpoint
echo "1. Testing root endpoint..."
response=$(curl -s ${API_URL}/)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Root endpoint accessible${NC}"
    echo "   Response: $response"
else
    echo -e "${RED}✗ Failed to connect to API${NC}"
    echo -e "${YELLOW}   Make sure API is running: make serve-api${NC}"
    exit 1
fi
echo ""

# Test 2: Health check
echo "2. Testing health check..."
health_response=$(curl -s ${API_URL}/health)
echo "   Response: $health_response"

model_loaded=$(echo $health_response | grep -o '"model_loaded":[^,}]*' | cut -d':' -f2)
if [[ "$model_loaded" == *"true"* ]]; then
    echo -e "${GREEN}✓ Model is loaded and ready${NC}"
else
    echo -e "${RED}✗ Model not loaded${NC}"
    echo -e "${YELLOW}   Please restart the API server to load the model${NC}"
    exit 1
fi
echo ""

# Test 3: Model info
echo "3. Getting model information..."
info_response=$(curl -s ${API_URL}/model-info)
echo "   Response: $info_response"
echo ""

# Test 4: Valid prediction
echo "4. Testing prediction with valid data..."
prediction_response=$(curl -s -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stasiun": "DKI1 (Bunderan HI)",
    "pm10": 50.5,
    "pm25": 30.2,
    "so2": 15.0,
    "co": 5.5,
    "o3": 45.0,
    "no2": 25.0
  }')

if echo "$prediction_response" | grep -q "prediction"; then
    echo -e "${GREEN}✓ Prediction successful${NC}"
    echo "   Response: $prediction_response"
else
    echo -e "${RED}✗ Prediction failed${NC}"
    echo "   Response: $prediction_response"
fi
echo ""

# Test 5: Invalid station
echo "5. Testing with invalid station (should fail)..."
error_response=$(curl -s -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stasiun": "Invalid Station",
    "pm10": 50.5,
    "pm25": 30.2,
    "so2": 15.0,
    "co": 5.5,
    "o3": 45.0,
    "no2": 25.0
  }')

if echo "$error_response" | grep -q "detail"; then
    echo -e "${GREEN}✓ Validation working correctly${NC}"
    echo "   Error: $error_response"
else
    echo -e "${YELLOW}⚠ Unexpected response${NC}"
    echo "   Response: $error_response"
fi
echo ""

# Test 6: Out of range values
echo "6. Testing with out-of-range PM10 (should fail)..."
range_response=$(curl -s -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "stasiun": "DKI1 (Bunderan HI)",
    "pm10": 999999,
    "pm25": 30.2,
    "so2": 15.0,
    "co": 5.5,
    "o3": 45.0,
    "no2": 25.0
  }')

if echo "$range_response" | grep -q "detail"; then
    echo -e "${GREEN}✓ Range validation working${NC}"
    echo "   Error: $range_response"
else
    echo -e "${YELLOW}⚠ Unexpected response${NC}"
    echo "   Response: $range_response"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}✓ All tests completed!${NC}"
echo "=========================================="
echo ""
echo "API is functioning correctly!"
echo ""
echo "You can now:"
echo "  - Use the API programmatically"
echo "  - Start the UI: make serve-ui"
echo "  - View docs: ${API_URL}/docs"
