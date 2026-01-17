#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Credit Risk Frontend Server${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if we're in the correct directory
if [ ! -d "frontend" ]; then
    echo -e "${RED}[✗] ERROR: frontend/ directory not found!${NC}"
    echo -e "${YELLOW}[*] Please run this script from the project root directory${NC}"
    echo -e "${YELLOW}[*] Current location: $(pwd)${NC}"
    exit 1
fi

# Check if index.html exists
if [ ! -f "frontend/index.html" ]; then
    echo -e "${RED}[✗] ERROR: frontend/index.html not found!${NC}"
    exit 1
fi

echo -e "${GREEN}[✓] frontend/index.html found${NC}\n"

# Find available port starting from 8000
PORT=8000
MAX_PORT=8010

while netstat -tuln 2>/dev/null | grep -q ":$PORT " || nc -z 127.0.0.1 $PORT 2>/dev/null; do
    echo -e "${YELLOW}[*] Port $PORT is in use, trying next port...${NC}"
    PORT=$((PORT + 1))
    if [ $PORT -gt $MAX_PORT ]; then
        echo -e "${RED}[✗] No available ports found between 8000-8010${NC}"
        exit 1
    fi
done

echo -e "${GREEN}[✓] Port $PORT is available${NC}\n"

# Display connection info
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}[✓] Starting Frontend Server${NC}"
echo -e "${BLUE}========================================${NC}\n"
echo -e "${YELLOW}🌐 Frontend URL: http://localhost:$PORT${NC}"
echo -e "${YELLOW}⚙️  Backend URL:  http://localhost:8001${NC}"
echo -e "${YELLOW}📝 Status: Waiting for connections...${NC}"
echo -e "${YELLOW}🛑 Press Ctrl+C to stop the server\n${NC}"

# Start the server from frontend directory
cd frontend
python -m http.server $PORT --bind 127.0.0.1