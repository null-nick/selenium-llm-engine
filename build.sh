#!/usr/bin/env bash
set -euo pipefail

# build.sh - Build and optionally run the Selenium LLM Engine docker stack
# Usage:
#   ./build.sh            # build only
#   ./build.sh up         # build and start containers
#   ./build.sh down       # stop and remove containers (optional)

COMPOSE_FILE="docker-compose.yml"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: $COMPOSE_FILE not found in $(pwd)"
  exit 1
fi

case "${1:-}" in
  "" )
    echo "Building Docker images..."
    docker compose -f "$COMPOSE_FILE" build --pull --no-cache
    echo "Build complete. Run './build.sh up' to start the stack." ;;
  up )
    echo "Building Docker images..."
    docker compose -f "$COMPOSE_FILE" build --pull --no-cache
    echo "Starting the stack..."
    docker compose -f "$COMPOSE_FILE" up -d
    echo "Stack started. Access API at http://localhost:8000" ;;
  down )
    echo "Stopping and removing stack..."
    docker compose -f "$COMPOSE_FILE" down
    echo "Stack stopped." ;;
  * )
    echo "Usage: $0 [up|down]" ;;
esac
