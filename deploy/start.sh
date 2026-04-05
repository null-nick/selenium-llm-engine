#!/usr/bin/env bash
set -euo pipefail

# start.sh — Bootstrap the production stack with self-signed SSL
#
# Usage:
#   ./deploy/start.sh              # default (localhost)
#   ./deploy/start.sh 172.0.0.6  # custom hostname/IP for the cert CN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${1:-localhost}"

# ---- local volume dirs ----
mkdir -p data config certs

# ---- self-signed certificate ----
if [[ ! -f certs/selkies.crt || ! -f certs/selkies.key ]]; then
  echo "Generating self-signed certificate for CN=${HOST}..."
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout certs/selkies.key -out certs/selkies.crt \
    -subj "/CN=${HOST}" 2>/dev/null
  echo "Certificate created in deploy/certs/"
else
  echo "Certificates already exist, skipping generation."
fi

# ---- start stack ----
docker compose down 2>/dev/null || true
docker compose up -d

echo ""
echo "Waiting for services to start..."
sleep 8
docker compose ps

echo ""
echo "✅ Stack running!"
echo ""
echo "📍 Access:"
echo "   🔒 Web UI (HTTPS):          https://${HOST}"
echo "   🔓 API (HTTP, direct):      http://${HOST}:14848"
echo "   📡 API ping:                http://${HOST}:14848/api/ping"
echo ""
echo "📁 Local data:"
echo "   Data:    ${SCRIPT_DIR}/data"
echo "   Config:  ${SCRIPT_DIR}/config"
echo "   Certs:   ${SCRIPT_DIR}/certs"
echo ""
echo "📋 Commands:"
echo "   docker compose -f ${SCRIPT_DIR}/docker-compose.yml ps"
echo "   docker compose -f ${SCRIPT_DIR}/docker-compose.yml logs -f"
echo "   docker compose -f ${SCRIPT_DIR}/docker-compose.yml down"
