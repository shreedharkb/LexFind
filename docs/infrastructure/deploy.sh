#!/bin/bash
# ─────────────────────────────────────────────
# LexFind – Production VM Bootstrap Script
#
# Run this once after SSH-ing into a fresh Ubuntu 22.04 VM:
#   chmod +x deploy.sh
#   sudo ./deploy.sh
#
# What it does:
#   1. Installs Docker + Docker Compose
#   2. Installs Nginx
#   3. Clones/updates the repo
#   4. Copies Nginx config and reloads
#   5. Starts the API container via Docker Compose
# ─────────────────────────────────────────────

set -euo pipefail

REPO_URL="https://github.com/shreedharkb/LexFind.git"
APP_DIR="/opt/LexFind"
NGINX_CONF="/etc/nginx/sites-available/LexFind"

echo "════════════════════════════════════════"
echo "  LexFind VM Setup"
echo "════════════════════════════════════════"

# ── 1. System update ─────────────────────────
echo "[1/6] Updating system packages..."
apt-get update -q && apt-get upgrade -y -q

# ── 2. Install Docker ────────────────────────
echo "[2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    # Allow current user to run docker without sudo
    usermod -aG docker "$SUDO_USER" || true
    echo "Docker installed. You may need to log out and back in for group changes."
else
    echo "Docker already installed: $(docker --version)"
fi

# Docker Compose (v2 plugin)
if ! docker compose version &> /dev/null; then
    apt-get install -y docker-compose-plugin
fi
echo "Docker Compose: $(docker compose version)"

# ── 3. Install Nginx ─────────────────────────
echo "[3/6] Installing Nginx..."
apt-get install -y nginx
systemctl enable nginx
systemctl start nginx

# ── 4. Clone / update repo ───────────────────
echo "[4/6] Setting up application directory..."
if [ -d "$APP_DIR/.git" ]; then
    echo "Repo exists — pulling latest..."
    git -C "$APP_DIR" pull
else
    echo "Cloning repo to $APP_DIR..."
    git clone "$REPO_URL" "$APP_DIR"
fi

# ── 5. Configure Nginx ───────────────────────
echo "[5/6] Configuring Nginx..."
cp "$APP_DIR/docs/infrastructure/nginx.conf" "$NGINX_CONF"
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/LexFind
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
echo "Nginx configured and reloaded."

# ── 6. Start API container ───────────────────
echo "[6/6] Starting LexFind API container..."

# Make sure .env exists
if [ ! -f "$APP_DIR/backend/.env" ]; then
    echo ""
    echo "⚠️  WARNING: $APP_DIR/backend/.env not found!"
    echo "    Copy backend/.env.example → backend/.env and fill in:"
    echo "      GROQ_API_KEY"
    echo "      AZURE_STORAGE_CONNECTION_STRING  (if using Azure Blob)"
    echo "    Then run:  cd $APP_DIR && docker compose up -d --build"
    echo ""
else
    cd "$APP_DIR"
    docker compose pull 2>/dev/null || true
    docker compose up -d --build
    echo "Container started. Check logs with:  docker compose logs -f api"
fi

echo ""
echo "════════════════════════════════════════"
echo "  Setup complete!"
echo "  API available at: http://$(curl -s ifconfig.me)/api/health"
echo "  API docs at:      http://$(curl -s ifconfig.me)/docs"
echo ""
echo "  Next steps:"
echo "  1. Fill in $APP_DIR/api/.env with real keys"
echo "  2. Open ports 80 and 443 in your Network Security Group/Firewall"
echo "  3. (Optional) Add SSL: sudo certbot --nginx -d yourdomain.com"
echo "════════════════════════════════════════"
