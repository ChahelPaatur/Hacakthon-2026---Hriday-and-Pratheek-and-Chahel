#!/usr/bin/env bash
set -euo pipefail

# NeuroLang Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/ChahelPaatur/Hacakthon-2026---Hriday-and-Pratheek-and-Chahel/main/Nueral%20Network%20Langauge/scripts/install.sh | bash
#
# Environment variables:
#   NEUROLANG_VERSION   - version to install (default: latest)
#   NEUROLANG_INSTALL   - install directory (default: /usr/local/bin)

VERSION="${NEUROLANG_VERSION:-latest}"
INSTALL_DIR="${NEUROLANG_INSTALL:-/usr/local/bin}"

Y='\033[33m'
G='\033[32m'
R='\033[31m'
B='\033[1m'
D='\033[2m'
X='\033[0m'

echo ""
echo -e "${B}${Y}╔═══════════════════════════════════════════╗${X}"
echo -e "${B}${Y}║       NeuroLang Installer                 ║${X}"
echo -e "${B}${Y}╚═══════════════════════════════════════════╝${X}"
echo ""

# Check for curl
if ! command -v curl &>/dev/null; then
  echo -e "  ${R}Error: curl is required but not installed.${X}"
  exit 1
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)   PLATFORM="linux" ;;
  Darwin)  PLATFORM="macos" ;;
  MINGW*|MSYS*|CYGWIN*)
    echo -e "  ${R}Error: Use the PowerShell installer on Windows:${X}"
    echo -e "  ${Y}irm https://raw.githubusercontent.com/ChahelPaatur/Hacakthon-2026---Hriday-and-Pratheek-and-Chahel/main/Nueral%20Network%20Langauge/scripts/install.ps1 | iex${X}"
    exit 1
    ;;
  *)
    echo -e "  ${R}Unsupported OS: $OS${X}"
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64)  MACHINE="x64" ;;
  arm64|aarch64) MACHINE="arm64" ;;
  *)
    echo -e "  ${R}Unsupported architecture: $ARCH${X}"
    exit 1
    ;;
esac

BINARY="neurolang-${PLATFORM}-${MACHINE}"
echo -e "  Platform:  ${G}${PLATFORM} ${MACHINE}${X}"
echo -e "  Version:   ${G}${VERSION}${X}"
echo -e "  Install:   ${G}${INSTALL_DIR}/neurolang${X}"
echo ""

REPO="ChahelPaatur/Hacakthon-2026---Hriday-and-Pratheek-and-Chahel"

if [ "$VERSION" = "latest" ]; then
  DOWNLOAD_URL="https://github.com/${REPO}/releases/latest/download/${BINARY}"
else
  DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY}"
fi

echo -e "  Downloading ${Y}${BINARY}${X}..."
HTTP_CODE=$(curl -fsSL -w "%{http_code}" "$DOWNLOAD_URL" -o "/tmp/neurolang" 2>/dev/null || true)

if [ ! -f "/tmp/neurolang" ] || [ "$(wc -c < /tmp/neurolang)" -lt 1000 ]; then
  echo -e "  ${R}Error: Download failed (HTTP ${HTTP_CODE:-???}).${X}"
  echo -e "  ${D}URL: ${DOWNLOAD_URL}${X}"
  echo ""
  echo -e "  ${Y}Make sure a release exists at:${X}"
  echo -e "  ${D}https://github.com/${REPO}/releases${X}"
  echo ""
  echo -e "  ${Y}Or install via npm instead:${X}"
  echo -e "  ${G}npm install -g neurolang${X}"
  rm -f /tmp/neurolang
  exit 1
fi

chmod +x /tmp/neurolang

if [ -w "$INSTALL_DIR" ]; then
  mv /tmp/neurolang "$INSTALL_DIR/neurolang"
else
  echo -e "  Installing to ${INSTALL_DIR} ${D}(requires sudo)${X}..."
  sudo mv /tmp/neurolang "$INSTALL_DIR/neurolang"
fi

# Verify installation
if command -v neurolang &>/dev/null; then
  echo ""
  echo -e "  ${G}${B}✓ NeuroLang installed successfully!${X}"
else
  echo ""
  echo -e "  ${G}${B}✓ NeuroLang installed to ${INSTALL_DIR}/neurolang${X}"
  echo -e "  ${D}Make sure ${INSTALL_DIR} is in your PATH.${X}"
fi

echo ""
echo -e "  ${B}Get started:${X}"
echo -e "    ${Y}neurolang --help${X}"
echo -e "    ${Y}echo \"Predict species from iris\" | neurolang --run${X}"
echo ""
echo -e "  ${B}VS Code extension:${X}"
echo -e "    ${D}Search \"NeuroLang\" in the Extensions tab${X}"
echo ""
