#!/usr/bin/env bash
# ==============================================================================
# APEX-PULSE — $0/month Google Cloud Run + Cloud Scheduler Deployment Script
# Region: africa-south1 (Johannesburg)
#   - Tier 1 Always Free Cloud Run pricing
#   - Low latency from West Africa
#   - Supported by Bybit Global (outside both US and EEA geo-restrictions)
#   (Fallback region if needed: europe-west1)
# ==============================================================================
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || true)}"
SERVICE_NAME="${CLOUD_RUN_SERVICE:-apex-pulse}"
REGION="${CLOUD_RUN_REGION:-africa-south1}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: Set GCP_PROJECT_ID or run 'gcloud config set project YOUR_PROJECT_ID' first."
  exit 1
fi

# Required production environment variables (export before running this script)
: "${DATABASE_URL:?ERROR: Export DATABASE_URL (free Neon or Supabase postgresql:// connection string).}"
: "${DASHBOARD_PIN:?ERROR: Export DASHBOARD_PIN (4-8+ digit PIN to protect your dashboard).}"
: "${SESSION_SECRET:?ERROR: Export SESSION_SECRET (stable 64-hex key: python3 -c 'import secrets; print(secrets.token_hex(32))').}"
: "${CRON_SECRET:?ERROR: Export CRON_SECRET (random shared secret for Cloud Scheduler X-Cron-Secret).}"

# Optional trading & alert environment variables (defaults stay safely in PAPER mode)
TRADING_MODE="${TRADING_MODE:-PAPER}"
BYBIT_MODE="${BYBIT_MODE:-paper}"
BYBIT_API_KEY="${BYBIT_API_KEY:-}"
BYBIT_API_SECRET="${BYBIT_API_SECRET:-}"
FOREX_PROVIDER="${FOREX_PROVIDER:-paper}"
DERIV_API_TOKEN="${DERIV_API_TOKEN:-}"
DERIV_APP_ID="${DERIV_APP_ID:-1089}"
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

echo ">>> Enabling Cloud Run, Cloud Build, Artifact Registry, and Cloud Scheduler APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com \
  --project="${PROJECT_ID}"

echo ">>> Deploying ${SERVICE_NAME} to Cloud Run (${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --source="." \
  --allow-unauthenticated \
  --min-instances=0 \
  --max-instances=1 \
  --memory=512Mi \
  --cpu=1 \
  --timeout=60 \
  --set-env-vars="TRADING_MODE=${TRADING_MODE},ENABLE_BACKGROUND_LOOP=false,DATABASE_URL=${DATABASE_URL},DASHBOARD_PIN=${DASHBOARD_PIN},SESSION_SECRET=${SESSION_SECRET},CRON_SECRET=${CRON_SECRET},BYBIT_MODE=${BYBIT_MODE},BYBIT_API_KEY=${BYBIT_API_KEY},BYBIT_API_SECRET=${BYBIT_API_SECRET},FOREX_PROVIDER=${FOREX_PROVIDER},DERIV_API_TOKEN=${DERIV_API_TOKEN},DERIV_APP_ID=${DERIV_APP_ID},TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --project="${PROJECT_ID}" --region="${REGION}" --format='value(status.url)')"
echo ">>> Deployed successfully: ${SERVICE_URL}"

echo ""
echo ">>> Next Step: Create your 2 FREE Cloud Scheduler jobs (uses 2 of 3 Always Free monthly jobs):"
echo "    (Alternatively, point free https://cron-job.org at the same URLs with the X-Cron-Secret header.)"
echo ""
cat <<EOF
# 1) 15-minute Trading Tick (drives candle evaluation, risk checks, & alerts)
gcloud scheduler jobs create http apex-pulse-tick-15m \\
  --project="${PROJECT_ID}" \\
  --location="${REGION}" \\
  --schedule="*/15 * * * *" \\
  --time-zone="Etc/UTC" \\
  --uri="${SERVICE_URL}/api/cron/tick" \\
  --http-method=POST \\
  --headers="X-Cron-Secret=${CRON_SECRET}" \\
  --attempt-deadline=60s

# 2) Daily 08:00 UTC Performance & Risk Shield Digest to Telegram
gcloud scheduler jobs create http apex-pulse-daily-summary \\
  --project="${PROJECT_ID}" \\
  --location="${REGION}" \\
  --schedule="0 8 * * *" \\
  --time-zone="Etc/UTC" \\
  --uri="${SERVICE_URL}/api/cron/daily-summary" \\
  --http-method=POST \\
  --headers="X-Cron-Secret=${CRON_SECRET}" \\
  --attempt-deadline=60s
EOF
