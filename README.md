# APEX-PULSE

**24/7 Cloud-Ready Multi-Market (Bybit Crypto + Forex) Algorithmic Trading Bot**

APEX-PULSE is a self-hosted, cross-platform (Windows 11 + Linux/Docker) FastAPI + SQLAlchemy 2.x algorithmic trading desk built for disciplined paper/demo validation and $0/month serverless cloud operation.

### Key Capabilities (Phases 1–6)
- **Hard-Locked Paper/Demo Safety:** Defaults to `TRADING_MODE=PAPER`, `BYBIT_MODE=paper`, and `FOREX_PROVIDER=paper`. Live mainnet execution is blocked by config locks; blank API keys automatically fall back to `PaperSimBroker`.
- **Quantitative Regime & Risk Engine:** Auto-detects 4 market regimes (`TRENDING_UP`, `TRENDING_DOWN`, `RANGE_BOUND`, `HIGH_VOLATILITY`), dispatches 4 strategies (`Trend Rider`, `Range Mean Reversion`, `Volatility Breakout`, `Safety Standby`), enforces 1% risk per trade, 3% daily loss circuit breaker / Kill Switch, 3-candle anti-revenge cooldown, +1.0R partial scale-out, and trailing stops.
- **Institutional Confluence & Forex Guard:** Multi-timeframe 4H macro bias + VWAP + volume confluence scoring, plus London/NY Killzone & macro news blackout guard.
- **Zero-Lookahead Backtester & Live Candlestick Desk:** Mobile-first obsidian dashboard with 3 tabs (**Live Terminal**, **Bot Summarizer**, **Strategy Lab**) and TradingView Lightweight Charts v5.
- **Production Security & Crash Recovery (Phase 6):**
  - **Phone PIN Gate** (`DASHBOARD_PIN` + HMAC-SHA256 `HttpOnly` session cookies + dual per-IP/global anti-spoofing brute-force lockout).
  - **HTML-Safe Telegram Sleep Alerts** (`parse_mode="HTML"`, full `html.escape`, non-blocking).
  - **2-Way Execution Mode:** Continuous background loop locally (`ENABLE_BACKGROUND_LOOP=true`) OR serverless 15-minute cron ticks on Cloud Run (`ENABLE_BACKGROUND_LOOP=false` + `POST /api/cron/tick`) protected by an `asyncio.Lock`.
  - **Startup Crash Recovery:** Reconciles open positions on boot and strictly preserves an engaged Kill Switch across restarts/redeploys.

---

## 1. Local Setup & Run (Windows 11 PowerShell)

```powershell
# 1. Clone the repository
git clone https://github.com/peacefulme232-debug/apex-pulse-bot.git
cd apex-pulse-bot

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create your local .env from the template
Copy-Item .env.example .env

# 5. Run the full test suite (all 6 phases)
pytest -v

# 6. Start APEX-PULSE locally
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Open **`http://localhost:8000`** in your browser.
- Locally, if `DASHBOARD_PIN` is blank in `.env`, the dashboard opens unlocked for convenience.
- To test the Phone PIN Gate locally, set `DASHBOARD_PIN=123456` and `SESSION_SECRET=<random-hex>` in `.env` and restart Uvicorn.

---

## 2. $0/Month 24/7 Cloud Run Deployment (`africa-south1`)

APEX-PULSE is engineered to run **24/7 for \$0/month** on Google Cloud Run's Always Free tier using:
1. **Compute:** Google Cloud Run in **`africa-south1` (Johannesburg)** — Tier 1 Always Free pricing, low latency from West Africa, and supported by Bybit Global (outside both US and EEA geo-restrictions). *(Fallback region if needed: `europe-west1`.)*
2. **Database:** Free managed PostgreSQL from **[Neon](https://neon.tech)** or **[Supabase](https://supabase.com)** (SQLite on Cloud Run is ephemeral because containers scale to zero; PostgreSQL persists trades, Kill Switch state, and logs permanently).
3. **Scheduler:** 2 of Google Cloud Scheduler's 3 free monthly jobs — OR **[cron-job.org](https://cron-job.org)** (100% free alternative).

### Step A: Provision Free Cloud PostgreSQL
1. Create a free project on **Neon** or **Supabase**.
2. Copy the PostgreSQL connection string (`postgresql://user:pass@host/dbname?sslmode=require`).
3. APEX-PULSE automatically normalizes `postgres://` to `postgresql://`, disables SQLite pragmas, and enables `pool_pre_ping=True` + `pool_recycle=300` for serverless Postgres.

### Step B: Generate Your Production Secrets
Run this in PowerShell or bash to generate strong random secrets:
```bash
python3 -c "import secrets; print('SESSION_SECRET=' + secrets.token_hex(32)); print('CRON_SECRET=' + secrets.token_hex(32))"
```

### Step C: Deploy with `deploy/deploy_cloud_run.sh`
```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export DATABASE_URL="postgresql://user:pass@your-neon-host/neondb?sslmode=require"
export DASHBOARD_PIN="246810"
export SESSION_SECRET="paste-your-generated-session-secret"
export CRON_SECRET="paste-your-generated-cron-secret"

# Optional: Telegram alerts & Demo broker keys
export TELEGRAM_BOT_TOKEN=""
export TELEGRAM_CHAT_ID=""

chmod +x deploy/deploy_cloud_run.sh
./deploy/deploy_cloud_run.sh
```

### Step D: Schedule 15-Minute Ticks & Daily Summary
The script prints the exact two `gcloud scheduler jobs create http` commands for:
- `POST https://<your-service-url>/api/cron/tick` every 15 minutes (`*/15 * * * *`) with header `X-Cron-Secret: <CRON_SECRET>`.
- `POST https://<your-service-url>/api/cron/daily-summary` daily at `08:00 UTC` (`0 8 * * *`) with header `X-Cron-Secret: <CRON_SECRET>`.

**Free Alternative (`cron-job.org`):** If you prefer not to enable Cloud Scheduler, create two free HTTP `POST` cron jobs at [cron-job.org](https://cron-job.org) pointing to the same two URLs with custom request header `X-Cron-Secret: <CRON_SECRET>`.

---

## 3. Creating Safe Demo Broker Keys (NEVER Enable Withdrawals)

### Bybit Demo Trading (Crypto)
1. Log in to Bybit and switch to **Demo Trading** (virtual funds — NOT Mainnet).
2. Go to **API Management** inside Demo Trading -> **Create New Key** -> **System-generated API Keys**.
3. Permissions: enable **Unified Trading (Read-Write)** for Orders/Positions **ONLY**.
4. **NEVER enable Withdrawals, Asset Transfers, or Sub-account permissions.**
5. Set in `.env` / Cloud Run:
   ```env
   BYBIT_MODE=demo
   BYBIT_API_KEY=your_demo_key
   BYBIT_API_SECRET=your_demo_secret
   ```

### Deriv Demo (Cloud Forex)
1. Log in to Deriv and switch to your **Demo (VRTC)** account.
2. Go to **Settings -> API Token**.
3. Create a token with **Read** and **Trade** scopes **ONLY**.
4. **NEVER enable Payments or Admin scopes.**
5. Set in `.env` / Cloud Run:
   ```env
   FOREX_PROVIDER=deriv
   DERIV_API_TOKEN=your_demo_token
   DERIV_APP_ID=1089
   ```

---

## 4. Optional Telegram Sleep Alerts (`@BotFather`)

1. Open Telegram and message **`@BotFather`**.
2. Send `/newbot`, choose a name and username, and copy the **Bot Token** -> `TELEGRAM_BOT_TOKEN`.
3. Start a chat with your new bot and send `hello`.
4. Visit `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates` in your browser and copy `"chat":{"id": 123456789}` -> `TELEGRAM_CHAT_ID`.
5. Test it anytime by calling `POST /api/alerts/test` (while logged in to the dashboard).

---

## 5. Complete `.env` Reference Table

| Variable | Default | Description |
| --- | --- | --- |
| `TRADING_MODE` | `PAPER` | Global safety lock (`PAPER` recommended). |
| `DATABASE_URL` | `sqlite:///./apex_pulse.db` | SQLite WAL locally; set to Neon/Supabase `postgresql://...` on Cloud Run. |
| `LOOP_INTERVAL_SECONDS` | `10` | Seconds between local background ticks. |
| `MAX_RISK_PER_TRADE_PCT` | `1.0` | Max account equity risked per trade (1%). |
| `MAX_DAILY_LOSS_PCT` | `3.0` | Daily drawdown cap that auto-triggers Kill Switch (3%). |
| `PARTIAL_TP_ENABLED` | `true` | Scales out 50% at +1.0R and moves runner SL to breakeven. |
| `BYBIT_MODE` | `paper` | `paper` or `demo` (live mainnet blocked). |
| `BYBIT_CATEGORY` | `linear` | Bybit V5 category (`linear` or `spot`). |
| `BYBIT_API_KEY` / `BYBIT_API_SECRET` | *(blank)* | Bybit Demo API credentials. Blank forces `paper` fallback. |
| `BYBIT_SYMBOLS` | `BTCUSDT,ETHUSDT,SOLUSDT` | Comma-separated crypto symbols. |
| `BYBIT_LEVERAGE` | `2` | Hard-capped at `MAX_LEVERAGE=3`. |
| `CONFLUENCE_MIN_SCORE` | `65` | Minimum 0–100 institutional confluence score required to enter. |
| `FOREX_PROVIDER` | `paper` | `paper`, `deriv`, or `oanda`. |
| `FOREX_SYMBOLS` | `EUR_USD,GBP_USD,XAU_USD` | Comma-separated Forex / Gold symbols. |
| `FOREX_KILLZONE_ONLY` | `true` | Restricts standard FX entries to London/NY high-liquidity sessions. |
| `NEWS_BLACKOUT_MINUTES` | `30` | Minutes before/after high-impact macro events to pause new FX entries. |
| `DERIV_API_TOKEN` / `DERIV_APP_ID` | *(blank)* / `1089` | Deriv Demo WebSocket credentials. |
| `OANDA_API_KEY` / `OANDA_ACCOUNT_ID` | *(blank)* | OANDA Practice credentials (`api-fxpractice.oanda.com` only). |
| `DASHBOARD_PIN` | *(blank)* | Phone PIN Gate. Blank = unlocked locally; set 4–8+ digits in production. |
| `SESSION_SECRET` | *(blank)* | Stable HMAC-SHA256 signing key for cookies (required on Cloud Run with PIN). |
| `SESSION_TTL_HOURS` | `24` | Session cookie lifetime in hours (1–720). |
| `ENABLE_BACKGROUND_LOOP` | `true` | `true` locally; set `false` on Cloud Run to run purely on cron ticks. |
| `CRON_SECRET` | *(blank)* | Shared secret verified via `X-Cron-Secret` for `/api/cron/tick` & `/api/cron/daily-summary`. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | *(blank)* | Optional Telegram Bot API credentials for HTML sleep alerts. |

---

## 6. Mandatory Paper / Demo Safety Checklist

Before leaving APEX-PULSE running unattended:
- [ ] `TRADING_MODE=PAPER` is verified on `GET /health`.
- [ ] `DASHBOARD_PIN`, `SESSION_SECRET`, and `CRON_SECRET` are all set to strong non-default values in Cloud Run.
- [ ] `DATABASE_URL` points to persistent PostgreSQL (never local SQLite on Cloud Run).
- [ ] `ENABLE_BACKGROUND_LOOP=false` is set on Cloud Run (`--max-instances=1`, `--workers 1`).
- [ ] All broker API keys are **Demo / Practice keys ONLY** with **Withdrawals disabled**.
- [ ] `.env` and `*.db` files are never committed to Git.
- [ ] Emergency **KILL SWITCH** tested once from your phone and confirmed to stay locked across a restart.