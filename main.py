import asyncio
import os
import sys
from src.bot import AITradingBot
from src.config import BotConfig

def main():
    # --- Load config from env or defaults ---
    # Expiry options: comma-separated seconds (e.g. "60,120,180,300")
    expiry_str = os.environ.get("PO_EXPIRY_OPTIONS", "60,120,180,300")
    try:
        expiry_opts = tuple(int(x.strip()) for x in expiry_str.split(",") if x.strip())
    except ValueError:
        print(f"Warning: Invalid PO_EXPIRY_OPTIONS '{expiry_str}', defaulting to (60, 120, 180, 300)")
        expiry_opts = (60, 120, 180, 300)

    cfg = BotConfig(
        ssid=os.environ.get("PO_SSID", ""),
        asset=os.environ.get("PO_ASSET", "EURUSD"),
        timeframe=int(os.environ.get("PO_TIMEFRAME", "60")),
        expiry_options=expiry_opts,
        default_expiry=int(os.environ.get("PO_DEFAULT_EXPIRY", "120")),
        base_stake=float(os.environ.get("PO_BASE_STAKE", "10.0")),
        max_stake=float(os.environ.get("PO_MAX_STAKE", "100.0")),
        min_confidence=float(os.environ.get("PO_MIN_CONF", "0.60")),
        max_daily_loss=float(os.environ.get("PO_MAX_DAILY_LOSS", "300.0")),
        dataset_path=os.environ.get("PO_DATASET", ""),
    )

    if not cfg.ssid:
        print("=" * 60)
        print("  ERROR: No SSID provided!")
        print()
        print("  Set your PocketOption session ID:")
        print("    export PO_SSID='your-session-id-here'  # Linux/Mac")
        print("    set PO_SSID=your-session-id-here       # Windows")
        print()
        print("  Or create a .env file with your configuration.")
        print("=" * 60)
        sys.exit(1)

    bot = AITradingBot(cfg)

    async def run():
        try:
            await bot.start()
        except KeyboardInterrupt:
            await bot.stop()
        except Exception as e:
            print(f"Critcal error: {e}")
            await bot.stop()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
