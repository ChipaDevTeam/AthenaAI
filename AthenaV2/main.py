import os
import sys
import asyncio
from src.config import BotConfig
from src.bot import AITradingBot

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # ── CONFIGURATION ──
    config = BotConfig(
        ssid=r'42["auth",{"session":"vtftn12e6f5f5008moitsd6skl","isDemo":1,"uid":27658142,"platform":2,"isFastHistory":true,"isOptimized":true}]',  # <--- PUT YOUR SSID HERE
        asset="AUDNZD_otc",
        timeframe=60,            # 1 minute candles
        default_expiry=55,      # will be overridden by AI selector
        expiry_options=[35, 45, 50, 55, 60, 65, 75, 90],  # AI chooses from these (seconds)
        min_confidence=0.61,     # min model agreement to trade
        max_confidence=0.92,     # avoid overfitted "too good" signals
        dataset_path="./EURUSD_M1.csv",
        brain_path="brain_v6.pkl",
        trading_hours=None,  # Trade all hours
        require_indicator_alignment=True,
    )

    bot = AITradingBot(config)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nCTRL+C detected, stopping...")
        asyncio.run(bot.stop())
    except Exception as e:
        print(f"Critial Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
