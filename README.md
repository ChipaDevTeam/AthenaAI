# AthenaAI Trading Bot
An advanced AI-powered trading bot for PocketOption using ensemble machine learning.

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features
- **Ensemble Learning**: Uses 5 models (SGD, PA, NB, GBM, RandomForest)
- **Feature Engineering**: 40+ core features + 17 experimental features
- **Adaptive Strategy**: Automatic regime detection and performance-based tuning
- **Risk Management**: Kelly criterion staking and daily loss limits
- **Persistence**: Saves trained "brain" to disk 

## Installation

1. Install Python 3.9+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Set environment variables or create a launch script:

| Variable | Description | Default |
|----------|-------------|---------|
| `PO_SSID` | Your PocketOption Session ID (Required) | |
| `PO_ASSET` | Asset to trade (e.g. EURUSD) | EURUSD |
| `PO_TIMEFRAME` | Candle timeframe in seconds | 60 |
| `PO_BASE_STAKE` | Minimum trade amount | 10.0 |
| `PO_MAX_STAKE` | Maximum trade amount | 100.0 |
| `PO_MIN_CONF` | Minimum AI confidence (0.0-1.0) | 0.60 |

## Usage
```bash
python main.py
```

## Disclaimer
This software is for educational purposes only. Use at your own risk.
