# 🚀 FULL GOLD TRADING AI EXECUTION PROMPT
# ✅ No features skipped, no modules bypassed
# ✅ Includes crash logging for debugging
# ✅ Full Transformer+CNN+LSTM pipeline with all 5 feature sets
# ✅ Designed for Cursor / notebook / CLI

from main import run_gold_ai_system
import traceback

if __name__ == "__main__":
    try:
        run_gold_ai_system(
            symbol="XAUUSD",                   # 🔁 Gold symbol (spot/futures)
            mode="live",                       # 'live', 'batch', or 'backtest'
            timeframe="1h",                    # Candle frame: '1m', '5m', '15m', '1h', '1d'
            model_type="hybrid",               # Transformer + CNN + LSTM
            features="all",                    # Use ALL 5 engineered feature categories
            pattern_recognition=True,          # Enable real-time pattern detection
            quantile_prediction=True,          # Predict with range & confidence scoring
            outlier_classification=True,       # Detect major 5%+ moves
            confidence_threshold=0.85,         # Alert only if model is ≥85% confident
            output_format=["json", "csv", "html", "png"],  # All supported formats
            alerting_enabled=True,             # Telegram/Slack/Email alert system
            save_path="./output",              # Save location for all reports
            logs_enabled=True,                 # Save detailed logs for each prediction
            verbose=True                       # Print key steps live to terminal
        )

    except Exception as e:
        print("❌ ERROR: Gold AI System failed during execution.")
        with open("crash_report.log", "w") as f:
            f.write("🔥 CRASH REPORT — Gold Trading AI\n\n")
            f.write(traceback.format_exc())
        print("📄 Log saved to crash_report.log — please review the traceback.")

    else:
        print("✅ DONE: Gold AI System ran successfully. Outputs saved in './output'")