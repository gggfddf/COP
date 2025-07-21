# 🔥 Autonomous Gold Trading AI v1.0

**Fully Autonomous High-Accuracy Pattern Learning & Prediction System for Gold Futures (XAU/USD)**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](README.md)

---

## 🎯 **System Overview**

This system delivers **high-confidence directional predictions**, **repeatable custom pattern detection**, and **annotated, interpretable results** for Gold Futures trading using cutting-edge AI techniques.

### ✅ **Key Features**
- **≥ 75% Directional Accuracy** target
- **≤ 10% MAPE** for magnitude predictions  
- **≥ 0.65 F1 Score** on extreme move detection
- **Real-time pattern discovery** and classification
- **Comprehensive annotated charts** with prediction zones
- **Multi-format outputs** (JSON, CSV, Excel, HTML charts)
- **Autonomous operation** with minimal human intervention

---

## 🏗️ **System Architecture**

### **1. Hybrid Neural Architecture**
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                           │
│              (5 Mandatory Categories)                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │  TRANSFORMER    │ ◄── Temporal Attention
         │    LAYERS       │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   CNN LAYERS    │ ◄── Pattern Detection
         │                 │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │  LSTM LAYERS    │ ◄── Sequential Dependencies
         │                 │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ FEATURE FUSION  │
         └────────┬────────┘
                  │
    ┌─────────────▼─────────────┐
    │     AUXILIARY MODULES     │
    │ • Contrastive Learning    │
    │ • Quantile Regression     │
    │ • Outlier Classifier      │
    │ • Pattern Memory          │
    └─────────────┬─────────────┘
                  │
         ┌────────▼────────┐
         │ PREDICTION HEADS│
         │ • Direction     │
         │ • Magnitude     │
         │ • Volatility    │
         └─────────────────┘
```

### **2. Feature Engineering Pipeline**

#### **📌 1. Candle Anatomy Features** *(MANDATORY)*
- Wick-to-body ratios (Top, Bottom, Relative)
- Climax candle detection (volume + body spike)
- Multi-candle engulfing patterns (3–7 candle span)
- Inside bar compression sequences
- False breakout wick traps
- Doji, Hammer, Shooting Star patterns

#### **📌 2. Volatility & Range Features** *(MANDATORY)*
- ATR percentile regime (low/mid/high)
- Implied Volatility shock zones
- Relative range expansion/compression vs 10-bar median
- Bollinger band % width analysis
- Volatility clustering detection

#### **📌 3. Time Context Features** *(MANDATORY)*
- Time of day (encoded in sine/cosine for cycles)
- Session-based strength (Asia, Europe, US)
- Days since last major breakout/fakeout
- Relative bar position within week/month
- Market open/close proximity

#### **📌 4. Price Action Context Features** *(MANDATORY)*
- Distance from recent swing high/low
- Deviation from rolling VWAP (local + multi-day)
- Time elapsed since OB/Breaker/Imbalance fill
- Clustering of equal highs/lows
- Liquidity sweep detection
- Support/Resistance level analysis

#### **📌 5. Market Psychology Features** *(MANDATORY)*
- Wick Pressure (long wick + failure to follow through)
- Volume Anomaly (volume deviation for similar range bars)
- Fakeout Pattern Score (probabilistic)
- Displacement Score (impulsive breakout vs grind)
- Fear/Greed indices
- Exhaustion signals

---

## 🔍 **Pattern Discovery System**

### **Unsupervised Pattern Discovery**
- **DBSCAN/K-Means clustering** over LSTM embeddings
- **Automatic pattern naming** based on characteristics
- **Pattern Bank** maintains 50-100 discovered classes
- **Real-time pattern classification** with similarity scoring

### **Known Pattern Templates**
- `false_break_liquidity_sweep`
- `three_bar_exhaustion` 
- `volume_climax_reversal`
- `session_gap_fill`
- `liquidity_grab_continuation`

---

## 🔮 **Prediction Output System**

### **Every Prediction Includes:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "direction_label": "UP",
  "direction_probability": 0.85,
  "expected_move_pct": 1.2,
  "expected_move_points": 24.5,
  "volatility_level": "NORMAL",
  "volatility_confidence": 0.75,
  "pattern_name": "HighVolume_Asian_Pattern",
  "pattern_similarity": 0.82,
  "confidence_score": 0.87,
  "quantile_predictions": {
    "10th": 2010.5,
    "25th": 2015.2,
    "50th": 2020.0,
    "75th": 2025.8,
    "90th": 2030.5
  },
  "risk_metrics": {
    "max_favorable": 30.5,
    "max_adverse": 15.2,
    "risk_reward_ratio": 2.0
  },
  "market_context": {
    "regime": "NORMAL_VOLATILITY",
    "session": "Asian",
    "key_levels": {
      "support": 2015.0,
      "resistance": 2025.0
    }
  }
}
```

### **Annotated Charts Include:**
- ✅ **Candle highlights** for pattern recognition
- ✅ **Prediction zones** with confidence bands
- ✅ **Probability cones** showing uncertainty
- ✅ **Pattern overlays** with similarity scores
- ✅ **Key level markers** (Support/Resistance)
- ✅ **Volume analysis** with anomaly detection

---

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone repository
git clone https://github.com/your-repo/autonomous-gold-ai.git
cd autonomous-gold-ai

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs output models data
```

### **2. Configuration**
Edit `config/config.yaml` to customize:
- Data sources and API keys
- Model hyperparameters
- Performance targets
- Output formats
- Alert settings

### **3. Run the System**

#### **Single Prediction Mode**
```bash
python main.py --mode single --training-days 90
```

#### **Continuous Prediction Mode**
```bash
python main.py --mode continuous --training-days 90
```

#### **Batch Prediction Mode**
```bash
python main.py --mode batch --training-days 90
```

---

## 📊 **Performance Targets**

| Metric | Target | Description |
|--------|--------|-------------|
| **Directional Accuracy** | ≥ 75% | Correct UP/DOWN/NEUTRAL predictions |
| **Magnitude MAPE** | ≤ 10% | Mean Absolute Percentage Error for move size |
| **Extreme Move F1** | ≥ 0.65 | F1 score for detecting 5%+ moves |
| **Confidence Calibration** | ≤ 5% | Difference between predicted and actual confidence |

---

## 📁 **Project Structure**

```
autonomous-gold-ai/
├── main.py                          # Main orchestrator
├── config/
│   └── config.yaml                  # System configuration
├── src/
│   ├── data/
│   │   └── data_collector.py        # Multi-source data collection
│   ├── features/
│   │   └── feature_engineer.py      # 5-category feature engineering
│   ├── models/
│   │   └── hybrid_model.py          # Transformer+CNN+LSTM model
│   ├── patterns/
│   │   └── pattern_discovery.py     # Pattern discovery engine
│   └── prediction/
│       └── prediction_system.py     # Comprehensive predictions
├── output/                          # Generated predictions & charts
├── models/                          # Saved models & pattern banks
├── logs/                           # System logs
└── requirements.txt                # Dependencies
```

---

## 🔧 **Advanced Usage**

### **Custom Pattern Discovery**
```python
from src.patterns.pattern_discovery import PatternDiscoveryEngine

# Initialize pattern engine
pattern_engine = PatternDiscoveryEngine()

# Discover patterns from embeddings
results = pattern_engine.discover_patterns(embeddings, market_data)

# Classify new patterns
pattern_name, similarity = pattern_engine.classify_pattern(new_embedding)
```

### **Feature Engineering**
```python
from src.features.feature_engineer import ComprehensiveFeatureEngineer

# Initialize feature engineer
feature_engineer = ComprehensiveFeatureEngineer()

# Engineer all 5 mandatory categories
engineered_data = feature_engineer.engineer_all_features(raw_data)
```

### **Model Training**
```python
from src.models.hybrid_model import HybridTransformerCNNLSTM

# Initialize hybrid model
model = HybridTransformerCNNLSTM()

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_dataloader)
```

---

## 📈 **Output Formats**

### **1. JSON Predictions**
- Structured prediction data
- API-friendly format
- Real-time integration ready

### **2. CSV/Excel Reports**
- Batch prediction analysis
- Performance tracking
- Historical comparisons

### **3. Interactive HTML Charts**
- Plotly-based visualizations
- Annotated candlestick charts
- Pattern overlay analysis
- Prediction zone visualization

### **4. Real-time Alerts**
- Email notifications
- Telegram/Slack integration
- High-confidence signal alerts

---

## 🔬 **Model Components**

### **Auxiliary Modules**

#### **Contrastive Learning Block**
- Classifies "high move" vs "low move" conditions
- Learns representations in latent space
- Improves pattern similarity detection

#### **Quantile Regression Head**
- Predicts range + confidence, not just point forecasts
- 5 quantiles (10th, 25th, 50th, 75th, 90th percentiles)
- Uncertainty quantification

#### **Outlier Movement Classifier**
- Detects 5%+ upcoming moves
- Binary classification with confidence scores
- Early warning system for extreme events

#### **Pattern Memory Encoder**
- Identifies historically recurring candle structures
- Maintains memory bank of 1000+ patterns
- Similarity-based pattern matching

---

## 🛠️ **Development & Customization**

### **Adding New Features**
1. Extend `ComprehensiveFeatureEngineer` class
2. Add feature calculation methods
3. Update configuration file
4. Test feature importance

### **Custom Pattern Types**
1. Add pattern templates to `PatternDiscoveryEngine`
2. Implement pattern detection logic
3. Update pattern naming system
4. Train pattern classifier

### **Model Architecture Changes**
1. Modify `HybridTransformerCNNLSTM` class
2. Adjust layer configurations
3. Update loss functions
4. Retrain with new architecture

---

## 📊 **Monitoring & Analytics**

### **Real-time Metrics**
- Prediction accuracy tracking
- Pattern discovery rate
- Model confidence levels
- Feature importance analysis

### **Performance Dashboards**
- Live prediction monitoring
- Pattern analysis visualization
- Performance trend analysis
- Alert management system

---

## 🔒 **Security & Risk Management**

### **Data Security**
- API key encryption
- Secure data transmission
- Local data storage options
- Access control mechanisms

### **Risk Controls**
- Position size recommendations
- Risk/reward ratio analysis
- Maximum drawdown alerts
- Correlation monitoring

---

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **PyTorch Lightning** for training framework
- **Plotly** for interactive visualizations  
- **scikit-learn** for machine learning utilities
- **pandas** for data manipulation
- **TA-Lib** for technical analysis functions

---

## 📞 **Support**

- **Documentation**: [Wiki](wiki)
- **Issues**: [GitHub Issues](issues)
- **Discussions**: [GitHub Discussions](discussions)

---

## ⚡ **Quick Demo**

```bash
# Run a quick demo with sample data
python main.py --mode single --training-days 30

# Check outputs
ls output/
# prediction.json  prediction.csv  prediction_chart.html
```

---

**🔥 Ready to revolutionize your Gold trading with AI? Get started now!**