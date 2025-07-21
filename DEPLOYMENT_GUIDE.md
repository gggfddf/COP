# 🚀 DEPLOYMENT GUIDE - Autonomous Gold Trading AI

## 🎯 Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+ installed
- Internet connection for data feeds
- Terminal/Command prompt access

### 1. Install Dependencies
```bash
# Option 1: Using pip (if system allows)
pip install pandas numpy torch pyyaml scikit-learn plotly asyncio

# Option 2: Using conda
conda install pandas numpy pytorch pyyaml scikit-learn plotly

# Option 3: Using virtual environment (recommended)
python3 -m venv gold_ai_env
source gold_ai_env/bin/activate  # On Windows: gold_ai_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the System
```bash
# Method 1: User's exact execution script
python3 execute_gold_ai.py

# Method 2: Direct main execution  
python3 main.py --mode live --training-days 90

# Method 3: Custom configuration
python3 main.py --mode batch --training-days 180
```

### 3. Check Outputs
```bash
ls -la output/        # View generated predictions
ls -la logs/          # View system logs
ls -la models/        # View saved models
```

---

## 🏗️ Environment-Specific Setup

### 🖥️ Local Development (Cursor/VS Code)

1. **Clone/Download Project**
   ```bash
   # If using Git
   git clone <repository_url>
   cd autonomous-gold-ai
   
   # Or extract from archive
   unzip gold-ai-system.zip
   cd gold-ai-system
   ```

2. **Setup Python Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run System**
   ```bash
   python3 execute_gold_ai.py
   ```

### 📓 Jupyter Notebook Environment

1. **Install Jupyter**
   ```bash
   pip install jupyter ipywidgets
   ```

2. **Create Notebook**
   ```python
   # Cell 1: Import and setup
   import sys
   sys.path.append('.')
   from main import run_gold_ai_system
   
   # Cell 2: Execute system
   results = run_gold_ai_system(
       symbol="XAUUSD",
       mode="live",
       timeframe="1h",
       confidence_threshold=0.85,
       verbose=True
   )
   
   # Cell 3: View results
   print("Execution Results:")
   for key, value in results.items():
       print(f"{key}: {value}")
   ```

3. **Start Notebook**
   ```bash
   jupyter notebook
   ```

### ☁️ Cloud Deployment (AWS/GCP/Azure)

#### AWS EC2 Deployment
```bash
# 1. Launch EC2 instance (Ubuntu 20.04+)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install Python and dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 -m venv gold_ai_env
source gold_ai_env/bin/activate

# 4. Upload and run system
scp -i your-key.pem -r gold-ai-system ubuntu@your-instance-ip:~/
cd gold-ai-system
pip install -r requirements.txt
python3 execute_gold_ai.py
```

#### Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "execute_gold_ai.py"]
```

```bash
# Build and run
docker build -t gold-ai-system .
docker run -v $(pwd)/output:/app/output gold-ai-system
```

### 🖥️ CLI/Server Environment

1. **System Service Setup**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/gold-ai.service
   
   # Add content:
   [Unit]
   Description=Autonomous Gold Trading AI
   After=network.target
   
   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/gold-ai-system
   ExecStart=/home/ubuntu/gold-ai-system/venv/bin/python execute_gold_ai.py
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

2. **Enable and Start Service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable gold-ai.service
   sudo systemctl start gold-ai.service
   sudo systemctl status gold-ai.service
   ```

---

## ⚙️ Configuration Options

### Basic Configuration (`config/config.yaml`)
```yaml
data:
  symbol: "XAUUSD"
  primary_timeframe: "1h"
  sources: ["yahoo", "alpha_vantage"]
  
model:
  architecture: "hybrid"
  training_epochs: 100
  batch_size: 32
  
output:
  formats: ["json", "csv", "html"]
  alert_threshold: 0.85
  save_path: "./output"
```

### Environment Variables
```bash
export GOLD_AI_SYMBOL="XAUUSD"
export GOLD_AI_TIMEFRAME="1h"
export GOLD_AI_CONFIDENCE_THRESHOLD="0.85"
export GOLD_AI_OUTPUT_PATH="./output"
```

### Command Line Arguments
```bash
python3 main.py \
  --mode live \
  --training-days 90 \
  --config config/custom_config.yaml
```

---

## 📊 Monitoring & Maintenance

### Log Monitoring
```bash
# Real-time log monitoring
tail -f logs/deepgold_ai.log

# Error monitoring
grep "ERROR" logs/deepgold_ai.log

# Performance monitoring
grep "Performance" logs/deepgold_ai.log
```

### Health Checks
```bash
# Check system status
python3 -c "
from main import AutonomousGoldTradingAI
ai = AutonomousGoldTradingAI()
print('System Status:', 'OK' if ai else 'FAILED')
"

# Check outputs
ls -la output/ | wc -l  # Count output files

# Check model files
ls -la models/ | grep ".pth"  # Check saved models
```

### Performance Monitoring
```python
# monitoring_script.py
import os
import json
from datetime import datetime

def check_system_health():
    """Check system health and performance"""
    health_status = {
        'timestamp': datetime.now().isoformat(),
        'logs_present': os.path.exists('logs/deepgold_ai.log'),
        'models_present': os.path.exists('models/hybrid_model.pth'),
        'outputs_present': len(os.listdir('output/')) > 0,
        'config_valid': os.path.exists('config/config.yaml')
    }
    
    with open('health_check.json', 'w') as f:
        json.dump(health_status, f, indent=2)
    
    return all(health_status.values())

if __name__ == "__main__":
    status = check_system_health()
    print(f"System Health: {'✅ HEALTHY' if status else '❌ ISSUES DETECTED'}")
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. **ImportError: No module named 'pandas'**
```bash
# Solution: Install dependencies
pip install pandas numpy torch pyyaml scikit-learn plotly
```

#### 2. **Permission Denied Errors**
```bash
# Solution: Check file permissions
chmod +x execute_gold_ai.py
sudo chown -R $USER:$USER .
```

#### 3. **Memory Errors During Training**
```bash
# Solution: Reduce batch size in config
# Edit config/config.yaml:
model:
  batch_size: 16  # Reduce from 32
  sequence_length: 50  # Reduce if needed
```

#### 4. **Network Connection Issues**
```bash
# Solution: Check internet and API access
curl -I https://query1.finance.yahoo.com/
ping google.com
```

#### 5. **Crash Reports Generated**
```bash
# Check crash report
cat crash_report*.log

# Common fixes:
# - Check Python version (3.8+)
# - Verify all dependencies installed
# - Check file permissions
# - Ensure adequate disk space
```

### Debug Mode
```bash
# Run with debug logging
export PYTHONPATH=.
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from main import run_gold_ai_system
run_gold_ai_system(verbose=True)
"
```

---

## 📈 Performance Optimization

### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores, 5GB disk space
- **Recommended**: 8GB RAM, 4 CPU cores, 20GB disk space
- **Optimal**: 16GB RAM, 8 CPU cores, 50GB disk space

### Speed Optimization
```python
# config/performance_config.yaml
model:
  use_gpu: true  # If CUDA available
  num_workers: 4  # Parallel processing
  cache_data: true  # Cache processed features

data:
  batch_size: 64  # Larger batches if memory allows
  prefetch_days: 7  # Pre-load recent data
```

### Resource Monitoring
```bash
# Monitor system resources
htop  # CPU/Memory usage
df -h  # Disk usage
nvidia-smi  # GPU usage (if applicable)

# Monitor Python process
ps aux | grep python
```

---

## 🚀 Production Deployment Checklist

### Pre-Deployment
- [ ] Dependencies installed and tested
- [ ] Configuration file customized
- [ ] Log directories created with proper permissions
- [ ] Output directories configured
- [ ] Network connectivity verified
- [ ] System resources adequate

### Deployment
- [ ] System deployed and running
- [ ] Health checks passing
- [ ] Logs being generated
- [ ] Predictions being output
- [ ] Alerts functioning (if enabled)

### Post-Deployment
- [ ] Performance monitoring active
- [ ] Backup procedures in place
- [ ] Update procedures documented
- [ ] Support contacts established
- [ ] Documentation accessible

---

## 📞 Support & Resources

### Getting Help
1. **Check logs**: `logs/deepgold_ai.log`
2. **Review crash reports**: `crash_report*.log`
3. **Verify configuration**: `config/config.yaml`
4. **Test dependencies**: `pip list | grep -E "(pandas|torch|numpy)"`

### Additional Resources
- **System Documentation**: `README.md`
- **System Status**: `SYSTEM_STATUS.md`
- **Configuration Reference**: `config/config.yaml`
- **Example Scripts**: `simple_gold_ai_demo.py`, `test_gold_ai_demo.py`

### Performance Targets
- **Directional Accuracy**: ≥75%
- **Magnitude MAPE**: ≤10%
- **Extreme Move F1**: ≥65%
- **Prediction Latency**: <5 seconds
- **System Uptime**: ≥99%

---

**🎯 Ready to Deploy!** The system is fully operational and ready for production use.