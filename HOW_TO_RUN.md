# Research Project Execution Guide

**Project:** Matrix-based Prediction Approach for Intraday Instantaneous Volatility Vector
**Method:** TIP-PCA (Tensor-Invariant Principal Component Analysis)

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.7+
- Required packages (see requirements.txt)

### Step 1: Navigate to Project Directory
```bash
cd /Users/nehapriya/Desktop/research_project
```

### Step 2: Install Dependencies (First Time Only)
```bash
pip install -r requirements.txt
```

### Step 3: Run Phase 1 (Data Generation)
```bash
python3 run_data_generation.py
```

### Step 4: Run Phase 2 (TIP-PCA Implementation)
```bash
python3 run_tippca.py
```

### Step 5: View Results
```bash
open data/processed/volatility_analysis.png
open data/processed/tippca_analysis.png
```

## 📋 Detailed Execution Steps

### Phase 1: Data Generation
```bash
# Navigate to project directory
cd /Users/nehapriya/Desktop/research_project

# Run data generation script
python3 run_data_generation.py

# What this does:
# - Generates 98,280 synthetic data points using GBM
# - Creates volatility matrices (252 × 390)
# - Saves data files and visualizations
# - Outputs statistical analysis

# Expected output:
# - data/raw/synthetic_dataset_full.csv (2.0 MB)
# - data/processed/volatility_matrix_full_raw.csv (2.0 MB)
# - data/processed/volatility_matrix_full_normalized.csv (1.8 MB)
# - data/processed/volatility_analysis.png (705 KB)
```

### Phase 2: TIP-PCA Implementation
```bash
# Navigate to project directory
cd /Users/nehapriya/Desktop/research_project

# Run TIP-PCA script
python3 run_tippca.py

# What this does:
# - Loads volatility matrix from Phase 1
# - Performs TIP-PCA decomposition
# - Separates signal and noise components
# - Generates volatility predictions
# - Creates analysis visualizations

# Expected output:
# - data/processed/low_rank_component.csv (1.8 MB)
# - data/processed/noise_component.csv (1.8 MB)
# - data/processed/volatility_predictions.csv (39 KB)
# - data/processed/tippca_analysis.png (316 KB)
```

## 🔧 Troubleshooting

### Module Not Found Error
```bash
# Solution: Install missing packages
pip install numpy pandas matplotlib scipy
```

### Permission Denied Error
```bash
# Solution: Make script executable (Mac/Linux)
chmod +x run_data_generation.py
chmod +x run_tippca.py
```

### File Not Found Error
```bash
# Solution: Ensure you're in the correct directory
cd /Users/nehapriya/Desktop/research_project
pwd  # Should show: /Users/nehapriya/Desktop/research_project
```

## 📊 Running Individual Modules

### Run Data Simulator
```bash
python3 -c "from src.data import DataSimulator; sim = DataSimulator(); print('Simulator working!')"
```

### Run TIP-PCA Model
```bash
python3 -c "from src.models import TIPPCA; model = TIPPCA(); print('TIP-PCA working!')"
```

### Run Data Processor
```bash
python3 -c "from src.data import DataProcessor; proc = DataProcessor(); print('Processor working!')"
```

## 🌐 Git Operations

### Check Repository Status
```bash
cd /Users/nehapriya/Desktop/research_project
git status
```

### View Commit History
```bash
git log --oneline
```

### Push Changes to GitHub
```bash
git add .
git commit -m "Your commit message"
git push origin main
```

## 📈 Monitoring Progress

### Check Generated Files
```bash
# List all data files
ls -lh data/raw/
ls -lh data/processed/

# List all visualizations
ls -lh data/processed/*.png

# Check file sizes
du -sh data/
```

### View Results in Real-time
```bash
# Watch data directory changes (Mac/Linux)
watch -n 1 'ls -lh data/processed/'

# Or just check updates
ls -lt data/processed/ | head -10
```

## 🔍 Viewing Results

### Open Visualizations
```bash
# Mac
open data/processed/volatility_analysis.png
open data/processed/tippca_analysis.png

# Linux
xdg-open data/processed/volatility_analysis.png
xdg-open data/processed/tippca_analysis.png

# Windows
start data/processed/volatility_analysis.png
start data/processed/tippca_analysis.png
```

### View CSV Files
```bash
# View first few lines
head -20 data/raw/synthetic_dataset_full.csv
head -10 data/processed/volatility_predictions.csv

# View file info
wc -l data/processed/volatility_matrix_full_normalized.csv
```

## 🚀 Complete Workflow (From Scratch)

```bash
# 1. Navigate to project
cd /Users/nehapriya/Desktop/research_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Phase 1
python3 run_data_generation.py

# 4. Run Phase 2
python3 run_tippca.py

# 5. View results
open data/processed/volatility_analysis.png
open data/processed/tippca_analysis.png

# 6. Check git status
git status

# 7. Push changes if needed
git add .
git commit -m "Updated results"
git push origin main
```

## 💡 Tips for Best Performance

1. **Use Python 3.7+** for best compatibility
2. **Run phases in order** (Phase 1 first, then Phase 2)
3. **Check disk space** - generated files are ~12 MB total
4. **Monitor CPU usage** - matrix operations can be intensive
5. **Backup results** - commit to git regularly

## 📞 Getting Help

### Check Python Version
```bash
python3 --version
```

### Check Installed Packages
```bash
pip list | grep -E "numpy|pandas|matplotlib|scipy"
```

### Test Environment
```bash
python3 << 'EOF'
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Matplotlib version: {plt.__version__}")
print("✅ All dependencies installed!")
EOF
```

---

**Project Location:** `/Users/nehapriya/Desktop/research_project`
**GitHub Repository:** https://github.com/neha-priyaa/research-project

**Status:** Ready to run! 🚀
