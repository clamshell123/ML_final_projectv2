# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🔧 Common Commands

### Environment Setup
```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate nba-salary-valuation

# Pip alternative
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### Model Training
```bash
# Train pricing model (XGBoost)
python src/models/train.py

# Train duration model (Ordinal Logistic Regression)
python src/models/train_duration.py
```

### Running the Application
```bash
# Start Streamlit dashboard
streamlit run src/app/main.py

# Launch Jupyter for EDA
jupyter lab notebooks/
```

### Data Operations
```bash
# Download/update data (see download_data.py)
python download_data.py

# Verify environment setup
python verify_env.py
```

## 🏗️ Architecture Overview

### Core Principles (Immutable Laws)
1. **Target Variable**: Always predict `% of Salary Cap` (`pct_of_cap`), NEVER absolute USD
2. **Pure Skill Extraction**: Use SHAP to separate Group A (on-court skills) from Group B (market noise)
3. **Future Projections**: Apply CBA 10% annual salary cap growth rule for forecasting
4. **Algorithms**: XGBoost + SHAP only (KNN, LIME, LightGBM prohibited)

### System Layers
```
Data Collection → Cleaning & Feature Engineering → 
XGBoost + Ordinal Regression Models → SHAP Explainability → 
Streamlit Frontend (4-Layer Response) → GM Decision Support
```

### Key Directories
- `src/data/`: Data pipelines (clean.py, merge.py, features.py)
- `src/models/`: Model training (train.py, train_duration.py) and projection logic
- `src/xai/`: SHAP explainability (shap_explainer.py)
- `src/app/`: Streamlit interface (main.py + player_query.py, simulator.py pages)
- `data/raw/`: Original datasets from Kaggle/Basketball-Reference
- `data/processed/`: Cleaned, merged, feature-engineered datasets
- `models/`: Saved model artifacts (.pkl files)
- `notebooks/`: EDA and experimentation

### Four-Layer Response (Streamlit Output)
When querying a player, the system outputs:
1. **Base Valuation**: Predicted salary range (% of Cap) + contract duration probabilities
2. **Pure Skill vs Market Noise**: SHAP waterfall showing Group A (blue) vs Group B (orange)
3. **Past-to-Present Conversion**: Historical salary adjusted to modern cap terms
4. **Multi-Year Projection**: Y1-Y5 forecasts with 10% cap smoothing applied

## 📊 Data Flow
1. Raw data collected from Kaggle (player stats) and Basketball-Reference (salary cap, advanced stats)
2. Data cleaned (`src/data/clean.py`) and merged (`src/data/merge.py`)
3. Features engineered including A/B group separation (`src/data/features.py`)
4. Models trained on `pct_of_cap` target (XGBoost for pricing, Ordinal LogReg for duration)
5. SHAP explainer isolates skill vs market contributions
6. Streamlit app presents results in the four-layer format for GM decision making

## ⚠�0 Critical Reminders
- **Never** modify the target variable to predict absolute dollar amounts
- **Always** validate that SHAP values are properly split into Group A/B
- **Apply** the 10% CBA rule strictly in all projection tools
- **Use only** XGBoost and SHAP for explainability - no exceptions
- When converting to absolute dollars: `salary = predicted_pct * target_year_salary_cap`

## 🧪 Testing
Currently no test suite exists. To add tests:
1. Create test files in `tests/` directory
2. Focus on data pipeline integrity, model output ranges, and SHAP group separation
3. Verify that predictions remain in % of Cap format (0-1 range typically)