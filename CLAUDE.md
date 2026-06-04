```markdown
# CLAUDE.md

This file provides strict guidance to Claude Code (claude.ai/code) when working with code in this repository. 
Every architectural decision, feature generation, and refactoring step must align with the laws below[cite: 1].

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

### Model Pipeline & Training

```bash
# Data pipelines (cleaning, merging, advanced feature engineering)
python src/data/features.py

# Train models with automatic hyperparameter tuning (HalvingGridSearchCV)
python src/models/train.py

```

### Running the Application (Two-Terminal Production Architecture)

```bash
# Terminal 1: Launch FastAPI Backend (Port 8000)
uvicorn src.app.api:app --reload

# Terminal 2: Launch Streamlit Dashboard UI (Port 8501)
streamlit run src/app/dashboard.py

```

### Data Operations & EDA

```bash
# Download/update data from raw sources
python download_data.py

# Launch Jupyter for EDA and feature testing
jupyter lab notebooks/

# Verify environment setup
python verify_env.py

```

## 🏗️ Architecture Overview

### Core Principles (Immutable Laws)

1. **Target Variable Boundary**: Always model and predict `% of Salary Cap` (`Cap_Pct` / `pct_of_cap`), **NEVER absolute USD**. Absolute dollars are always derived: `USD_Salary = Predicted_Pct * Target_Year_Salary_Cap`.


2. **Dynamic Pure Skill Extraction**: Use SHAP to logically decompose predictions into two distinct buckets:


* **Group A (Pure Basketball Skill)**: On-court impact vectors (`VORP`, `BPM`, `PTS`, `MP`, tracking and playoff elevation indices).


* **Group B (External Market Noise)**: Ecosystem factors (`is_retained` Bird Rights, `Is_Big_Market` team market-size metrics, franchise financial status).
* *Formula*: `Pure_Skill_Valuation = Base_Value + Sum(SHAP_Group_A)`.


3. **Future Projections**: Apply CBA "Cap Smoothing" rules strictly. Assume a flat **10% year-over-year growth** for future league Salary Caps.


4. **Tool Restrictions**: Utilize **XGBoost Regressor** (for valuation), **Pipeline Scaled Logistic Regression** (for contract duration), and **SHAP TreeExplainer** only. KNN, LIME, LightGBM, or CPI adjustments are strictly prohibited.



### System Layers

```
Data Collection → Cleaning & Feature Engineering → 
XGBoost + Scaled Logistic Regression Models → SHAP Explainability → 
Supabase Central Database (Historical Pre-computed Values) →
FastAPI Endpoints → Streamlit Frontend Dashboard

```

### Key Directories

* `src/data/`: Data pipelines (`clean.py`, `merge.py`, `features.py` featuring dynamic A/B split logic).


* `src/models/`: Model training (`train.py`), inference script (`inference.py`), and saved `.pkl` model artifacts.
* `src/app/`: Web deployment layer (`api.py` for FastAPI backend, `dashboard.py` for Streamlit interface).
* `data/raw/`: Original datasets from Kaggle/Basketball-Reference.


* `data/processed/`: Cleaned, merged, and feature-engineered datasets containing `featured_nba_data.csv`.
* `notebooks/`: EDA and hyperparameter experimentation.



### Four-Layer Response (Streamlit Output)

When querying a player, the dashboard renders:

1. **Base Valuation**: Predicted salary range (% of Cap) + contract duration probabilities.


2. **Pure Skill vs Market Noise**: Plotly Waterfall chart showing Group A (Pure Basketball Value) vs Group B (Market Premium/Discount).
3. **Past-to-Present Conversion**: Historical player production adjusted to modern cap environments.


4. **Multi-Year Projection**: Y1-Y5 contract forecasts with the 10% cap smoothing rules applied.



## ⚠0 Critical Production Reminders

* **Path Resolution**: When configuring file paths inside `src/app/api.py`, resolve path lookups using absolute directory configurations (`BASE_DIR = os.path.abspath(...)`) to guarantee Uvicorn's hot-reload performs flawlessly.
* **Feature Inclusivity**: Ensure performance tracking code inside `inference.py` evaluates Group A columns dynamically (`[col for col in X.columns if col not in group_b]`) so that no newly added performance vectors are excluded from the SHAP summation.
* **Logistic Scale Sensitivity**: Always wrap the contract duration model in a Scikit-learn Pipeline containing `StandardScaler` prior to fitting to eliminate convergence warning issues.

## 🧪 Testing

Currently no test suite exists. To add tests:

1. Create test files in `tests/` directory.


2. Focus on data pipeline integrity, model output ranges, and SHAP group separation.


3. Verify that predictions remain in % of Cap format (0-1 range typically).


```