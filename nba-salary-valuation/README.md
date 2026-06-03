# NBA Player Salary Valuation & Contract Decision Support System

## 🎯 Project Overview

This project builds an **Explainable AI (XAI) decision support system** for NBA front-office executives (GMs) to:
1. **Cross-era valuation (Past-to-Present)**: Calculate equivalent salary for historical players in modern salary cap terms.
2. **Pure skill extraction**: Isolate basketball value from market noise using SHAP explainability.
3. **Future contract projections**: Apply CBA 10% salary cap growth rule to multi-year forecasting.

## 🏗️ Architecture

```
Data Collection (Kaggle/B-Ref/Spotrac)
         ↓
Data Cleaning & Feature Engineering
         ↓
XGBoost Salary Model + Ordinal Regression Duration Model
         ↓
SHAP Explainability (Group A/B Split)
         ↓
Streamlit Frontend (4-Layer Response Builder)
         ↓
GM Decision Support Dashboard
```

## 📋 Core Principles (From instruction.md)

### ⚠️ Four Immutable Laws

1. **Target Variable: `pct_of_cap` (NEVER USD)**
   - Always predict `pct_of_cap = salary / salary_cap`
   - Absolute dollars = `predicted pct × target year's cap`
   - Eliminates inflation issues & enables cross-era comparison

2. **Pure Skill Extraction (SHAP A/B Grouping)**
   - **Group A**: On-court skill (VORP, age, injury, games played, position)
   - **Group B**: Market noise (Bird rights, team cap space, draft pick, contract options)
   - Use SHAP TreeExplainer to separate contributions
   - Output: `pure_skill_pct = base + shap_A_sum`, `market_noise_pct = shap_B_sum`

3. **Future Projections (CBA 10% Rule)**
   - Salary cap grows **exactly 10%/year**
   - Player salary grows 8%/year (standard escalator)
   - `pct_of_cap` gradually decreases (this is expected)

4. **Algorithms: XGBoost + SHAP Only**
   - Primary model: **XGBoost** (pricing)
   - Duration model: **Ordinal Logistic Regression** (contract years 1-5)
   - XAI: **SHAP TreeExplainer**
   - ❌ Banned: KNN, LIME, LightGBM, CPI adjustment, direct USD prediction

## 📁 Project Structure

```
nba-salary-valuation/
├── data/
│   ├── raw/            # Original datasets (Kaggle, B-Ref)
│   ├── processed/      # Cleaned & merged datasets
│   └── external/       # Salary cap history
├── notebooks/          # EDA & experimentation
├── src/
│   ├── data/
│   │   ├── clean.py           # Data cleaning pipeline
│   │   ├── merge.py           # Data integration
│   │   └── features.py        # Feature engineering (weighted stats, A/B grouping)
│   ├── models/
│   │   ├── train.py           # XGBoost pricing model
│   │   ├── train_duration.py  # Ordinal LogReg for contract years
│   │   ├── evaluate.py        # Model evaluation & residual analysis
│   │   └── projection.py      # CBA 10% cap projection logic
│   ├── xai/
│   │   └── shap_explainer.py  # SHAP A/B decomposition
│   └── app/
│       ├── main.py            # Streamlit home page
│       └── pages/
│           ├── player_query.py    # 4-layer response builder
│           └── simulator.py       # Contract simulator
├── models/             # Saved model artifacts
├── reports/            # Analysis outputs & report drafts
├── tests/              # Unit tests
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
```

## 🚀 Setup & Installation

### Option 1: Conda Environment
```bash
conda env create -f environment.yml
conda activate nba-salary-valuation
```

### Option 2: Pip
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 📊 Data Sources

| Type | Source | Format | Notes |
|---|---|---|---|
| Salary & Stats (Main) | Kaggle - NBA Player Stats 2010-2025 | CSV | **Primary source** |
| Salary Cap History | Basketball-Reference | Web scrape | Required for `pct_of_cap` |
| Advanced Stats | Basketball-Reference | Web scrape | VORP, BPM, WS, etc. |
| Injury History | Kaggle - 10-Year Injury | CSV | For injury flags |

**Key Requirement**: Data from **2011-2026 seasons only** (small-ball era consistency)

## 🎬 Execution Plan (3-Day Sprint)

### Day 1: Environment & Data Collection
- [ ] GitHub repo + folder structure
- [ ] Data downloads (Kaggle, B-Ref scraping)
- [ ] EDA on `pct_of_cap` distribution
- [ ] Data inventory document

### Day 2: Modeling & XAI
- [ ] Feature engineering (weighted stats, A/B grouping)
- [ ] XGBoost training (target: R² ≥ 0.75)
- [ ] Ordinal regression for contract duration
- [ ] SHAP TreeExplainer implementation
- [ ] Streamlit player query page (Layers 1-4)

### Day 3: Integration & Reporting
- [ ] Contract Simulator (interactive sliders)
- [ ] CBA 10% projection simulator
- [ ] Full system testing (edge cases)
- [ ] Report writing & PDF export

## 💻 Usage

### Train Models
```bash
python src/models/train.py
python src/models/train_duration.py
```

### Run Streamlit App
```bash
streamlit run src/app/main.py
```

### Jupyter EDA
```bash
jupyter lab notebooks/
```

## 📈 Model Performance Targets

| Model | Target Metric | Minimum Threshold |
|---|---|---|
| Pricing (XGBoost) | R² Score | ≥ 0.75 |
| Pricing | RMSE (% of Cap) | < 0.04 |
| Duration (Ordinal) | Weighted F1 | ≥ 0.70 |

## 🔬 Four-Layer Response Builder (Streamlit Output)

When GM queries a player:

```
Layer 1: Base Valuation
  ↓ Predicted salary range (% of Cap) + contract duration probabilities

Layer 2: Pure Skill vs Market Noise (SHAP Waterfall)
  ↓ Group A (blue): Pure basketball value
  ↓ Group B (orange): Market premium/discount
  
Layer 3: Past-to-Present Conversion
  ↓ "If this player signed in 2010 (cap=$57.7M), what's today's equivalent?"
  
Layer 4: Multi-Year CBA Projection
  ↓ Y1-Y5 salary forecasts with 10% cap smoothing applied
```

## ⚠️ Known Limitations

- Regression model struggles with superstar salary ceilings (35% cap limit)
- Injury severity classification is binary (major vs none)
- Playoff experience limited for younger players
- Market dynamics (free agency momentum, team desperation) not explicitly modeled

## 📝 Report Structure

1. **Introduction**: NBA salary cap system, valuation challenges, project objectives
2. **Literature Review**: NBA valuation methods, XAI in sports, ordinal regression
3. **Data & Methodology**: Dataset description, feature engineering (A/B split), model design
4. **Experimental Results**: Model performance, SHAP global/local analysis, case studies
5. **Discussion**: Business value, limitations, future work
6. **Conclusion**

## 👥 Team Roles (3-Day Sprint)

| Role | Primary Responsibility | Key Deliverables |
|---|---|---|
| P1 (Data Eng) | Data collection, cleaning, feature engineering | `final_dataset.csv`, A/B feature groups |
| P2 (ML) | Model training, evaluation, parameter tuning | `best_model.pkl`, `duration_model.pkl` |
| P3 (XAI) | SHAP analysis, A/B decomposition | `shap_explainer.py`, case study visualizations |
| P4 (Frontend) | Streamlit UI, Contract Simulator | MVP app, live demo |
| P5 (Report) | Methodology writing, results integration | Full paper, PDF export |

## 📚 Key References

- Berri & Schmidt (2006) - *Stumbling on Wins*
- Lundberg & Lee (2017) - *A Unified Approach to Interpreting Model Predictions* (SHAP)
- Hollinger (2003) - *Pro Basketball Forecast* (PER metric)
- Goldsberry (2019) - *Sprawlball* (3-point revolution)
- McCullagh (1980) - *Regression models for ordinal data*

## 📧 Contact & Changelog

**Version**: v2.1 (Optimized, instruction.md aligned)  
**Last Updated**: 2026-06-02  
**Next Milestone**: Phase 1 completion (Day 1 data collection)

---

**⚡ Prime Directive**: Every prediction must be in `% of Cap`, every SHAP output must split A/B, every projection must apply CBA 10% rule. No exceptions.
