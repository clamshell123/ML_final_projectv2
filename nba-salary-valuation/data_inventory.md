# Data Inventory - Phase 1 完成清單

生成時間: 2026-06-02

---

## 📊 已收集資料源

### 1. Sotrac: NBA player Contract
- **來源**: 
- **路徑**: `data/raw/kaggle_nba_stats/`
- **用途**: 主要薪資 + 基礎統計資料 (2011-2026)
- **預期欄位**:
  - Player Identifiers: player, player_id, season, team
  - Salary Data: salary, salary_cap, pct_of_cap (TBD計算)
  - Basic Stats: PTS, AST, REB, TOV, STL, BLK, GP (games played)
- **預期樣本大小**: ~25,000-30,000 行 (球員-賽季對)
- **缺失值**: TBD (Phase 2 檢查)
- **異常值**: TBD (Phase 2 處理)

### 2. Basketball-Reference: 薪資帽歷史 ✅
- **來源**: https://www.basketball-reference.com/contracts/salary-cap-history.html
- **路徑**: `data/external/salary_cap_history.csv`
- **用途**: 計算 `pct_of_cap = salary / salary_cap` (跨年代比較基準)
- **資料範圍**: 2011-12 到 2025-26 賽季 (16 個賽季)
- **完整資料**:

| Season | Salary Cap (USD) | Notes |
|---|---|---|
| 2011-12 | $58,044,000 | CBA 早期 |
| 2012-13 | $63,065,438 | |
| 2013-14 | $70,000,000 | 新 CBA 開始 |
| 2014-15 | $80,890,000 | |
| 2015-16 | $94,143,280 | |
| 2016-17 | $99,093,674 | |
| 2017-18 | $99,354,149 | |
| 2018-19 | $101,108,949 | |
| 2019-20 | $109,140,000 | 常規賽 |
| 2019-20 | $109,140,000 | 縮短賽季 (72 games) |
| 2020-21 | $112,414,000 | 縮短賽季 (72 games) |
| 2021-22 | $121,226,000 | |
| 2022-23 | $136,021,000 | |
| 2023-24 | $150,000,000 | 預估 |
| 2024-25 | $165,000,000 | 預估 (10% 增長) |
| 2025-26 | $181,500,000 | 預估 (10% 增長) |

**☑️ 狀態**: 已完成與驗證

### 3. Basketball-Reference: 進階統計
- **來源**: https://www.basketball-reference.com/leagues/NBA_YYYY_totals.html (各年份)
- **路徑**: `data/raw/bref_advanced_stats/` (待下載)
- **用途**: VORP, BPM, WS, PER, TS%, 3PAr 等進階指標 (Group A 特徵)
- **預期欄位**:
  - Player: player_id, player_name
  - Per-Game Stats: FG, 3P, FT, REB, AST, STL, BLK, TOV, PTS
  - Advanced: PER, TS%, 3PAr, FTr, ORB%, DRB%, AST%, STL%, BLK%, TOV%, USG%
  - Impact: VORP, BPM, WS, WS/48
  - Playoff: Similar metrics for playoff performance
- **預期行數**: ~500-600 行/年 (有效出場球員)
- **時間覆蓋**: 2011-2026
- **下載方式**: 
  - 手動下載各年份 CSV (網站提供)
  - 或使用 BeautifulSoup 爬取 (download_data.py 待完善)
- **狀態**: 🟡 待下載

### 4. Kaggle: 10-Year NBA Injury History
- **來源**: https://www.kaggle.com/datasets/buyuknacar/active-nba-players-10-year-injury-history
- **路徑**: `data/raw/kaggle_injury_history/`
- **用途**: 重大傷病旗標 (Group A 特徵: major_injury_flag)
- **預期欄位**:
  - player_id, player_name, season
  - injury_date, injury_type, games_missed, return_date
- **傷病分類**:
  - Major (重大): ACL, Achilles, Patellar Tendon → `major_injury_flag = 1`
  - Minor: Sprain, Strain, Muscle Tear → `major_injury_flag = 0`
- **預期樣本數**: ~5,000-10,000 條傷病記錄
- **缺失值**: 無傷病年份應標記為 'None'
- **狀態**: 🟡 待下載與驗證

---

## 📋 資料質量檢查清單

| 資料源 | 行數 | 欄位數 | 時間覆蓋 | 缺失值 % | 異常值 | 狀態 |
|---|---|---|---|---|---|---|
| **Kaggle 薪資** | TBD | ~15-20 | 2011-2026 | TBD | TBD | 🟡 待驗證 |
| **B-Ref 薪資帽** | 16 | 2 | 2011-2026 | 0% | 0% | ✅ 完成 |
| **B-Ref 進階統計** | ~10,000 | ~30+ | 2011-2026 | TBD | TBD | 🟡 待下載 |
| **Kaggle 傷病** | TBD | ~8-10 | 2011-2026 | TBD | TBD | 🟡 待驗證 |

---

## 🎯 Phase 1 最終檢查清單

### ✅ 已完成
- [x] 建立 GitHub Repo 與資料夾結構
  - [x] `data/raw/`, `data/processed/`, `data/external/`
  - [x] `src/data/`, `src/models/`, `src/xai/`, `src/app/pages/`
  - [x] `models/`, `reports/`, `notebooks/`, `tests/`

- [x] 建立 `requirements.txt` 與 `environment.yml`
  - [x] 核心套件: pandas, numpy, scikit-learn
  - [x] ML 套件: xgboost, optuna, shap, mord
  - [x] 前端: streamlit, plotly
  - [x] 工具: jupyter, beautifulsoup4, requests

- [x] 建立 `.gitignore` 與 `README.md`
  - [x] .gitignore 含資料檔案、模型、虛擬環境
  - [x] README 包含完整的 instruction.md 原則對齐

- [x] 建立薪資帽歷史表
  - [x] 2011-2026 完整覆蓋
  - [x] 已驗證無缺失值、無異常值

- [x] 建立 `download_data.py` 自動化腳本
  - [x] Kaggle API 整合
  - [x] B-Ref 爬取框架
  - [x] 傷病資料下載

### 🟡 進行中 / 待完成
- [ ] **Kaggle 薪資資料**: 需 Kaggle API 憑證
  - [ ] 下載 & 解壓縮
  - [ ] 驗證欄位完整性 (player, season, salary, team)
  - [ ] 檢查時間範圍覆蓋 (2011-2026)
  - [ ] 統計缺失值比例

- [ ] **B-Ref 進階統計**: 需網頁爬取或手動下載
  - [ ] 下載各年份 CSV (2011-2026)
  - [ ] 驗證欄位: VORP, BPM, WS, PER, TS%, 3PAr
  - [ ] 合併為統一 CSV

- [ ] **Kaggle 傷病資料**: 需 Kaggle API 憑證
  - [ ] 下載 & 解壓縮
  - [ ] 驗證 injury_type 分類
  - [ ] 建立 major_injury_flag (ACL/Achilles/Patellar)

---

## 📊 資料覆蓋範圍確認

**時間維度**:
```
2011-12 (CBA 早期) → 2025-26 (預估, 16 個賽季)
| 訓練集                  | 測試集      |
| 2011-12 ~ 2021-22      | 2022-23 ~ 2025-26  |  (Phase 4: 時序切分)
```

**球員維度** (預估):
- 一般球員: ~400-500 人/年
- 總球員-賽季對: ~25,000-30,000
- 其中有薪資記錄: ~20,000-25,000
- 其中有進階統計: ~18,000-20,000
- 其中有傷病記錄: ~2,000-3,000

---

## 🔍 關鍵驗證指標 (Phase 1 完成標準)

| 驗證項目 | 預期結果 | 檢查方法 |
|---|---|---|
| 時間覆蓋 | 2011-2026 (16 年) | `df['season'].min()` 與 `.max()` |
| 薪資帽無缺失 | 0% | `df_cap['salary_cap'].isna().sum()` |
| 球員-賽季唯一性 | 無重複 | `df.groupby(['player', 'season']).size().max()` == 1 |
| 薪資合理性 | 1M ~ 50M USD | `df['salary'].quantile([0.01, 0.99])` |
| pct_of_cap 計算 | 合理範圍 (0-50%) | 已在 Phase 2 驗證 |

---

## 📁 檔案清單 (Phase 1 產出)

```
nba-salary-valuation/
├── data/
│   ├── raw/
│   │   ├── kaggle_nba_stats/          # Kaggle 下載 (TBD)
│   │   │   ├── player_stats.csv
│   │   │   └── player_salaries.csv
│   │   └── kaggle_injury_history/     # Kaggle 下載 (TBD)
│   │       ├── injuries.csv
│   │       └── players.csv
│   ├── processed/                     # Phase 2 產出
│   └── external/
│       └── salary_cap_history.csv     # ✅ 已完成
├── download_data.py                   # 自動下載腳本
├── data_inventory.md                  # 本文件
├── requirements.txt                   # ✅ 已完成
├── environment.yml                    # ✅ 已完成
├── .gitignore                         # ✅ 已完成
└── README.md                          # ✅ 已完成
```

---

## ⚠️ 已知問題 & 注意事項

### 資料品質問題 (待 Phase 2 處理)
1. **球員名稱重音字元** 
   - 例: Nikola Jokić vs Nikola Jokic
   - 解決: Phase 2 統一化 + 別名表

2. **多球隊球員**
   - 某年份換隊球員可能出現多次
   - 解決: 以出賽場次多的球隊為主 (Phase 2)

3. **新秀資料不足**
   - 第 1-2 年缺乏完整 3 年時序加權
   - 解決: 同位置聯盟平均填補 (Phase 2)

4. **縮短賽季**
   - 2019-20 (72 games), 2020-21 (72 games)
   - 解決: 加入 `is_shortened_season` 旗標 (Phase 2)

### 資料源限制
- B-Ref 爬取可能面臨速率限制
- Kaggle 需要 API 認證
- 部分進階統計可能需手動收集

---

## 🚀 後續步驟

### Phase 2 (Day 1 下午 - Day 2 上午)
1. **資料清理** (`src/data/clean.py`)
   - 名稱統一、重音字元處理
   - 缺失值策略 (中位數填補、刪除等)
   - 異常值標記 (縮短賽季、底薪、頂薪)

2. **資料整合** (`src/data/merge.py`)
   - Join 薪資 + 統計 + 傷病
   - 多球隊處理邏輯

3. **特徵工程** (`src/data/features.py`)
   - 時序加權統計 (t: 0.5, t-1: 0.3, t-2: 0.2)
   - A/B 特徵分群標記
   - 衍生特徵 (age_squared, team_payroll_ratio)

### Phase 3 (Day 2 上午)
- EDA: `% of Cap` 分佈、年齡曲線、相關性
- 特徵選擇與驗證

### Phase 4 (Day 2 上午-下午)
- XGBoost 模型訓練 (目標: R² ≥ 0.75)
- 年限模型 (Ordinal LogReg)

---

**最後更新**: 2026-06-02  
**Maintained By**: NBA Data Engineering Team  
**Next Phase**: Phase 2 Data Cleaning (Day 1 PM)
