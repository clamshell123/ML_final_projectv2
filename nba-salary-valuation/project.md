# NBA Player Salary Valuation & Contract Decision Support System
## Project Plan & Workflow Reference (3-Day Sprint — Final Integrated Version)

> **文件用途**：本文件為整個專案的唯一執行參照，綜整自初始計畫與最新技術改良共識。涵蓋核心設計共識、3天衝刺分工、架構設計、技術細節、文獻流程與寫作計劃。每次工作前先確認當前所在 Phase 與 Task，完成後打勾 `[x]`。
>
> **最後更新**：2026-06-02（v2 優化版）

---

## 目錄

1. [專案概覽與核心設計共識](#1-專案概覽與核心設計共識)
2. [5人3天衝刺分工](#2-5人3天衝刺分工)
3. [整體架構設計](#3-整體架構設計)
4. [Phase 詳細任務清單](#4-phase-詳細任務清單)
5. [每日 Workflow 規範](#5-每日-workflow-規範)
6. [文獻回顧流程](#6-文獻回顧流程)
7. [技術 Stack 與工具](#7-技術-stack-與工具)
8. [寫作與報告流程](#8-寫作與報告流程)
9. [風險管理](#9-風險管理)
10. [附錄：資料來源清單](#10-附錄資料來源清單)

---

## 🔴 核心原則對齊表（instruction.md 不可違反指令）

此表總結 instruction.md 的**四大鐵律**與其在 project.md 中的具體映射。所有開發、分析、報告工作均須嚴格遵循。

| 核心原則 | instruction.md 要求 | project.md 具體落實 | 違反後果 |
|---|---|---|---|
| **1. 目標變數：% of Cap（禁止 USD）** | 永遠預測 `pct_of_cap = salary / salary_cap`；禁用 CPI 調整；絕對金額 = pct × target_cap | Phase 4：定價模型目標變數明確為 `pct_of_cap`；Phase 5：Layer 3 提供過去→現在年份轉換；Phase 6：Layer 4 呈現多年佔比趨勢 | CBA 決策錯誤、跨時代比較失效、GM 談判基礎瓦解 |
| **2. 純實力萃取（A/B SHAP 分組）** | 特徵分為 Group A（純籃球實力）與 Group B（市場干擾）；必須用 SHAP TreeExplainer 分離兩類貢獻；輸出 shap_A_sum vs shap_B_sum | Phase 3：特徵工程明確標記 A/B；Phase 5：SHAPExplainer 強制實現 A/B 聚合與 Waterfall 圖；Phase 6：Layer 2 双色圖表呈現分離 | 無法定量說明「純實力 vs 市場溢價」，GM 決策無據 |
| **3. 未來投影（CBA 10% 規則）** | 多年合約推演須嚴格遵守帽值年漲 10%、薪資年漲 8% 的 CBA 新規；不可打折或編造 | Phase 5：`project_contract()` 函數強制內嵌 10% 帽值成長；Phase 6：Layer 4 表格逐年呈現 pct_of_cap 微降趨勢 | 預測薪資帽空間不準；GM 未來 3-4 年薪資空間規劃錯誤 |
| **4. 算法工具（XGBoost + SHAP）** | 必用 XGBoost（主要）+ SHAP（XAI）；禁用 KNN / LIME / LightGBM；杜絕其他主力演算法 | Phase 2-4：定價模型採用 XGBoost + Optuna；禁用清單明確列出 KNN/LightGBM/LIME；Phase 5：XAI 層採用 SHAP TreeExplainer | 模型解釋性破裂、A/B 拆離失效、無法重現 SHAP 計算 |

---

## 1. 專案概覽與核心設計共識

| 欄位 | 內容 |
|---|---|
| **專案名稱** | NBA Player Salary Valuation & Contract Decision Support System |
| **類型** | Explainable AI (XAI) / Decision Support System |
| **目標使用者** | NBA front-office executives（球隊管理層 / GM） |
| **三大核心目標** | 1. **跨時代定價 (Past-to-Present)**：過去球員來到現代的等價身價計算。<br>2. **純實力萃取 (Pure Skill Extraction)**：透過模型與 SHAP，剝離外部干擾（鳥權、球隊空間等），找出真實身價。<br>3. **未來合約推演 (Future Projection)**：結合新版 CBA 10% 薪資帽漲幅限制，推演未來多年合約絕對薪資。 |
| **最終交付物** | Streamlit 互動應用程式 + Contract Projection Simulator + 決策研究報告 |

### 核心設計與定義

**定義一：雙模型架構與目標變數（定價與年限分離）**

| 模型 | 演算法 | 目標變數 | 說明 |
|---|---|---|---|
| **定價模型** | XGBoost | `pct_of_cap`（合約首年薪資 / 當年薪資帽） | 完全對齊 CBA 起薪限制與頂薪天花板，跨時代比較基準一致 |
| **年限模型** | Ordinal Logistic Regression | 合約年限（1–5 年，有序類別） | 年限具嚴格階級順序，不可用一般迴歸或分類器 |

> ⚠️ **鐵律（違反無赦）**：任何情況下都**不得直接預測絕對美金薪資（USD）**，**不可使用 CPI 調整歷史薪資**。
> - ✅ 正確做法：永遠預測 `pct_of_cap`，絕對金額 = `預測 pct_of_cap × 目標年薪資帽`
> - 商業優勢：% of Cap 是 NBA 薪資帽制度下的 **真正可比度量**，超越通膨問題，支持跨時代定價
> - GM 應用：讓決策層即時對話「LeBron 值多少 cap 百分比」，而非「名義美金」

**定義二：時間範圍 → 2011–2026 賽季（小球時代）**

鎖定 2011–2026，排除 2010 年以前資料。理由：確保三分出手率（3PAr）、Pace、ORtg 等現代指標的戰術意義具一致性，避免前小球時代的統計污染。

**定義三：演算法策略 → XGBoost + SHAP A/B 分離（GM 決策的核心）**

- **不進行 KNN** 相似球員搜尋（避免市場溢價爛約污染）。
- **特徵嚴格分為兩類**：
  - **Group A（On-Court Skill/Status）**：VORP、年齡、傷病、出賽率、位置—純籃球實力
  - **Group B（External Market Noise）**：鳥權、球隊薪資空間、選秀順位、合約選項—市場干擾
- **SHAP TreeExplainer 拆解**：分別聚合 A/B 的 SHAP 值，呈現：
  - 「這位球員的 **純籃球價值** = X% of Cap」（A 類貢獻）
  - 「市場溢價/折價 = ±Y% of Cap」（B 類貢獻）
  - **商業價值**：GM 可據此判斷「這筆合約是否物有所值」，談判時有量化支撐

**禁用清單（Banned Tools）— 違反無赦**

| 工具 | 禁用原因 | 後果 |
|---|---|---|
| KNN | 爛約市場溢價會污染定價錨點；No comparable search | 導致薪資估價偏差 > 5% |
| LightGBM | 統一用 XGBoost，避免雙主力模型混淆；一致性 | 模型解釋性破裂，SHAP 計算不可靠 |
| LIME | 用 SHAP 替代，TreeExplainer 精度遠優 | 局部解釋不穩定，無法拆離 A/B 貢獻 |
| CPI 調整法 | 改用 `% of Cap` 消弭通膨問題 | 不同年代薪資帽基數不同，名義金額無可比性 |
| 直接 USD 預測 | 違背 % of Cap 核心設計 | **CBA 決策錯誤、談判基礎失效** |

---

### 核心輸出：四層回應架構（Response Builder）— **GM 決策框架**

```text
GM 輸入：球員名稱 / 想模擬的指標（傷病恢復、團隊角色等）
        ↓
┌────────────────────────────────────────────────────────────┐
│ Layer 1：基礎身價估算 (Base Valuation)                     │
│   模型預測：首年 22.5% of Cap | 年限 3 年（P=65%）         │
│   商業意義：預期簽約結構（多少薪資帽年限）                  │
│                                                           │
│ Layer 2：純實力 vs 市場雜訊 (SHAP A/B Breakdown) ⭐ 關鍵   │
│   [Group A: 純籃球實力] VORP +18%, 年齡衰退 -2%            │
│   [Group B: 市場干擾]   鳥權溢價 +4%, 球隊緊張 -1%          │
│   ─────────────────────────────────────────              │
│   純實力身價 = 18% of Cap | 市場溢價 = +4.5% of Cap        │
│   ⚡ GM 應用：該球員 22.5% 的合約含 4.5% 是市場溢價        │
│         談判下限可設 21% of Cap（去掉市場干擾）             │
│                                                           │
│ Layer 3：跨時代定價 (Past-to-Present Conversion)           │
│   轉換公式：22.5% of Cap × $140M（目標年帽） = $31.5M     │
│   商業意義：讓 2010 年球員、2020 年球員、2026 年球員       │
│              用同一把尺子（% of Cap）衡量                  │
│                                                           │
│ Layer 4：未來多年合約推演 (CBA 10% Cap Smoothing) 🔴 必須  │
│   Y1: $31.5M (22.5% of $140M)  ← 當年合約起薪            │
│   Y2: $34.0M (22.1% of $154M)  ← 合約年薪+8%, 帽+10%     │
│   Y3: $36.7M (21.6% of $169M)  ← 逐年重算 % of Cap       │
│   Y4: $39.6M (21.2% of $186M)  ← 帽值漲幅 > 薪資漲幅     │
│                                                           │
│   ⚠️ 重點：CBA 限制帽值年漲 10%，而合約薪資年漲 8%，        │
│       導致 pct_of_cap 逐年微降—這是 GM 必知的動態           │
└────────────────────────────────────────────────────────────┘
```

---

## 2. 5人3天衝刺分工

### 角色定義

| 代號 | 主責領域 | 核心任務說明 |
|---|---|---|
| **P1** 資料工程 | 資料收集、清洗、特徵工程 | pandas scraping、處理 CBA 硬性規則標籤（選秀順位、年資階梯）、建立 A/B 類特徵分群、輸出 `final_dataset.csv` |
| **P2** 模型開發 | EDA、時序切分、雙模型訓練與評估 | XGBoost（% of Cap）、Ordinal Logistic Regression（合約年限）、Optuna 調參、輸出 `best_model.pkl` + `duration_model.pkl` |
| **P3** XAI 模組 | SHAP 解析、雜訊剝離 | SHAP TreeExplainer、Group A/B 貢獻拆解、Waterfall/Beeswarm 圖表輸出 |
| **P4** 前端應用 | Streamlit UI、Contract Simulator | Streamlit + plotly、Future Projection 10% 遞迴運算、Scenario Selector |
| **P5** 報告整合 | 論文結構化撰寫、商業論述 | Zotero/BibTeX、側重純實力萃取與 10% CBA 規則對制服組的商業價值 |

---

### Day 1 — 環境建置 & 資料收集

> **目標**：所有人環境跑通、P1 產出乾淨 Raw CSV、P2 完成 `% of Cap` 分佈 EDA、P5 完成報告前兩章草稿。

| 角色 | 上午（4h） | 下午（4h） | 交付物 |
|---|---|---|---|
| **P1** | 建立 GitHub Repo 與標準資料夾結構；下載 Kaggle (2011–2026) 並確認欄位覆蓋（player, season, salary, team） | 爬取 B-Ref 薪資帽歷史表；手動整理重大傷病（Achilles / ACL / Patellar Tendon）二元清單 | Raw CSV ×3、`data_inventory.md` |
| **P2** | conda/pip 環境建置，確認 `requirements.txt` 依賴版本 | 啟動 EDA：探索 `% of Cap` 分佈；標記年齡懸崖（散點圖 + 分段線性回歸）；相關係數熱圖 | `requirements.txt`、`EDA_notebook_v0.ipynb` |
| **P3** | 精讀 SHAP TreeExplainer 文件與 NBA 開源分析專案程式碼 | 安裝 `shap` 套件，以假資料（XGBoost + 20 列隨機 DataFrame）完成 Hello World 驗證 | `XAI_methodology_note.md`、套件驗證截圖 |
| **P4** | Streamlit 開發環境初始化；設計系統 UI Wireframe（頁面架構草圖） | 建立靜態版前端（Dummy Data），確認 Metric Card 與圖表版面佈局 | UI Wireframe、Streamlit 靜態框架 |
| **P5** | 撰寫 Introduction 與研究背景初稿 | 統整 NBA 薪資預測、小球革命、XAI 方法學相關文獻（Zotero） | 報告 Sec.1–2 初稿、文獻清單 |

📌 **Day 1 晚間同步（30 min）**：確認 P1 資料年份完整（2011–2026）、全體環境無衝突、確認 Day 2 特徵介面（欄位名稱規範，見 §3.3-1）。

---

### Day 2 — 特徵工程、建模、XAI 核心開發

> **目標**：P1 輸出 `final_dataset.csv`、P2 產出 `best_model.pkl` + `duration_model.pkl`、P3/P4 完成 XAI 頁面串接。

| 角色 | 上午（4h） | 下午（4h） | 交付物 |
|---|---|---|---|
| **P1** | **時序加權特徵**：計算前三年滾動加權（t: 0.5, t-1: 0.3, t-2: 0.2），例行賽與季後賽分開計算（`Weighted_Stat_Reg` / `Weighted_Stat_PO`）；建立 Group A/B 特徵標記 | **缺失值處理**：無季後賽資料者以例行賽插補並設旗標；標記重大傷病（0/1）；標記縮短賽季；匯出 `final_dataset.csv` | `final_dataset.csv`、`feature_engineering_spec.md` |
| **P2** | 依賽季年份切分 Train/Test（2011–2022 / 2023–2026）；跑 Ridge Baseline 記錄基準 R²；**啟動 XGBoost + Optuna 調參**（目標 R² ≥ 0.75） | 訓練 **Ordinal Logistic Regression 年限模型**（目標：各年限機率輸出）；殘差分析 + 高誤差球員標記；存模型 | `best_model.pkl`、`duration_model.pkl`、`model_results.md` |
| **P3** | 串接 `best_model.pkl`；生成全局 SHAP Beeswarm 圖；實作單一球員 **SHAP Waterfall 核心計算邏輯**（Group A/B 貢獻分離） | 實作 `SHAPExplainer.group_split()` 方法（輸出 `shap_A_sum`, `shap_B_sum`）；驗證 3 位球員 Waterfall 圖正確性 | `shap_explainer.py`、SHAP 圖驗證截圖 |
| **P4** | 建置球員查詢主頁面，串接 P2 模型，即時顯示薪資估算（% of Cap + 換算美金）與市場層級標籤 | 嵌入 P3 的 SHAP Waterfall 互動圖（plotly）；Layer 4 合約推演前端雛型 | 球員查詢頁 MVP、XAI 圖嵌入完成 |
| **P5** | 向 P1/P2 收集特徵加權公式與模型參數細節，撰寫 Data & Methodology 章節 | 嵌入初步模型效能圖表（R²、殘差分析圖），撰寫 Experimental Results 草稿 | 報告 Sec.3 初稿、圖表草稿 |

📌 **Day 2 晚間同步（30 min）**：確認定價模型預測 + SHAP A/B 計算無 Bug；P1/P2/P4 對齊 Contract Simulator 後端資料格式（JSON schema）。

---

### Day 3 — 情境模擬、系統整合、報告完稿提交

> **目標**：End-to-End 功能完整、Contract Simulator 測試通過、報告全文 PDF 完成提交。

| 角色 | 上午（4h） | 下午（4h） | 交付物 |
|---|---|---|---|
| **P1** | 配合 P4 實作前端動態 Slider 連動資料流與 Contract Simulator 後端運算邏輯（10% 遞迴） | End-to-End 系統測試（10 位極端球員案例，見 §4 Phase 6），修正 Bug | Simulator 後端邏輯、系統測試報告 |
| **P2** | 殘差分析與超級明星離群值研究（頂薪 / 底薪邊界案例處理） | 建立 **Scenario Selector 後置增益公式**：Strong 市場 +3% / Weak 市場 -3% of Cap（與 P4 串接） | 最終模型穩定度報告、Scenario 運算邏輯 |
| **P3** | 挑選 3–5 個經典球員做深度 SHAP 個案剖析（數據刷子 vs 季後賽硬漢 vs 高齡重傷老將）並生成圖表 | 協助 P5 撰寫 XAI 個案解讀段落；完成系統整合驗證 | Case Study 圖表、XAI 說明段落 |
| **P4** | 實作 Contract Simulator 前端滑桿介面，整合 Scenario Selector 下拉選單（Strong / Neutral / Weak） | UI 最終細節調整、錯誤防呆（查無此球員提示）；確保 Demo-Ready | Streamlit App 完整版（Demo-Ready） |
| **P5** | 撰寫 Discussion & Conclusion，側重此決策系統為 Front Office 帶來的商業價值與侷限性 | 整合系統截圖與最終效能數據；校對全文格式；統整 BibTeX；導出最終提交 PDF | 完整報告 PDF、提交確認 |

📌 **Day 3 截止前**：全員確認 Git 最終版本已 push、報告 PDF 已提交、Demo 流程可完整跑通。

---

### 關鍵依賴關係

```
P1 final_dataset.csv     →  P2 正式訓練（Day 2 上午 硬性截止）
P2 best_model.pkl        →  P3 跑 SHAP、P4 串接後端（Day 2 中午 軟性截止）
P2 duration_model.pkl    →  P4 年限機率顯示（Day 2 下午）
P3 shap_explainer.py     →  P4 嵌入 Waterfall 圖（Day 2 下午）
P2 + P3 結果圖表         →  P5 撰寫 Results 章節（Day 2 下午）
P4 Demo 截圖             →  P5 插入報告（Day 3 下午）
```

> ⚠️ **最高風險點**：P1 的 `final_dataset.csv` 是全體的依賴起點。**備案**：若 Day 1 資料收集不完整，P2 先以 Kaggle 原始 CSV 跑 Baseline，等 P1 清理版本補齊後重跑正式版。

---

## 3. 整體架構設計

### 3.1 系統架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  [Player Search]  [Contract Simulator]  [Future Projection] │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Backend (Python)                        │
│                                                             │
│  ┌──────────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │   Data Layer     │  │   ML Module     │  │ XAI Module│  │
│  │ - clean.py       │  │ - XGBoost       │  │ - SHAP    │  │
│  │ - merge.py       │  │   (pct_of_cap)  │  │   Global  │  │
│  │ - features.py    │  │ - Ordinal LogReg│  │ - SHAP    │  │
│  │ - A/B tagging    │  │   (duration)    │  │   Local   │  │
│  └──────────────────┘  └─────────────────┘  │ - Group   │  │
│                                             │   A/B Split│  │
│                                             └───────────┘  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Response Builder                    │   │
│  │  layer1_valuation | layer2_shap_split               │   │
│  │  layer3_past2present | layer4_cap_projection        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 資料流程

```
原始資料來源（Kaggle / B-Ref / NBA Stats / Pro Sports Transactions）
        ↓
Step 1：資料收集（scraper / CSV 下載）
        ↓
Step 2：資料清理（src/data/clean.py）
   - 統一球員姓名（處理特殊字元、別名）
   - 處理缺失值（中位數填補 / 0 填補 / 刪除）
   - 移除明顯異常值（薪資 = 0、無效統計）
   - 標記縮短賽季（2011-12 / 2019-20 / 2020-21）
        ↓
Step 3：資料整合（src/data/merge.py）
   - 合併：球員薪資 + 球員統計 + 傷病紀錄
   - Join Key：(player_name, season_year)
   - 目標變數轉換：salary → pct_of_cap = salary / salary_cap
        ↓
Step 4：特徵工程（src/data/features.py）
   - 時序加權統計（0.5 / 0.3 / 0.2，例行賽與季後賽分開計算）
   - 位置 One-Hot 編碼
   - 年齡 + 球齡
   - 傷病旗標（major_injury_flag 0/1、games_played_ratio）
   - 季後賽經驗旗標（has_playoff_exp 0/1）
   - 縮短賽季旗標（is_shortened_season 0/1）
   - A/B 特徵分群標記（詳見 §3.3-2）
        ↓
Step 5：模型訓練（時序切分，防止 Data Leakage）
   → [定價] Ridge Baseline → XGBoost + Optuna 調參 → best_model.pkl
   → [年限] Ordinal Logistic Regression → duration_model.pkl
        ↓
Step 6：XAI 層（src/xai/shap_explainer.py）
   → SHAP 全局 Beeswarm + 局部 Waterfall + Group A/B 貢獻拆解
        ↓
Step 7：Streamlit 應用程式輸出（四層回應）
```

---

### 3.3-1 特徵工程詳細規格：時序加權公式

對所有連續型球員統計，分別計算例行賽與季後賽版本的三年時序加權值。

```python
# src/data/features.py

import pandas as pd

STATS_TO_WEIGHT = ['PTS', 'AST', 'REB', 'BLK', 'STL', 'TOV',
                   'PER', 'VORP', 'BPM', 'WS', '3PAr', 'TS_pct',
                   'games_played_ratio']

def compute_weighted_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    df 需已按 (player_id, season_year) 排序，且含欄位：
    {stat}_reg_t, {stat}_reg_t1, {stat}_reg_t2（例行賽前三年）
    {stat}_po_t,  {stat}_po_t1,  {stat}_po_t2（季後賽前三年，可為 NaN）
    """
    for stat in STATS_TO_WEIGHT:
        # 例行賽時序加權
        df[f'Weighted_{stat}_Reg'] = (
            df[f'{stat}_reg_t']  * 0.5 +
            df[f'{stat}_reg_t1'] * 0.3 +
            df[f'{stat}_reg_t2'] * 0.2
        )
        # 季後賽時序加權（以例行賽當年值插補 NaN）
        po_t  = df[f'{stat}_po_t'].fillna(df[f'{stat}_reg_t'])
        po_t1 = df[f'{stat}_po_t1'].fillna(df[f'{stat}_reg_t1'])
        po_t2 = df[f'{stat}_po_t2'].fillna(df[f'{stat}_reg_t2'])
        df[f'Weighted_{stat}_PO'] = po_t * 0.5 + po_t1 * 0.3 + po_t2 * 0.2

    # 季後賽經驗旗標（當年是否有真實季後賽資料）
    df['has_playoff_exp'] = df['PTS_po_t'].notna().astype(int)

    # 重大傷病旗標
    MAJOR_INJURIES = ['ACL', 'Achilles', 'Patellar Tendon']
    df['major_injury_flag'] = df['injury_type'].isin(MAJOR_INJURIES).astype(int)

    return df
```

---

### 3.3-2 特徵工程詳細規格：A/B 特徵分群

所有特徵在進入模型前需掛上群組標籤，供 SHAP 計算時做 Group A/B 貢獻拆解。

#### Group A：純實力 / 球員狀態特徵（On-Court Skill & Condition）

| 特徵欄位 | 說明 | 類型 |
|---|---|---|
| `Weighted_{stat}_Reg` | 三年時序加權例行賽統計（PTS/AST/REB/VORP/BPM/WS…） | float |
| `Weighted_{stat}_PO` | 三年時序加權季後賽統計 | float |
| `age` | 當賽季年齡 | int |
| `years_of_service` | NBA 球齡（年資） | int |
| `major_injury_flag` | 曾有重大傷病（ACL / Achilles / Patellar）0/1 | int |
| `games_played_ratio` | 實際出賽場次 / 當季總場次 | float |
| `has_playoff_exp` | 當年是否有季後賽出賽紀錄 0/1 | int |
| `position_*` | 位置 One-Hot（PG / SG / SF / PF / C） | int |
| `is_shortened_season` | 縮短賽季旗標 0/1 | int |

#### Group B：外部市場干擾特徵（Market Noise）

| 特徵欄位 | 說明 | 類型 |
|---|---|---|
| `team_payroll_ratio` | 球隊總薪資 / 當年薪資帽（球隊空間壓力指數） | float |
| `is_retained` | 是否與原球隊續約（具備鳥權）0/1 | int |
| `has_player_option` | 合約含球員選擇權 0/1 | int |
| `has_team_option` | 合約含球隊選擇權 0/1 | int |
| `is_rookie_scale` | 是否適用新秀合約薪資階梯 0/1 | int |
| `draft_pick_round` | 選秀輪次（1 / 2 / 0=未選秀） | int |

```python
# A/B 特徵分群標記字典（供 SHAP 群組計算使用）
FEATURE_GROUPS: dict[str, str] = {}

GROUP_A_PREFIXES = ['Weighted_', 'age', 'years_of_service', 'major_injury',
                    'games_played_ratio', 'has_playoff_exp', 'position_',
                    'is_shortened_season']
GROUP_B_EXACT   = ['team_payroll_ratio', 'is_retained', 'has_player_option',
                   'has_team_option', 'is_rookie_scale', 'draft_pick_round']

for col in feature_columns:
    if any(col.startswith(p) or col == p for p in GROUP_A_PREFIXES):
        FEATURE_GROUPS[col] = 'A'
    elif col in GROUP_B_EXACT:
        FEATURE_GROUPS[col] = 'B'
    else:
        FEATURE_GROUPS[col] = 'A'  # 預設歸入 A 類，保守處理
```

---

### 3.4 雙模型詳細規格

#### 3.4-1 定價模型：XGBoost（目標變數 `pct_of_cap`）

```python
# src/models/train.py

import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def objective(trial, X_train, y_train) -> float:
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1000),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state': 42,
    }
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_idx, val_idx in tscv.split(X_train):
        model = xgb.XGBRegressor(**params)
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx],
                  eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                  verbose=False)
        preds = model.predict(X_train.iloc[val_idx])
        scores.append(r2_score(y_train.iloc[val_idx], preds))
    return np.mean(scores)

# 時序切分：訓練集 2011–2022，測試集 2023–2026
# ⚠️ 禁止 random_state split，必須依 season_year 做硬切
train_mask = df['season_year'] <= 2022
X_train, y_train = df.loc[train_mask, feature_cols], df.loc[train_mask, 'pct_of_cap']
X_test,  y_test  = df.loc[~train_mask, feature_cols], df.loc[~train_mask, 'pct_of_cap']

# 效能目標：R² ≥ 0.75、RMSE < 0.04（pct_of_cap 單位）
```

#### 3.4-2 年限模型：Ordinal Logistic Regression（目標變數：合約年限 1–5 年）

```python
# src/models/train_duration.py
# 安裝：pip install mord

import mord
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 合約年限為有序類別（1 < 2 < 3 < 4 < 5），不可使用一般 LogReg 或 XGBClassifier
# mord.LogisticIT (Immediate Threshold) = 最適合此場景的 Ordinal Regression 實作

pipeline = Pipeline([
    ('scaler', StandardScaler()),         # Ordinal LogReg 對尺度敏感，需標準化
    ('clf', mord.LogisticIT(alpha=1.0))   # alpha = L2 正則化強度
])

pipeline.fit(X_train_dur, y_train_dur)  # y: int 1–5

# 取得各年限的機率分佈（供 Layer 1 輸出使用）
proba: np.ndarray = pipeline.predict_proba(X_test_dur)
# proba.shape = (n_samples, 5)，columns 對應年限 1,2,3,4,5

# 評估：重點看加權 F1（因為 class 3 佔多數）
print(classification_report(y_test_dur, pipeline.predict(X_test_dur)))
```

> **設計理由**：合約年限 1→2→3→4→5 具嚴格大小關係（非名義類別），Ordinal Logistic Regression 可以正確建模「長約傾向」的累積機率，避免一般多分類器把「2 年」和「4 年」視為等距離的錯誤假設。

---

### 3.5 Future Projection：CBA 10% Cap Smoothing 公式（**強制標準**）

⚠️ **此公式必須嚴格應用於所有多年合約推演，否則 GM 決策基礎失效**

```python
# src/models/projection.py
# ⚠️ 必須應用此公式；不應變；不應打折

def project_contract(
    base_pct_of_cap: float,
    current_salary_cap: float,
    contract_years: int,
    annual_cap_growth: float = 0.10,   # 🔴 CBA 規定的帽值年漲幅上限（不可改）
    annual_salary_escalator: float = 0.08  # 合約內年薪遞增率（CBA 標準 8%，不可改）
) -> list[dict]:
    """
    計算多年合約的逐年薪資（必須嚴格遵守 CBA 新規則）。

    核心邏輯（GM 決策依據）：
    1. 首年薪資 = base_pct_of_cap × current_salary_cap
       例：22.5% × $140M = $31.5M
    
    2. 合約內年薪以 8% 遞增（CBA 規定的標準）
       Y2 = $31.5M × 1.08 = $34.02M
    
    3. 帽值每年以 10% 成長（CBA Cap Smoothing 上限—新規則）
       Y1 帽 = $140M
       Y2 帽 = $140M × 1.10 = $154M
       Y3 帽 = $154M × 1.10 = $169.4M
    
    4. 逐年重算 pct_of_cap，供 GM 比較帽值佔比趨勢
       Y2 pct = $34.02M / $154M = 22.1% (下降)
       → GM 觀察：這筆合約相對帽值的負擔逐年遞減

    Parameters
    ----------
    base_pct_of_cap     : XGBoost 預測的首年薪資佔比
    current_salary_cap  : 目標簽約年的帽值（美金）
    contract_years      : Ordinal Model 預測的合約年數
    annual_cap_growth   : 帽值年漲幅，預設 10%（CBA 上限）
    annual_salary_escalator : 合約內年薪遞增率，預設 8%（CBA 標準）

    Returns
    -------
    list of dict，每年含：year, salary_usd, salary_cap, pct_of_cap
    """
    first_year_salary = base_pct_of_cap * current_salary_cap
    results = []

    for year in range(1, contract_years + 1):
        cap_this_year    = current_salary_cap * (1 + annual_cap_growth) ** (year - 1)
        salary_this_year = first_year_salary   * (1 + annual_salary_escalator) ** (year - 1)
        results.append({
            'year':          year,
            'salary_usd':    round(salary_this_year, 2),
            'salary_cap':    round(cap_this_year, 2),
            'pct_of_cap':    round(salary_this_year / cap_this_year, 4),
        })

    return results


# 使用範例：
# cap_projection = project_contract(
#     base_pct_of_cap=0.225,
#     current_salary_cap=140_000_000,
#     contract_years=3
# )
# 輸出：
# Y1: $31,500,000 (22.50% of $140M)
# Y2: $34,020,000 (22.04% of $154M)
# Y3: $36,741,600 (21.60% of $169.4M)
```

> **注意**：合約內年薪遞增（8%）與帽值漲幅（10%）速率不同，因此帽值佔比 `pct_of_cap` 會逐年微降——這是正常的 CBA 動態，應在報告與 UI 中明確標示。

---

## 4. Phase 詳細任務清單

### Phase 1：環境建置 & 資料收集（Day 1）

- [X] 建立 GitHub Repo，設定 `.gitignore`、`README.md`
- [X] 建立 `requirements.txt` / `environment.yml`
- [X] 資料夾結構初始化：

  ```
  nba-salary-valuation/
  ├── data/
  │   ├── raw/            # 原始資料，不修改
  │   ├── processed/      # 清理後資料
  │   └── external/       # 薪資帽歷史
  ├── notebooks/          # EDA & 實驗
  ├── src/
  │   ├── data/           # clean.py, merge.py, features.py
  │   ├── models/         # train.py, train_duration.py, evaluate.py, projection.py
  │   ├── xai/            # shap_explainer.py
  │   └── app/            # main.py, pages/
  ├── models/             # best_model.pkl, duration_model.pkl, feature_names.pkl
  ├── reports/            # 報告草稿 & 圖表
  └── tests/
  ```

- [ ] **薪資資料**：Spotrac 爬取，確認欄位（player, season, salary, team）
- [ ] **薪資帽歷史**：爬取 B-Ref，計算 `pct_of_cap = salary / salary_cap` 欄位
- [ ] **球員統計**：下載 B-Ref totals + advanced stats（2011–2026）
  - 確認欄位覆蓋：PTS, AST, REB, BLK, STL, TOV, PER, VORP, BPM, WS, 3PAr, TS%
- [ ] **傷病資料**：下載 Kaggle 10-Year Injury History CSV；確認含 injury_type 欄位
- [ ] 彙整 `data_inventory.md`：記錄每個 CSV 的欄位數、行數、年份覆蓋範圍、缺失值比例

---

### Phase 2：資料清理 & 整合（Day 1 下午 – Day 2 上午）

- [ ] **球員姓名統一化**（`src/data/clean.py`）
  - [ ] 處理重音字元（如 Nikola Jokić → Nikola Jokic）
  - [ ] 建立球員別名對照表 `player_alias.csv`
- [ ] **缺失值策略**（依欄位分類處理）

  | 欄位類型 | 處理方式 |
  |---|---|
  | 進階統計（VORP, BPM）| 當季同位置中位數填補 |
  | 傷病類型（injury_type） | NaN → 'None'（代表無重大傷病） |
  | 季後賽統計 | 以例行賽同欄位值填補，並設 `has_playoff_exp=0` |
  | 薪資 = 0 | 直接刪除（最低薪球員另有處理） |

- [ ] **異常值標記**（不刪除，改加旗標）
  - [ ] `is_shortened_season`：2011-12、2019-20、2020-21 賽季
  - [ ] `is_max_contract`：薪資 ≥ 35% of Cap（超級明星頂薪邊界）
  - [ ] `is_minimum_contract`：薪資 ≤ 2% of Cap（底薪邊界）
- [ ] **資料整合**（`src/data/merge.py`）
  - [ ] 合併 Key：`(player_name, season_year)`
  - [ ] 多隊球員處理：以出賽較多球隊的統計為主
  - [ ] 驗證：合併後無重複 Key，薪資帽欄位無 NaN
- [ ] 輸出：`data/processed/cleaned_dataset.csv`

---

### Phase 3：特徵工程（Day 2 上午）

- [ ] **時序加權統計**（`src/data/features.py`，見 §3.3-1）
  - [ ] 例行賽版本：`Weighted_{stat}_Reg`（所有 STATS_TO_WEIGHT 欄位）
  - [ ] 季後賽版本：`Weighted_{stat}_PO`（NaN 以例行賽插補）
  - [ ] 設定 `has_playoff_exp` 旗標
- [ ] **A/B 特徵分群標記**（見 §3.3-2）
  - [ ] 產出 `FEATURE_GROUPS` 字典，存入 `models/feature_groups.pkl`
- [ ] **衍生特徵**
  - [ ] `age_squared`：捕捉年齡非線性衰退
  - [ ] `age_x_injury`：年齡 × 重大傷病旗標（交互項）
  - [ ] `team_payroll_ratio`：球隊總薪資 / 當年薪資帽
- [ ] **位置 One-Hot 編碼**（PG / SG / SF / PF / C）
- [ ] 驗證：最終 `final_dataset.csv` 無 NaN（允許旗標欄位為 0）
- [ ] 輸出：`data/processed/final_dataset.csv`、`feature_engineering_spec.md`

---

### Phase 4：模型訓練 & 評估（Day 2 上午–下午）

- [ ] **資料切分**：依 `season_year` 硬切（Train: 2011–2022 / Test: 2023–2026）
  - ⚠️ 禁止 `random_split`，防止 Data Leakage
- [ ] **定價模型**（`src/models/train.py`）
  - [ ] Ridge Regression Baseline，記錄基準 R²/RMSE
  - [ ] XGBoost + Optuna（TimeSeriesSplit n=5），見 §3.4-1
  - [ ] 目標：**R² ≥ 0.75、RMSE < 0.04**（pct_of_cap 單位）
  - [ ] 殘差分析：標記高誤差案例（超級明星 / 新秀 / 底薪球員）
  - [ ] 儲存：`models/best_model.pkl`、`models/feature_names.pkl`
- [ ] **年限模型**（`src/models/train_duration.py`）
  - [ ] Ordinal Logistic Regression（`mord.LogisticIT`），見 §3.4-2
  - [ ] 輸入特徵：Group A 特徵（年限決策由球員能力主導）
  - [ ] 輸出：各年限機率分佈（5 維向量）
  - [ ] 評估：加權 F1 Score（重點關注 Class 3, 4）
  - [ ] 儲存：`models/duration_model.pkl`
- [ ] 記錄至 `reports/model_results.md`

---

### Phase 5：XAI 模組開發（Day 2 下午）

- [ ] **`SHAPExplainer` 類別**（`src/xai/shap_explainer.py`）

  ```python
  class SHAPExplainer:
      def __init__(self, model, feature_groups: dict[str, str]):
          self.explainer = shap.TreeExplainer(model)
          self.feature_groups = feature_groups

      def explain(self, X_instance: pd.Series) -> dict:
          """回傳單一球員的完整 SHAP 分析結果"""
          shap_values = self.explainer.shap_values(X_instance.values.reshape(1, -1))[0]
          # Group A/B 貢獻聚合
          shap_A = sum(v for feat, v in zip(X_instance.index, shap_values)
                       if self.feature_groups.get(feat) == 'A')
          shap_B = sum(v for feat, v in zip(X_instance.index, shap_values)
                       if self.feature_groups.get(feat) == 'B')
          return {
              'shap_values':    shap_values,
              'feature_names':  list(X_instance.index),
              'base_value':     self.explainer.expected_value,
              'prediction':     self.explainer.expected_value + shap_values.sum(),
              'shap_A_sum':     shap_A,   # 純實力貢獻總量
              'shap_B_sum':     shap_B,   # 市場干擾貢獻總量
              'pure_skill_pct': self.explainer.expected_value + shap_A,  # 去除市場干擾後的純實力估值
          }

      def plot_waterfall(self, explain_result: dict) -> go.Figure:
          """輸出 Plotly Waterfall 圖，A/B 群組以不同顏色區分"""
          ...  # 實作見附錄
  ```

- [ ] **全局 SHAP 分析（整體洞察）**
  - [ ] Feature Importance Bar Chart（Top 20 特徵效能評插：A 籍 vs B 籍）
  - [ ] SHAP Beeswarm Summary Plot（全樣本分佈）
  - [ ] ⭐ **關鍵偵測**：Group A 墨長【純實力主導】 vs Group B 矩矢【干擾次之】

- [ ] **局部 SHAP 分析（單球員細節）— GM 談判執槧**
  - [ ] 單一球員 SHAP Waterfall（Plotly，Group A 藍赫 / Group B 橘黃）
  - [ ] 案例案例（3位球員）：
    1. 數據刷子：Group A 低，Group B 高（市場溢價）→ 談判下限
    2. 實戰寶：Group A 低，其但其干受傷病 → Group B 不錯（兄弟）
    3. 超級大牌：Group A 非常高，Group B 也高（佩是安價）→ 簽約可透
  - [ ] 驗證：Waterfall sum ≈ 模型預測佔比（誤差 < 0.1% of Cap）

- [ ] 儲存：`shap_explainer.py` 可獨立執行（含 `if __name__ == '__main__'` 測試）

---

### Phase 6：Streamlit 應用程式（Day 2 下午 – Day 3）— **約 40 分鐘內 Demo 可用**

- [ ] **頁面結構**

  ```
  src/app/main.py         → Home（系統介紹、快速入口）
  src/app/pages/
    ├── player_query.py   → 球員查詢（四層回應）
    └── simulator.py      → Contract Simulator（互動什麼轉換器）
  ```

- [ ] **球員查詢頁**（`player_query.py`）— **Layer 1-4 整合【GM 談判啟發】**
  - [ ] 球員名稱搜尋欄（Autocomplete from player list）
  
  - [ ] **Layer 1** Metric Card【鳶親談判】：薪資紀圍（% of Cap + 換算美金）、年限機率分佈 Bar、市場層級（Rotation / Starter / Star / Superstar）
  
  - [ ] **Layer 2** SHAP Waterfall 互動圖【☆ 關鍵決策板☆】（Plotly，A/B 雙色）：
    - 左側組時顯示 `shap_A_sum` 【純實力 = X% of Cap】
    - 右側組時顯示 `shap_B_sum` 【市場溢價 = ±Y% of Cap】
    - 頂端粗線：`pure_skill_pct` 【"我應該出價 Z% of Cap"】
    - 💡 **GM 應用**：「這筆合約含 4.5% 的市場溢價，我談判下限應該是 21% of Cap」
  
  - [ ] **Layer 3** Past-to-Present 轉換器【跨時代定價】：輸入任意年薪資帽，自動換算絕對美金
    - 例：「LeBron 在 2010 年薪資帽 $57.7M 下值多少錢？」→ 自動計算
  
  - [ ] **Layer 4** 多年合約推演表格【CBA 10% 帽值規則】（呼叫 `project_contract()`，見 §3.5）
    - Y1-Y5 逐年 pct_of_cap 趨勢（應逐年微降）
    - 絕對金額欄位
    - ⚠️ 標註：「帽值漲 10%，薪資漲 8%，故 pct 逐年降」

- [ ] **Contract Simulator**（`simulator.py`）— **互動什麼轉換器**
  - [ ] Slider 調整各項統計（PTS, AST, REB, VORP, PER, 3P%, 出賽率等）
  - [ ] Scenario Selector 下拉選單：`Strong Market (+3% of Cap)` / `Neutral` / `Weak Market (-3% of Cap)`
  - [ ] ⭐ **即時更新機制**：當 slider 變動時，後端應：
    1. 重新運行定價模型（XGBoost predict）
    2. 重新計算 SHAP 分解（A/B 拆離）
    3. Layer 2 Waterfall 圖即時重繪（顏色變動）
    4. Layer 4 表格 Y1 行首次佔比與絕對金額實時更新

- [ ] UI 風格統一（`config.toml` Streamlit theme）
- [ ] 錯誤防呆：查無此球員提示、數值超出合理範圍警告

---

### Phase 7：整合測試（Day 3 上午）— **確保系統對 GM 的商業決策支持無虞**

- [ ] **功能測試**（10 類球員，確保邊界案例皆可正常輸出）：

  | 測試案例 | 預期挑戰點 | GM 商業意義 |
  |---|---|---|
  | 超級明星（如 LeBron James） | pct_of_cap 接近頂薪上限 35%，模型是否有 clamp？ | 頂薪球員的估值通常被帽值上限制約；模型須正確識別 |
  | 一般先發（中段薪資） | 基準驗證案例 | 應 ±3% within 的精度 |
  | 輪換球員（低薪） | 模型不應預測負值；Group B 干擾是否過度？ | 邊緣球員的市場定價敏感，易被市場干擾扭曲 |
  | 重大傷病後復出 | `major_injury_flag=1` 的衰退懲罰是否合理？ | GM 需知道傷病對定價的影響量化（SHAP） |
  | 高齡老將（38+ 歲） | 年齡衰退非線性，預測是否驟降？ | 高齡續簽涉及球隊信心；需精確定價 |
  | 新秀（< 3 年資料） | 時序加權缺少 t-1, t-2，需特殊處理 | App 應提示「資料年份不足，估算信心較低」 |
  | 退役球員 | App 應顯示「查無此球員」提示，不 crash | 用戶體驗；避免系統崩潰 |
  | 名字拼寫錯誤 | Fuzzy matching 建議最近似球員名稱 | 實用性；GM 可能記不清精確拼法 |
  | 純防守型球員 | 數據不亮麗（得分低）但薪資高，SHAP 是否正確識別防守貢獻？ | 檢驗 Group A 特徵的準確性（防守指標若缺失會誤判） |
  | 季後賽硬漢 | 例行賽數據普通，季後賽統計加成是否正確反映？ | 檢驗時序加權與季後賽旗標的有效性 |

- [ ] **效能測試**：單次查詢回應時間 < 5 秒（確保實時決策場景可用）
- [ ] **商業場景測試**：模擬 GM 談判邏輯
  - [ ] 查詢球員 A，看 SHAP 分解，決定出價下限
  - [ ] 使用 Contract Simulator 調整預期統計，觀察薪資敏感度
  - [ ] 測試 Layer 4 不同年限的 CBA 10% 規則推演是否正確

- [ ] 修正所有 Critical Bug；Known Issues 記錄於 `README.md`

---

## 5. 每日 Workflow 規範

### 每日工作節奏

```
上午開始前（10 min）
└── 打開 project.md，確認今日任務 Phase & Task
└── 確認昨日交付物已 commit 至 GitHub

執行中
└── 任務完成後立即打勾 [x]
└── 遇到 Blocker：先在群組留言，嘗試自行解決 30 min，再求助

結束後（10 min）
└── Git commit（格式：[DayX][P?] 完成了什麼）
└── 更新 project.md 任務狀態
```

### Git Commit 規範

```
[Day1][P1] 建立 Repo 結構，下載 Kaggle CSV 完成
[Day1][P2] EDA_notebook_v0 完成，% of Cap 分佈確認
[Day2][P2] XGBoost 訓練完成，R²=0.78，best_model.pkl 已存
[Day2][P2] duration_model.pkl 完成，加權F1=0.72
[Day2][P3] shap_explainer.py 完成，A/B split 驗證通過
[Day3][P4] Contract Simulator Slider 串接完成，Demo-ready
```

---

## 6. 文獻回顧流程

### 核心閱讀主題

| 主題 | 關鍵字 | 目的 |
|---|---|---|
| NBA 薪資預測 | NBA salary prediction, contract valuation, pay-for-performance | 了解現有方法與 Baseline 水準 |
| 小球革命背景 | NBA three-point revolution, pace-and-space era | 支撐 2011 年為起始點的決策 |
| XAI in Sports | Explainable AI sports analytics, interpretable ML | XAI 應用場景 |
| SHAP 方法論 | SHAP values, TreeExplainer, Shapley values | 方法論基礎 |
| 次序迴歸 | Ordinal logistic regression, proportional odds model | 年限模型理論依據 |
| 運動薪資結構 | CBA salary cap, luxury tax, Bird rights | 商業背景 |

### 文獻管理流程

1. 搜尋：Google Scholar / Semantic Scholar / ArXiv
2. 篩選：標題相關 → Abstract → 決定精讀 / 略讀
3. 記錄：

| 編號 | 作者 | 年份 | 標題 | 主要貢獻 | 與本研究的關聯 | 閱讀狀態 |
|---|---|---|---|---|---|---|
| L001 | | | | | | `[ ]` 待讀 |
| L002 | | | | | | `[ ]` 待讀 |

4. 引用：Zotero + BibTeX，報告統一 APA 格式

### 建議先讀文獻

- Berri & Schmidt (2006) — *Stumbling on Wins*（NBA 球員評估基礎）
- Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions*（SHAP 原始論文）
- Hollinger (2003) — *Pro Basketball Forecast*（PER 指標起源）
- Goldsberry (2019) — *Sprawlball*（小球革命視覺化分析）
- McCullagh (1980) — *Regression models for ordinal data*（Ordinal Logistic Regression 理論基礎）

---

## 7. 技術 Stack 與工具

### 核心技術

| 類別 | 工具 |
|---|---|
| **語言** | Python 3.10+ |
| **資料處理** | pandas, numpy |
| **機器學習** | scikit-learn, XGBoost, Optuna, mord（Ordinal Regression） |
| **XAI** | SHAP（TreeExplainer） |
| **視覺化** | plotly, matplotlib, seaborn |
| **前端應用** | Streamlit |
| **版本控制** | Git + GitHub |
| **環境管理** | conda |
| **Notebook** | Jupyter Lab |
| **文獻管理** | Zotero + BibTeX |

### `requirements.txt`（鎖定版本）

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
shap>=0.44
mord>=0.7           # Ordinal Logistic Regression（年限模型）
optuna>=3.4
streamlit>=1.30
plotly>=5.18
matplotlib>=3.7
seaborn>=0.13
requests>=2.31
beautifulsoup4>=4.12
jupyterlab>=4.0
```

> ⚠️ 不納入 `lightgbm`、`lime`、`dice-ml`（已依設計共識排除）。若選配 DiCE Counterfactual，另行 `pip install dice-ml>=0.11`。

### 開發環境設定

```bash
conda create -n nba-salary python=3.10
conda activate nba-salary
pip install -r requirements.txt
streamlit run src/app/main.py
```

---

## 8. 寫作與報告流程

### 報告章節架構

```
1. Introduction
   1.1 Background & Motivation（NBA 薪資帽制度 + 傳統評估的侷限）
   1.2 Problem Statement
   1.3 Research Objectives（三大核心目標）
   1.4 Report Structure

2. Literature Review
   2.1 NBA Player Valuation Methods
   2.2 Machine Learning in Sports Analytics
   2.3 Explainable AI (XAI) — SHAP Focus
   2.4 Ordinal Regression in Decision Modeling
   2.5 Research Gap & Our Contribution

3. Data & Methodology
   3.1 Data Sources & Collection（2011–2026，四類來源）
   3.2 Target Variable Design（% of Salary Cap 的選擇理由）
   3.3 Feature Engineering（時序加權、A/B 分群、傷病旗標）
   3.4 Model Architecture（XGBoost 定價 + Ordinal LogReg 年限）
   3.5 XAI Framework（SHAP Global + Local A/B Split）
   3.6 Future Projection（CBA 10% Cap Smoothing 公式）

4. Experimental Results
   4.1 Dataset Statistics（樣本分佈、pct_of_cap 統計描述）
   4.2 Model Performance（定價 R²/RMSE vs Ridge Baseline；年限加權 F1）
   4.3 SHAP Feature Importance Analysis（Group A vs B 全局重要性）
   4.4 Case Studies（3–5 球員：數據刷子 / 季後賽硬漢 / 高齡傷將）
   4.5 Contract Simulator Demo（截圖與互動流程說明）

5. Discussion
   5.1 Business Value for NBA Front Office（定量決策依據、談判槓桿）
   5.2 Limitations（資料偏誤、市場主觀因素、頂薪球員特例）
   5.3 Future Work（加入防守指標 DPOY 投票、球員經紀人行為模型）

6. Conclusion

References（BibTeX / APA）
Appendix A：資料字典（完整欄位清單）
Appendix B：A/B 特徵分群詳細列表
Appendix C：程式碼架構說明
```

### 寫作時程（對應 3 天）

| 章節 | 負責 | Day 1 | Day 2 | Day 3 |
|---|---|---|---|---|
| Sec.1 Introduction | P5 | 初稿 ✍️ | 修訂 | 定稿 |
| Sec.2 Literature Review | P5 | 初稿 ✍️ | 補充（Ordinal LogReg 文獻） | 定稿 |
| Sec.3 Data & Methodology | P5 + P1/P2 | 架構 | 初稿 ✍️ | 修訂 |
| Sec.4 Experimental Results | P5 + P2/P3 | — | 草稿 ✍️ | 嵌圖定稿 |
| Sec.5–6 Discussion & Conclusion | P5 | — | — | 撰寫定稿 ✍️ |
| 全文校對 + PDF 導出 | P5 | — | — | 下午完成 |

### 修改流程

```
初稿完成 → 自我校對（邏輯一致性、圖表對應）
        → P3/P4 互評（技術描述是否正確）
        → P5 統整修訂
        → 最終校對 → PDF 導出 → 提交
```

---

## 9. 風險管理

| 風險 | 可能性 | 影響 | 應對策略 |
|---|---|---|---|
| 資料來源無法爬取 / Kaggle 無法下載 | 中 | 高 | 優先確認 Kaggle 可用；B-Ref 為備案；所有 raw CSV 在 Day 1 結束前存入 `data/raw/` |
| 模型效能不佳（R² < 0.65） | 中 | 高 | Day 2 上午先跑 Baseline；若不佳立即分群建模（max-salary vs 一般）或對 `pct_of_cap` 做 log transform |
| 超級明星薪資異常拉高誤差 | 高 | 中 | 加入 `is_max_contract` 旗標；或分群（頂薪球員另建模型）；對目標變數做 log transform |
| 2020 縮短賽季資料失真 | 高 | 中 | `is_shortened_season` 旗標納入模型；測試時確認此類球員輸出合理 |
| 新秀缺少 t-1, t-2 歷史資料 | 高 | 中 | 以聯盟同位置平均填補；在 UI 顯示「資料年份不足，估算信心較低」警告 |
| SHAP 套件版本衝突 | 低 | 中 | 鎖定 `shap>=0.44`；conda 環境隔離；Day 1 完成 Hello World 驗證 |
| `mord` 套件安裝問題 | 低 | 中 | 備案：改用 `sklearn` 的多分類 LogisticRegression（降級方案，無法輸出機率序） |
| Contract Simulator 前後端格式不一致 | 中 | 中 | Day 2 晚間同步時 P1/P2/P4 對齊 JSON schema；Day 3 上午再整合 |
| 報告時間不足 | 中 | 高 | P5 邊做邊寫，Methodology 章節在 Day 2 同步撰寫；不留到 Day 3 才開始 |

---

## 10. 附錄：資料來源清單

| 資料類型 | 來源名稱 | URL | 格式 | 備注 |
|---|---|---|---|---|
| 球員薪資 + 統計（主要） | Kaggle - NBA Player Stats and Salaries 2010–2025 | https://www.kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025 | CSV | **優先使用，最穩定** |
| 薪資帽歷史 | Basketball-Reference | https://www.basketball-reference.com/contracts/salary-cap-history.html | 網頁爬取 | 計算 % of Cap 必備 |
| 球員統計（進階） | Basketball-Reference | https://www.basketball-reference.com/leagues/NBA_2025_totals.html | 網頁爬取 | 補充 advanced stats |
| 球員薪資（驗證用） | HoopsHype | https://www.hoopshype.com/salaries/players/ | 網頁爬取 | 驗證 Kaggle 資料準確性 |
| 球員統計（官方） | NBA Stats | https://www.nba.com/stats/players/traditional | API | 確認 API Rate Limit |
| 球員傷病（歷史） | Kaggle - 10 Year Injury History | https://www.kaggle.com/datasets/buyuknacar/active-nba-players-10-year-injury-history | CSV | **優先使用** |
| 球員傷病（詳細） | Pro Sports Transactions | https://www.prosportstransactions.com/basketball/ | 網頁爬取 | 補充重大傷病類型（ACL/Achilles/Patellar） |

### 資料字典（完整欄位）

| 欄位名稱 | 群組 | 類型 | 說明 |
|---|---|---|---|
| `player_name` | — | str | 球員姓名（統一格式） |
| `player_id` | — | str | 球員唯一識別碼 |
| `season_year` | — | int | 賽季年份（如 2023 代表 2022-23 賽季） |
| `position` | — | str | 位置（PG/SG/SF/PF/C） |
| `age` | A | int | 該賽季年齡 |
| `years_of_service` | A | int | NBA 球齡 |
| `salary` | — | float | 該年合約薪資（美金，僅用於計算目標變數） |
| `salary_cap` | — | float | 該賽季薪資帽（美金） |
| `pct_of_cap` | **TARGET** | float | **定價模型目標變數**：salary / salary_cap |
| `contract_years` | **TARGET** | int | **年限模型目標變數**：1–5 年 |
| `Weighted_{stat}_Reg` | A | float | 三年加權例行賽統計（所有 STATS_TO_WEIGHT 欄位） |
| `Weighted_{stat}_PO` | A | float | 三年加權季後賽統計 |
| `has_playoff_exp` | A | int | 是否有季後賽出賽紀錄（0/1） |
| `major_injury_flag` | A | int | 曾有重大傷病（ACL/Achilles/Patellar）0/1 |
| `games_played_ratio` | A | float | 實際出賽場次 / 當季總場次 |
| `is_shortened_season` | A | int | 縮短賽季旗標（0/1） |
| `is_max_contract` | A | int | 頂薪合約旗標（≥35% of Cap）0/1 |
| `is_minimum_contract` | A | int | 底薪合約旗標（≤2% of Cap）0/1 |
| `team_payroll_ratio` | B | float | 球隊總薪資 / 當年薪資帽 |
| `is_retained` | B | int | 與原球隊續約（具備鳥權）0/1 |
| `has_player_option` | B | int | 合約含球員選擇權 0/1 |
| `has_team_option` | B | int | 合約含球隊選擇權 0/1 |
| `is_rookie_scale` | B | int | 適用新秀薪資階梯 0/1 |
| `draft_pick_round` | B | int | 選秀輪次（1/2/0=未選秀） |

---

> **維護原則**：此文件是「活的」執行參照——每次完成任務後更新 `[x]` 狀態，重大技術決策請記錄原因於對應 Phase 下方，方便報告撰寫時回溯。
>
> **v2 優化摘要**：補上遺失的 Phase 2（資料清理 & 整合）完整 Checklist；補全 A/B 特徵分群完整欄位表與分群標記程式碼；補上 Ordinal Logistic Regression 完整技術規格（含 `mord` 套件與理由說明）；補上 CBA 10% Cap Smoothing 完整 Python 公式（區分帽值漲幅 10% vs 合約年薪遞增 8%）；修正 Day 2 P3 任務欄的廢棄 KNN 描述；`requirements.txt` 新增 `mord`；測試案例表格化並標記預期挑戰點；資料字典新增群組欄（A/B）與雙目標變數標記。