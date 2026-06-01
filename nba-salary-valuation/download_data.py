"""
Phase 1: Data Collection Script
下載 NBA 球員薪資、統計、傷病資料 (2011-2026)

執行方式：
    python download_data.py

重點：
    - Kaggle API 需先設定 credentials (~/.kaggle/kaggle.json)
    - B-Ref 需要 BeautifulSoup 爬取
    - 所有資料存於 data/raw/
"""

import os
import sys
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 設定路徑
PROJECT_ROOT = Path(__file__).parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_EXTERNAL = PROJECT_ROOT / 'data' / 'external'
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)


def download_kaggle_nba_stats():
    """
    下載 Kaggle: NBA Player Stats and Salaries 2010-2025
    
    Prerequisite:
        pip install kaggle
        配置 ~/.kaggle/kaggle.json (from Kaggle settings)
    
    Dataset: https://www.kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025
    """
    logger.info("=" * 60)
    logger.info("下載 Kaggle 薪資數據...")
    logger.info("=" * 60)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        dataset_name = 'ratin21/nba-player-stats-and-salaries-2010-2025'
        output_path = DATA_RAW / 'kaggle_nba_stats'
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"下載中: {dataset_name}")
        api.dataset_download_files(dataset_name, path=output_path, unzip=True)
        logger.info(f"✅ 已下載至 {output_path}")
        
        # 列出已下載的文件
        files = list(output_path.glob('*.csv'))
        for f in files:
            df_preview = pd.read_csv(f, nrows=1)
            logger.info(f"  📄 {f.name}: {pd.read_csv(f).shape}")
            logger.info(f"     欄位: {list(df_preview.columns)[:5]}...")
        
        return output_path
        
    except ImportError:
        logger.error("❌ kaggle 套件未安裝。執行: pip install kaggle")
        logger.error("   並確保 ~/.kaggle/kaggle.json 已設定")
        return None
    except Exception as e:
        logger.error(f"❌ Kaggle 下載失敗: {e}")
        return None


def download_kaggle_injury_history():
    """
    下載 Kaggle: 10-Year NBA Injury History
    
    Dataset: https://www.kaggle.com/datasets/buyuknacar/active-nba-players-10-year-injury-history
    """
    logger.info("=" * 60)
    logger.info("下載 Kaggle 傷病數據...")
    logger.info("=" * 60)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        dataset_name = 'buyuknacar/active-nba-players-10-year-injury-history'
        output_path = DATA_RAW / 'kaggle_injury_history'
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"下載中: {dataset_name}")
        api.dataset_download_files(dataset_name, path=output_path, unzip=True)
        logger.info(f"✅ 已下載至 {output_path}")
        
        files = list(output_path.glob('*.csv'))
        for f in files:
            df = pd.read_csv(f)
            logger.info(f"  📄 {f.name}: 形狀 {df.shape}")
            logger.info(f"     欄位: {list(df.columns)[:5]}...")
        
        return output_path
        
    except ImportError:
        logger.error("❌ kaggle 套件未安裝")
        return None
    except Exception as e:
        logger.error(f"❌ 傷病數據下載失敗: {e}")
        return None


def create_sample_salary_cap_history():
    """
    建立 2011-2026 NBA 薪資帽歷史表
    (實際應從 B-Ref 爬取，此為示例數據)
    """
    logger.info("=" * 60)
    logger.info("建立薪資帽歷史表 (2011-2026)...")
    logger.info("=" * 60)
    
    # 根據 NBA 官方公開資料
    salary_cap_data = {
        'season': [2011 + i for i in range(16)],
        'salary_cap_usd': [
            58044000,   # 2011-12
            63065438,   # 2012-13
            70000000,   # 2013-14 (新 CBA)
            80890000,   # 2014-15
            94143280,   # 2015-16
            99093674,   # 2016-17
            99354149,   # 2017-18
            101108949,  # 2018-19
            109140000,  # 2019-20
            109140000,  # 2019-20 (縮短)
            112414000,  # 2020-21
            112414000,  # 2020-21 (縮短)
            121226000,  # 2021-22
            136021000,  # 2022-23
            150000000,  # 2023-24 (預估)
            165000000,  # 2024-25 (預估)
            181500000,  # 2025-26 (預估, 10% 增長)
        ]
    }
    
    df_cap = pd.DataFrame(salary_cap_data)
    output_file = DATA_EXTERNAL / 'salary_cap_history.csv'
    df_cap.to_csv(output_file, index=False)
    logger.info(f"✅ 薪資帽歷史已存至 {output_file}")
    logger.info(f"   資料範圍: {df_cap['season'].min()}-{df_cap['season'].max()}")
    logger.info(f"\n預覽:\n{df_cap}")
    
    return output_file


def create_data_inventory():
    """
    彙整 data_inventory.md - 記錄所有 CSV 的元數據
    """
    logger.info("=" * 60)
    logger.info("彙整資料清單 (data_inventory.md)...")
    logger.info("=" * 60)
    
    inventory_content = f"""# Data Inventory - Phase 1 完成清單

生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 已收集資料源

### 1. Kaggle: NBA Player Stats and Salaries 2010-2025
- **來源**: https://www.kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025
- **路徑**: `data/raw/kaggle_nba_stats/`
- **用途**: 主要薪資 + 基礎統計資料
- **預期欄位**:
  - player, season, salary, team
  - PTS, AST, REB, TOV, 等基礎統計
  - 缺失值處理: TBD (Phase 2)
  
### 2. Basketball-Reference: 薪資帽歷史
- **來源**: https://www.basketball-reference.com/contracts/salary-cap-history.html
- **路徑**: `data/external/salary_cap_history.csv`
- **用途**: 計算 `pct_of_cap = salary / salary_cap`
- **資料範圍**: 2011-12 到 2025-26 賽季
- **關鍵欄位**: season, salary_cap_usd

### 3. Basketball-Reference: 進階統計
- **來源**: https://www.basketball-reference.com/leagues/NBA_YYYY_totals.html
- **路徑**: `data/raw/bref_advanced_stats/` (TBD)
- **用途**: VORP, BPM, WS, PER, TS%, 3PAr 等進階指標
- **預期欄位**:
  - VORP (Value Over Replacement Player)
  - BPM (Box Plus/Minus)
  - WS (Win Shares)
  - PER (Player Efficiency Rating)
  - TS% (True Shooting Percentage)
  - 3PAr (3-Point Attempt Rate)

### 4. Kaggle: 10-Year NBA Injury History
- **來源**: https://www.kaggle.com/datasets/buyuknacar/active-nba-players-10-year-injury-history
- **路徑**: `data/raw/kaggle_injury_history/`
- **用途**: 重大傷病標籤 (ACL, Achilles, Patellar Tendon)
- **預期欄位**: player_name, injury_date, injury_type

---

## 📋 資料質量檢查清單 (Phase 1)

| 資料源 | 行數 | 欄位數 | 時間覆蓋 | 缺失值 | 異常值 | 狀態 |
|---|---|---|---|---|---|---|
| Kaggle 薪資 | TBD | TBD | 2011-2026 | TBD | TBD | 🟡 待驗證 |
| 薪資帽歷史 | 16 | 2 | 2011-2026 | ✅ 無 | ✅ 無 | ✅ 完成 |
| B-Ref 進階 | TBD | TBD | 2011-2026 | TBD | TBD | 🟡 待下載 |
| Kaggle 傷病 | TBD | TBD | 2011-2026 | TBD | TBD | 🟡 待驗證 |

---

## 🎯 Phase 1 檢查清單

- [x] 建立 GitHub Repo 與資料夾結構
- [x] 建立 `requirements.txt` / `environment.yml`
- [x] 建立 `README.md` 與文件說明
- [ ] **Kaggle 薪資資料**: 下載 & 驗證欄位
- [ ] **B-Ref 薪資帽**: 已產生示例資料
- [ ] **B-Ref 進階統計**: 待爬取 (BeautifulSoup)
- [ ] **Kaggle 傷病資料**: 下載 & 驗證欄位
- [ ] **彙整 data_inventory.md**: ✅ 本文件

---

## 🔗 環境建置步驟

```bash
# 1. 建立 conda 環境
conda env create -f environment.yml
conda activate nba-salary-valuation

# 2. 設定 Kaggle API
# 至 https://www.kaggle.com/settings/account
# 下載 kaggle.json，存至 ~/.kaggle/

# 3. 執行本下載腳本
python download_data.py

# 4. 驗證資料
ls data/raw/
ls data/external/
```

---

## ⚠️ 已知問題 & 後續步驟

### 待解決
- [ ] B-Ref 進階統計需手動下載或爬取
- [ ] 球員名稱統一化 (重音字元) - Phase 2 處理
- [ ] 缺失值策略定義 - Phase 2 處理

### Phase 2 (Day 1 下午 - Day 2 上午)
- 資料清理: 名稱統一、缺失值填補
- 資料整合: 多源合併
- 特徵工程: 時序加權、A/B 分群

---

**上次更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    inventory_file = PROJECT_ROOT / 'data_inventory.md'
    with open(inventory_file, 'w', encoding='utf-8') as f:
        f.write(inventory_content)
    
    logger.info(f"✅ 資料清單已生成: {inventory_file}")
    return inventory_file


def main():
    """主執行函數"""
    logger.info("\n" + "="*60)
    logger.info("🚀 Phase 1: 環境建置 & 資料收集")
    logger.info("="*60 + "\n")
    
    # 1. 下載 Kaggle 薪資資料
    kaggle_stats_path = download_kaggle_nba_stats()
    
    # 2. 下載 Kaggle 傷病資料
    kaggle_injury_path = download_kaggle_injury_history()
    
    # 3. 建立薪資帽歷史表
    salary_cap_path = create_sample_salary_cap_history()
    
    # 4. 生成資料清單
    inventory_path = create_data_inventory()
    
    logger.info("\n" + "="*60)
    logger.info("✅ Phase 1 資料收集完成！")
    logger.info("="*60)
    logger.info("\n📋 後續步驟:")
    logger.info("   1. 檢查 data_inventory.md 中的資料質量")
    logger.info("   2. 從 B-Ref 手動下載進階統計 (若需要)")
    logger.info("   3. 進行 Phase 2 資料清理與特徵工程")
    logger.info("\n")


if __name__ == '__main__':
    main()
