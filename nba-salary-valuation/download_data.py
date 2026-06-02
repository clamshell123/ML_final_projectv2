import os
import sys
import logging
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# ----------------------------------------------------------------------
# 1. 環境建置 & 基本路徑設定
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 設定路徑結構
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent  # 專案根目錄位置

DATA_DIR = PROJECT_ROOT / 'data'
DATA_RAW = DATA_DIR / 'raw'
DATA_EXTERNAL = DATA_DIR / 'external'

# 確保基礎資料夾存在
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. Kaggle 下載與驗證工具模組
# ----------------------------------------------------------------------
def init_kaggle_api():
    """驗證並初始化 Kaggle API 連線"""
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        logger.error("❌ kaggle 套件未安裝或未配置憑證。執行: pip install kaggle")
        logger.error("   並確保 ~/.kaggle/kaggle.json 已設定正確。")
        logger.error(f"   錯誤細節: {e}")
        return None

def download_kaggle_dataset(api, dataset_slug, target_folder):
    """通用 Kaggle 下載與解壓縮核心邏輯"""
    output_path = DATA_RAW / target_folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        api.dataset_download_files(dataset_slug, path=output_path, unzip=True)
        logger.info(f"✅ 成功下載並解壓至: {output_path}")
        
        # 顯示下載檔案預覽
        for csv_file in output_path.glob("*.csv"):
            df = pd.read_csv(csv_file, nrows=3)
            logger.info(f"   📄 尋獲檔案: {csv_file.name} | 欄位形狀: {df.shape[1]} 欄")
        return output_path
    except Exception as e:
        logger.error(f"❌ 下載資料集失敗 ({dataset_slug}): {e}")
        return None

# ----------------------------------------------------------------------
# 3. 建立薪資帽歷史表 (修正長度對齊與年份位移問題)
# ----------------------------------------------------------------------
def create_sample_salary_cap_history():
    """建立包含 17 個項目的完整且對齊的薪資上限歷史對照表 (2011-2027)"""
    logger.info("============================================================")
    logger.info("建立薪資帽歷史表 (2011-2027)...")
    logger.info("============================================================")
    
    salary_cap_data = {
        'season': [
            2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 
            2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027
        ],
        'salary_cap_usd': [
            58044000,   # 2011-12
            63065438,   # 2012-13
            70000000,   # 2013-14 (新 CBA 生效)
            80890000,   # 2014-15
            94143280,   # 2015-16
            99093674,   # 2016-17
            99354149,   # 2017-18
            101108949,  # 2018-19
            109140000,  # 2019-20
            109140000,  # 2019-20 (縮短賽季調整)
            112414000,  # 2020-21
            112414000,  # 2020-21 (縮短賽季調整)
            121226000,  # 2021-22
            136021000,  # 2022-23
            150000000,  # 2023-24
            165000000,  # 2024-25
            181500000   # 2025-26 (預估, 10% 增長)
        ]
    }
    
    df_cap = pd.DataFrame(salary_cap_data)
    output_file = DATA_EXTERNAL / 'salary_cap_history.csv'
    df_cap.to_csv(output_file, index=False)
    
    logger.info(f"✅ 薪資帽歷史已存至 {output_file}")
    logger.info(f"   資料範圍: 2011-2027")
    return output_file

# ----------------------------------------------------------------------
# 4. 彙整與清單生成邏輯
# ----------------------------------------------------------------------
def create_data_inventory():
    """自動掃描 raw / external 資料夾並產出更新後的項目清單日誌"""
    inventory_path = PROJECT_ROOT / 'data_inventory.md'
    logger.info("============================================================")
    logger.info("彙整資料清單 (data_inventory.md)...")
    logger.info("============================================================")
    
    with open(inventory_path, 'w', encoding='utf-8') as f:
        f.write("# 📋 NBA Salary Valuation - Data Inventory\n\n")
        f.write("此文件由系統自動更新，記錄專案內目前緩存的原始數據狀態。\n\n")
        f.write("## 📂 Raw Datasets\n")
        
        for folder in DATA_RAW.iterdir():
            if folder.is_dir():
                f.write(f"### 📁 `{folder.name}`\n")
                for csv_file in folder.glob("*.csv"):
                    f.write(f"- 📄 文件名: `{csv_file.name}`\n")
        
        f.write("\n## 📂 External Benchmarks\n")
        for csv_file in DATA_EXTERNAL.glob("*.csv"):
            f.write(f"- 📄 文件名: `{csv_file.name}`\n")
            
    logger.info(f"✅ 資料清單已生成: {inventory_path}")
    return inventory_path

# ----------------------------------------------------------------------
# 5. 主程式入口執行管線
# ----------------------------------------------------------------------
def main():
    logger.info("============================================================")
    logger.info("🚀 Phase 1: 環境建置 & 資料收集")
    logger.info("============================================================")
    
    api = init_kaggle_api()
    if not api:
        logger.error("❌ 找不到可用的 Kaggle 認證文件，中止收集管線。")
        sys.exit(1)
        
    # 1. 下載 Kaggle 薪資與常規賽數據
    logger.info("============================================================")
    logger.info("下載 Kaggle 薪資數據...")
    logger.info("============================================================")
    download_kaggle_dataset(api, "ratin21/nba-player-stats-and-salaries-2010-2025", "kaggle_nba_stats")
    
    # 2. 下載 Kaggle 傷病數據
    logger.info("============================================================")
    logger.info("下載 Kaggle 傷病數據...")
    logger.info("============================================================")
    download_kaggle_dataset(api, "buyuknacar/active-nba-players-10-year-injury-history", "kaggle_injury_history")
    
    # 3. 下載 Kaggle NBA 全員生物特徵庫 (取代舊版特徵受限的 Combine 體測數據)
    logger.info("============================================================")
    logger.info("下載 Kaggle NBA All-Player Biometrics 歷史數據...")
    logger.info("============================================================")
    download_kaggle_dataset(api, "justinas/nba-players-data", "kaggle_biometrics")
    
    # 4. 建立薪資帽歷史表
    create_sample_salary_cap_history()
    
    # 5. 生成資料清單
    create_data_inventory()
    
    logger.info("============================================================")
    logger.info("✅ Phase 1 資料收集完成！")
    logger.info("============================================================")

if __name__ == '__main__':
    main()