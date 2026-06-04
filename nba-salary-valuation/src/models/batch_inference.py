import os
import sys
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# 將專案的 src 目錄加入系統路徑
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from models.inference import SalaryValuationEngine

# 1. 載入環境變數與初始化 Supabase 客戶端
load_dotenv(os.path.join(BASE_DIR, '.env'))
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise ValueError("找不到 Supabase URL 或 Key，請確認 .env 檔案已正確設定。")

supabase: Client = create_client(url, key)

# 2. 初始化估值引擎
MODELS_DIR = os.path.join(SRC_DIR, 'models')
engine = SalaryValuationEngine(models_dir=MODELS_DIR)

def process_and_upload_batch(csv_path: str):
    """
    讀取特徵資料庫，進行批量推論，並 Upsert 至 Supabase
    """
    print(f"📦 開始處理批量資料: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # ==========================================
    # 1. 特徵對齊：補齊歷史資料沒有的欄位
    # ==========================================
    if 'GS_reg' not in df.columns:
        df['GS_reg'] = 0
        
    if 'GS_playoff' not in df.columns:
        df['GS_playoff'] = 0
    
    # 2. 移除不要的標籤
    X_database = df.drop(columns=['Player', 'year', 'Cap_Pct', 'YRS'], errors='ignore')
    ids_database = df[['Player', 'year']]
    
    # ==========================================
    # 3. [終極修正] 強制對齊欄位順序 (Fix for Order Mismatch)
    # 從 XGBoost 模型中抽出當初訓練的欄位藍圖，強制洗牌
    # ==========================================
    expected_cols = engine.pricing_model.feature_names_in_
    X_database = X_database[expected_cols] 
    
    success_count = 0
    records_to_upload = []

    # 4. 逐筆進行推論
    for index, row in ids_database.iterrows():
        player_name = row['Player']
        stat_year = int(row['year'])
        
        # 呼叫推論引擎
        result = engine.predict_player_value(
            player_name=player_name, 
            target_year=stat_year, 
            X_database=X_database, 
            ids_database=ids_database
        )
        
        if "error" in result:
            print(f"⚠️ 略過 {player_name} ({stat_year}): {result['error']}")
            continue
            
        # 構建 Supabase 需要的 Payload
        pure_skill_pct = result['valuation']['pure_skill_pct']
        player_id = f"{player_name.replace(' ', '_').lower()}_{stat_year}"
        
        record = {
            "player_id": player_id,
            "player_name": player_name,
            "stat_year": stat_year,
            "pure_skill_pct": pure_skill_pct,
            "key_features": {"base_pct": result['valuation']['base_pct']} 
        }
        records_to_upload.append(record)
        
        # 每 100 筆批次上傳
        if len(records_to_upload) >= 100:
            try:
                data, count = supabase.table('historical_predictions').upsert(records_to_upload).execute()
                success_count += len(records_to_upload)
                print(f"✅ 已成功 Upsert {success_count} 筆資料...")
                records_to_upload = [] 
            except Exception as e:
                print(f"❌ 上傳失敗: {e}")
                
    # 處理尾數
    if records_to_upload:
        try:
            data, count = supabase.table('historical_predictions').upsert(records_to_upload).execute()
            success_count += len(records_to_upload)
            print(f"✅ 已成功 Upsert 最後一批資料，總計 {success_count} 筆！")
        except Exception as e:
            print(f"❌ 最後一批上傳失敗: {e}")

if __name__ == "__main__":
    target_csv = os.path.join(BASE_DIR, 'data', 'processed', 'historical_legends_features.csv')
    process_and_upload_batch(target_csv)