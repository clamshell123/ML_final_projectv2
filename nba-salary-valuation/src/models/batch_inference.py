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
    
    # 切分特徵矩陣與識別碼
    X_database = df.drop(columns=['Player', 'year', 'Cap_Pct', 'YRS'], errors='ignore')
    ids_database = df[['Player', 'year']]
    
    success_count = 0
    records_to_upload = []

    # 3. 逐筆進行推論 (若資料量極大，實務上可改寫為矩陣運算)
    for index, row in ids_database.iterrows():
        player_name = row['Player']
        stat_year = int(row['year'])
        
        # 呼叫我們的推論引擎
        result = engine.predict_player_value(
            player_name=player_name, 
            target_year=stat_year, 
            X_database=X_database, 
            ids_database=ids_database
        )
        
        if "error" in result:
            print(f"⚠️ 略過 {player_name} ({stat_year}): {result['error']}")
            continue
            
        # 4. 構建 Supabase 需要的 Payload 格式
        # 我們只提取純實力佔比，因為這是跨時空估值的核心
        pure_skill_pct = result['valuation']['pure_skill_pct']
        
        # 建立複合主鍵，避免重複寫入
        player_id = f"{player_name.replace(' ', '_').lower()}_{stat_year}"
        
        record = {
            "player_id": player_id,
            "player_name": player_name,
            "stat_year": stat_year,
            "pure_skill_pct": pure_skill_pct,
            # 這裡可以儲存重要的 SHAP 特徵，先放個示意結構
            "key_features": {"base_pct": result['valuation']['base_pct']} 
        }
        records_to_upload.append(record)
        
        # 每 100 筆批次上傳一次，避免 Payload 過大
        if len(records_to_upload) >= 100:
            try:
                data, count = supabase.table('historical_predictions').upsert(records_to_upload).execute()
                success_count += len(records_to_upload)
                print(f"✅ 已成功 Upsert {success_count} 筆資料...")
                records_to_upload = [] # 清空暫存區
            except Exception as e:
                print(f"❌ 上傳失敗: {e}")
                
    # 處理剩餘未滿 100 筆的尾數
    if records_to_upload:
        try:
            data, count = supabase.table('historical_predictions').upsert(records_to_upload).execute()
            success_count += len(records_to_upload)
            print(f"✅ 已成功 Upsert 最後一批資料，總計 {success_count} 筆！")
        except Exception as e:
            print(f"❌ 最後一批上傳失敗: {e}")

if __name__ == "__main__":
    # 先用我們現有的 processed 資料測試管線
    target_csv = os.path.join(BASE_DIR, 'data', 'processed', 'featured_nba_data.csv')
    process_and_upload_batch(target_csv)