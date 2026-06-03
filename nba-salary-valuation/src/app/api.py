import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. 初始化環境變數與 Supabase 客戶端
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
load_dotenv(os.path.join(BASE_DIR, '.env'))

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    raise RuntimeError("找不到 Supabase 環境變數，請確認 .env 檔案。")

supabase: Client = create_client(url, key)

app = FastAPI(
    title="🏀 NBA 跨時空薪資估值 API (Supabase 雲端版)", 
    version="3.0"
)

# 定義前端傳入的 Request Body 格式
class ValuationRequest(BaseModel):
    player_name: str
    target_year: int

# 2. 核心預測端點：直接向 Supabase 查詢預先算好的純實力
@app.post("/predict")
def predict_salary(request: ValuationRequest):
    try:
        # 組合我們在 batch_inference 定義的 player_id 主鍵
        player_id = f"{request.player_name.replace(' ', '_').lower()}_{request.target_year}"
        
        # 向 Supabase 發起精準查詢
        response = supabase.table('historical_predictions').select("*").eq("player_id", player_id).execute()
        
        data = response.data
        if not data:
            raise HTTPException(status_code=404, detail=f"在雲端資料庫找不到 {request.player_name} ({request.target_year}) 的資料。")
            
        # 取出該筆球員資料
        player_data = data[0]
        
        # 回傳給前端
        return {
            "player": player_data["player_name"],
            "year": player_data["stat_year"],
            "valuation": {
                "pure_skill_pct": player_data["pure_skill_pct"],
                "key_features": player_data.get("key_features", {})
            }
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"資料庫查詢錯誤: {str(e)}")

# 3. (新增) 取得所有可用球員清單，供 Streamlit 下拉選單使用
@app.get("/players")
def get_available_players():
    try:
        # 只撈取姓名與年份，讓前端可以建立動態選單
        response = supabase.table('historical_predictions').select("player_name, stat_year").execute()
        return {"players": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)