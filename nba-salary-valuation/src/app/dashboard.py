import streamlit as st
import os
import sys
import pandas as pd
from supabase import create_client, Client

# --- 將專案根目錄加入路徑，這樣才能載入我們的 models 模組 ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.models.inference import SalaryValuationEngine

# --- 頁面基本設定 ---
st.set_page_config(
    page_title="NBA 跨時空薪資模擬器", 
    page_icon="🏀", 
    layout="wide"
)

st.title("🏀 NBA 跨時空 Moneyball 薪資模擬器")
st.markdown("透過 XGBoost 與 SHAP 剝離歷史市場雜訊，將球員的「純粹籃球實力」無縫轉換至現代或未來的薪資體系。")
st.divider()

# --- 核心邏輯：載入 Supabase 與 模型引擎 (使用快取避免重複載入) ---
@st.cache_resource
def init_system():
    # 這裡改成用 Streamlit 的 Secret 管理金鑰 (稍後在網頁上設定)
    # 本地測試時如果有 .env 依然會自動讀取
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    
    supabase_client: Client = create_client(url, key)
    
    # 初始化估值引擎
    models_dir = os.path.join(BASE_DIR, 'src', 'models')
    val_engine = SalaryValuationEngine(models_dir=models_dir)
    
    return supabase_client, val_engine

try:
    supabase, engine = init_system()
except Exception as e:
    st.error(f"系統初始化失敗 (金鑰或模型遺失)：{e}")
    st.stop()

def get_salary_cap(target_year):
    base_caps = {2022: 123655000, 2023: 136021000, 2024: 141000000}
    if target_year in base_caps:
        return base_caps[target_year]
    elif target_year > 2024:
        years_ahead = target_year - 2024
        return base_caps[2024] * (1.10 ** years_ahead)
    else:
        return 100000000 

# --- 從 Supabase 獲取可用的球員名單 ---
@st.cache_data(ttl=600)
def fetch_available_players():
    try:
        response = supabase.table('historical_predictions').select("player_name, stat_year").limit(50000).execute()
        return response.data
    except Exception as e:
        st.error(f"無法連線至 Supabase：{e}")
        return []

players_data = fetch_available_players()

# --- 建立側邊欄/輸入介面 ---
st.sidebar.header("⚙️ 模擬器參數設定")

if not players_data:
    st.sidebar.warning("目前資料庫中沒有可用的球員資料。")
else:
    unique_players = sorted(list(set([p['player_name'] for p in players_data])))
    selected_player = st.sidebar.selectbox("1️⃣ 球員名稱", unique_players)
    
    raw_years = [p['stat_year'] for p in players_data if p['player_name'] == selected_player]
    available_years = sorted(list(set(raw_years)), reverse=True)
    selected_year = st.sidebar.selectbox("2️⃣ 球員實力年分", available_years)
    
    st.sidebar.markdown("---")
    target_eras = list(range(2024, 2031))
    selected_era = st.sidebar.selectbox("3️⃣ 預估年代 (Target Era)", target_eras)
    
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("🚀 進行跨時空估值", use_container_width=True)

    # --- 處理估值邏輯與視覺化 ---
    if analyze_button:
        target_cap = get_salary_cap(selected_era)
        
        with st.spinner(f"正在分析 {selected_player} ({selected_year}) 的實力特徵..."):
            try:
                # 這裡直接去 Supabase 抓取預先算好的 pure_skill_pct
                # 因為我們已經用 batch_inference 上傳過了！
                player_id = f"{selected_player.replace(' ', '_').lower()}_{selected_year}"
                response = supabase.table('historical_predictions').select('pure_skill_pct').eq('player_id', player_id).execute()
                
                if response.data:
                    pure_skill_pct = response.data[0]['pure_skill_pct']
                    projected_salary = pure_skill_pct * target_cap
                    
                    st.header(f"📊 {selected_player} ({selected_year} 實力) ➡️ {selected_era} 年代身價解析")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info("🎯 **AI 判定：純粹籃球實力佔比**")
                        st.metric(label="剔除市場雜訊後的薪資帽佔比", value=f"{pure_skill_pct:.1%}")
                        
                    with col2:
                        st.success(f"💰 **跨時空換算：{selected_era} 賽季年薪**")
                        st.metric(
                            label=f"在 ${target_cap/1000000:.1f}M 薪資帽下的絕對薪資", 
                            value=f"${projected_salary:,.0f}",
                            delta=f"套用 CBA 10% 規則" if selected_era > 2024 else None
                        )
                    
                    st.divider()
                    st.markdown(f"""
                    ### 📝 GM 決策洞察 (Decision Insight)
                    根據系統的 **A/B 群組剝離分析**，{selected_player} 絕對值得佔據球隊 **{pure_skill_pct:.1%}** 的薪資空間。
                    球隊應該為他準備一份年均薪大約為 **${projected_salary:,.0f}** 的合約，這才是符合 Moneyball 邏輯的定價。
                    """)
                else:
                    st.error("資料庫中找不到該球員的詳細預測資料。")
            
            except Exception as e:
                st.error(f"發生錯誤：{e}")