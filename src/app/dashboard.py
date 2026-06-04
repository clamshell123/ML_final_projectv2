import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import shap
import matplotlib.pyplot as plt
from supabase import create_client, Client
from dotenv import load_dotenv

# ==========================================
# 0. 頁面基本設定與常數定義
# ==========================================
st.set_page_config(
    page_title="NBA 跨時空 Moneyball 薪資模擬器",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 各年度薪資帽 (美金)
SALARY_CAPS = {
    2023: 136021000,
    2024: 140588000,
    2025: 155122000,
    2026: 170814000  # 依據最新 CBA 預估
}

# 現代 NBA 30 支球隊縮寫
NBA_TEAMS = sorted([
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
    "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
])

# ==========================================
# 1. 初始化與快取讀取 (Supabase & Models)
# ==========================================
@st.cache_resource
def init_supabase() -> Client:
    # 嘗試從環境變數或 Streamlit secrets 讀取
    load_dotenv()
    url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
    if not url or not key:
        st.error("請在 .env 或 st.secrets 設定 SUPABASE_URL 與 SUPABASE_KEY")
        st.stop()
    return create_client(url, key)

@st.cache_resource
def load_models():
    # 取得 dashboard.py 當前所在的資料夾路徑 (例如: .../src/app)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 假設 1：dashboard.py 在 src/app/ 底下，我們要往上一層到 src/，再進 models/
    models_dir = os.path.abspath(os.path.join(current_dir, '..', 'models'))
    
    # 假設 2 (防呆)：如果你哪天把 dashboard.py 移到了專案根目錄
    if not os.path.exists(models_dir):
        models_dir = os.path.abspath(os.path.join(current_dir, 'src', 'models'))
        
    pricing_path = os.path.join(models_dir, 'pricing_model.pkl')
    explainer_path = os.path.join(models_dir, 'shap_explainer.pkl')
    
    try:
        with open(pricing_path, 'rb') as f:
            pricing_model = pickle.load(f)
        with open(explainer_path, 'rb') as f:
            shap_explainer = pickle.load(f)
        return pricing_model, shap_explainer
    except Exception as e:
        # 把找錯的路徑印在網頁上，方便 Debug
        st.error(f"模型載入失敗！\n系統嘗試尋找的路徑為: `{pricing_path}`\n錯誤訊息: {e}")
        st.stop()

# ==========================================
# 2. 資料獲取函式
# ==========================================
# ==========================================
# 2. 資料獲取函式 (安全連線版)
# ==========================================
@st.cache_data(ttl=3600)
def get_available_players():
    # 每次呼叫前，安全獲取快取中的 supabase 客戶端
    client = init_supabase()
    response = client.table('historical_predictions').select('player_name').execute()
    players = list(set([row['player_name'].title() for row in response.data]))
    return sorted(players)

@st.cache_data(ttl=3600)
def get_player_years(player_name: str):
    client = init_supabase()
    response = client.table('historical_predictions')\
        .select('stat_year')\
        .eq('player_name', player_name.lower())\
        .execute()
    years = [row['stat_year'] for row in response.data]
    return sorted(years, reverse=True)

@st.cache_data(ttl=3600)
def get_player_data(player_name: str, year: int):
    client = init_supabase()
    response = client.table('historical_predictions')\
        .select('*')\
        .eq('player_name', player_name.lower())\
        .eq('stat_year', year)\
        .execute()
    if response.data:
        return response.data[0]
    return None

# ==========================================
# 3. 側邊欄：總管沙盤推演參數設定
# ==========================================
st.sidebar.title("⚙️ 模擬器參數設定")

available_players = get_available_players()
if not available_players:
    st.warning("資料庫中尚無球員資料，請先執行 batch_inference.py")
    st.stop()

selected_player = st.sidebar.selectbox("1️⃣ 球員名稱", available_players)

available_years = get_player_years(selected_player)
selected_skill_year = st.sidebar.selectbox("2️⃣ 實力年份 (客觀實力基準)", available_years)

# 抓取該球員當年的資料與「特徵 DNA」
player_data = get_player_data(selected_player, selected_skill_year)
if not player_data:
    st.error("無法取得該球員的特徵資料！")
    st.stop()

original_team = player_data.get('team', 'Unknown')
raw_features = player_data['key_features'].get('model_features', {})

if not raw_features:
    st.error("此筆資料缺乏 'model_features' (特徵 DNA)，請確保 batch_inference.py 有正確上傳。")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("未來的市場環境設定")

destination_team = st.sidebar.selectbox(
    "3️⃣ 目標球隊", 
    NBA_TEAMS, 
    index=NBA_TEAMS.index(original_team) if original_team in NBA_TEAMS else 0
)

is_retained = st.sidebar.checkbox(
    "4️⃣ 是否以鳥權續約？(母隊加成)", 
    value=(destination_team == original_team)
)

estimation_year = st.sidebar.selectbox(
    "5️⃣ 預估薪資帽年份", 
    list(SALARY_CAPS.keys()), 
    index=list(SALARY_CAPS.keys()).index(2026)  # 預設為最新年份
)

# ==========================================
# 4. 即時推論邏輯 (Dynamic Inference)
# ==========================================
# 把 JSON 特徵轉回 DataFrame
X_infer = pd.DataFrame([raw_features])
X_infer.columns = X_infer.columns.astype(str) # 強制轉字串防報錯

# 🏀 [動態修改特徵]
# 1. 洗掉所有舊的球隊特徵
team_cols = [c for c in X_infer.columns if c.startswith('Team_')]
for c in team_cols:
    X_infer[c] = 0

# 2. 注入新的目標球隊與鳥權狀態
target_team_col = f"Team_{destination_team}"
if target_team_col in X_infer.columns:
    X_infer[target_team_col] = 1

X_infer['is_retained'] = 1 if is_retained else 0

# 🚀 [現場推論]
# 🚀 [現場推論]
pricing_model, shap_explainer = load_models()

# 🚨 [修復重點]：強制將 X_infer 的欄位順序對齊模型訓練時的標準順序
expected_cols = pricing_model.feature_names_in_

# 防呆機制：確保所有模型需要的欄位都在，缺少的補 0
for col in expected_cols:
    if col not in X_infer.columns:
        X_infer[col] = 0

# 強制對齊欄位順序
X_infer = X_infer[expected_cols]

# 執行預測
predicted_cap_pct = pricing_model.predict(X_infer)[0]
estimated_salary = predicted_cap_pct * SALARY_CAPS[estimation_year]

# ==========================================
# 5. 主畫面：結果展示與視覺化
# ==========================================
st.title("🏀 NBA 跨時空 Moneyball 薪資模擬器")
st.markdown("透過 XGBoost 與 SHAP 剝離歷史市場雜訊，將球員的「純粹籃球實力」無縫轉換至現代或未來的薪資體系。")

st.markdown(f"### 📊 {selected_player} ({selected_skill_year} 實力) ➡️ {destination_team} 身價解析")

# 🌟 [新增區塊]：基礎賽季數據概覽
# 從剛剛讀取下來的 raw_features 裡面直接抽出傳統數據 (找不到則預設為 0)
pts = raw_features.get('PTS_reg', 0)
trb = raw_features.get('TRB_reg', 0)
ast = raw_features.get('AST_reg', 0)
mp = raw_features.get('MP_reg', 0)
ts_pct = raw_features.get('TS%_reg', 0) * 100  # 轉為百分比
ws = raw_features.get('WS_reg', 0)

# 使用 st.columns 排版，並用 st.metric 呈現漂亮的數字卡片
st.markdown("##### ⛹️‍♂️ 該年度例行賽表現 (時間加權後基準)")
metric_cols = st.columns(6)
metric_cols[0].metric("得分 (PTS)", f"{pts:.1f}")
metric_cols[1].metric("籃板 (TRB)", f"{trb:.1f}")
metric_cols[2].metric("助攻 (AST)", f"{ast:.1f}")
metric_cols[3].metric("上場時間 (MP)", f"{mp:.1f}")
metric_cols[4].metric("真實命中率", f"{ts_pct:.1f}%")
metric_cols[5].metric("勝利貢獻 (WS)", f"{ws:.1f}")

st.markdown("---")

# 接著是原本的 AI 估值區塊
col1, col2 = st.columns(2)
with col1:
    st.info("🎯 AI 判定：目標市場薪資帽佔比")
    st.metric(label="預估佔比 (Cap Pct)", value=f"{predicted_cap_pct * 100:.2f}%")

with col2:
    st.success(f"💰 跨時空換算：{estimation_year} 賽季年薪")
    st.metric(
        label=f"在 ${SALARY_CAPS[estimation_year]:,.0f} 薪資帽下的絕對薪資", 
        value=f"${estimated_salary:,.0f}"
    )

st.markdown("---")
st.subheader("🧬 薪資估值拆解 (即時 SHAP Waterfall Analysis)")

with st.spinner("正在生成 SHAP 瀑布圖..."):
    # 現場生成 SHAP 解釋
    shap_values_obj = shap_explainer(X_infer)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # 為了讓暗色主題好看，設定一些 matplotlib 參數
    plt.style.use('dark_background')
    
    # 畫出瀑布圖
    shap.plots.waterfall(shap_values_obj[0], max_display=10, show=False)
    
    plt.title(f"Dynamic Salary Valuation Breakdown: {selected_player}", fontsize=14, pad=20)
    plt.tight_layout()
    
    st.pyplot(fig)
    
st.caption("說明：紅色柱狀體代表推升身價的正向特徵，藍色則為扣分項目。最下方的 f(x) 為模型最終輸出的薪資佔比預測值。")