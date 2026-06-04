import streamlit as st
import pandas as pd
import os
from sqlalchemy import create_engine

# ── 頁面基本設定 ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBA 跨時空薪資模擬器", page_icon="🏀", layout="wide")

st.title("🏀 NBA 跨時空 Moneyball 薪資模擬器")
st.markdown("透過 XGBoost 與 SHAP 剝離歷史市場雜訊，將球員的「純粹籃球實力」無縫轉換至現代或未來的薪資體系。")
st.divider()

# ── 核心邏輯：動態讀取歷史薪資帽與 CBA 10% 推算 ─────────────────────────────
@st.cache_data
def get_salary_cap(target_year):
    # 1. 取得當前檔案 (dashboard.py) 的絕對路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 2. 回溯到專案根目錄 (從 src/app 往上兩層)
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    csv_path = os.path.join(project_root, "data", "external", "salary_cap_history.csv")
    
    # 備用/基準薪資帽字典 (Fallback)
    base_caps = {2022: 123655000, 2023: 136021000, 2024: 141000000}
    
    # 嘗試從 CSV 抓取真實歷史數據
    if os.path.exists(csv_path):
        try:
            df_cap = pd.read_csv(csv_path)
            # 尋找對應年份的資料 (假設 CSV 欄位名稱包含 year / target_year / season 等字眼，這裡做彈性比對)
            # 為了確保相容性，我們比對第一欄(年份)並抓取第二欄(薪資帽)
            match = df_cap[df_cap.iloc[:, 0] == target_year]
            if not match.empty:
                # 成功從 CSV 找到歷史真實薪資帽
                return float(match.iloc[0, 1])
        except Exception as e:
            st.sidebar.warning(f"⚠️ 讀取薪資帽檔案時發生小錯誤，將使用系統預設值。({e})")

    # 若 CSV 找不到、裡面沒該年份資料，或是未來的年份，套用以下邏輯
    if target_year in base_caps:
        return base_caps[target_year]
    elif target_year > 2024:
        years_ahead = target_year - 2024
        return base_caps[2024] * (1.10 ** years_ahead)
    else:
        return 100000000 

# ── 讀取資料（SQLAlchemy 替換 Supabase API）────────────────────────────────
@st.cache_data(ttl=600)
def load_prediction_data() -> pd.DataFrame:
    """從 Supabase (PostgreSQL) 讀取所有球員的預測結果"""
    db_url = st.secrets.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    
    if not db_url:
        st.error("找不到資料庫連線字串 (DATABASE_URL)。請確認 Secrets 已經設定。")
        st.stop()

    engine = create_engine(db_url)
    
    # 直接用 SQL 撈出我們需要的三個欄位
    query = """
        SELECT player_name, stat_year, pure_skill_pct
        FROM historical_predictions
    """
    df = pd.read_sql(query, engine)
    return df

try:
    with st.spinner("正在從資料庫載入歷史特徵庫..."):
        df_preds = load_prediction_data()
except Exception as e:
    st.error(f"載入資料失敗，請檢查資料庫連線或密碼是否正確: {e}")
    st.stop()

if df_preds.empty:
    st.warning("目前資料庫中沒有球員資料。")
    st.stop()

# ── 側邊欄篩選器 ──────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 模擬器參數設定")

unique_players = sorted(df_preds["player_name"].unique().tolist())
selected_player = st.sidebar.selectbox("1️⃣ 球員名稱", unique_players)

# 根據選定的球員，動態過濾出他擁有的數據年份
player_data = df_preds[df_preds["player_name"] == selected_player]
available_years = sorted(player_data["stat_year"].unique().tolist(), reverse=True)

selected_year = st.sidebar.selectbox("2️⃣ 球員實力年分", available_years)

st.sidebar.markdown("---")
target_eras = list(range(2024, 2031))
selected_era = st.sidebar.selectbox("3️⃣ 預估年代 (Target Era)", target_eras)

st.sidebar.markdown("---")
analyze_button = st.sidebar.button("🚀 進行跨時空估值", use_container_width=True)

# ── 處理估值邏輯與視覺化 ───────────────────────────────────────────────────
if analyze_button:
    target_cap = get_salary_cap(selected_era)
    
    # 從剛剛撈好的 DataFrame 抓取該球員該年份的實力佔比
    specific_data = player_data[player_data["stat_year"] == selected_year]
    
    if not specific_data.empty:
        pure_skill_pct = specific_data.iloc[0]["pure_skill_pct"]
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
        st.error("系統異常：找不到對應的資料點。")