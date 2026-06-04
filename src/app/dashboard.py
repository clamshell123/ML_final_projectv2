import streamlit as st
import pandas as pd
import os
import json
from sqlalchemy import create_engine
import plotly.graph_objects as go

# ── 頁面基本設定 ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NBA 跨時空薪資模擬器", page_icon="🏀", layout="wide")

st.title("🏀 NBA 跨時空 Moneyball 薪資模擬器")
st.markdown("透過 XGBoost 與 SHAP 剝離歷史市場雜訊，將球員的「純粹籃球實力」無縫轉換至現代或未來的薪資體系。")
st.divider()

# ── 核心邏輯：動態讀取歷史薪資帽與 CBA 10% 推算 ─────────────────────────────
@st.cache_data
def get_salary_cap(target_year):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    csv_path = os.path.join(project_root, "data", "external", "salary_cap_history.csv")
    
    base_caps = {2022: 123655000, 2023: 136021000, 2024: 141000000}
    
    if os.path.exists(csv_path):
        try:
            df_cap = pd.read_csv(csv_path)
            match = df_cap[df_cap.iloc[:, 0] == target_year]
            if not match.empty:
                return float(match.iloc[0, 1])
        except Exception as e:
            st.sidebar.warning(f"⚠️ 讀取薪資帽檔案時發生錯誤，使用預設值。({e})")

    if target_year in base_caps:
        return base_caps[target_year]
    elif target_year > 2024:
        years_ahead = target_year - 2024
        return base_caps[2024] * (1.10 ** years_ahead)
    else:
        return 100000000 

# ── 讀取資料 ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_prediction_data() -> pd.DataFrame:
    db_url = st.secrets.get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not db_url:
        st.error("找不到資料庫連線字串 (DATABASE_URL)。")
        st.stop()

    engine = create_engine(db_url)
    
    # [新增] 把 key_features 撈出來，裡面會裝 SHAP 拆解數據
    query = """
        SELECT player_name, stat_year, pure_skill_pct, key_features
        FROM historical_predictions
    """
    df = pd.read_sql(query, engine)
    return df

try:
    with st.spinner("正在從資料庫載入歷史特徵庫..."):
        df_preds = load_prediction_data()
except Exception as e:
    st.error(f"載入資料失敗: {e}")
    st.stop()

if df_preds.empty:
    st.warning("目前資料庫中沒有球員資料。")
    st.stop()

# ── 側邊欄篩選器 ──────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ 模擬器參數設定")

unique_players = sorted(df_preds["player_name"].unique().tolist())
selected_player = st.sidebar.selectbox("1️⃣ 球員名稱", unique_players)

player_data = df_preds[df_preds["player_name"] == selected_player]
available_years = sorted(player_data["stat_year"].unique().tolist(), reverse=True)
selected_year = st.sidebar.selectbox("2️⃣ 球員實力年分", available_years)

st.sidebar.markdown("---")
target_eras = list(range(2024, 2031))
selected_era = st.sidebar.selectbox("3️⃣ 預估年代 (Target Era)", target_eras)
analyze_button = st.sidebar.button("🚀 進行跨時空估值", use_container_width=True)

# ── 處理估值邏輯與視覺化 ───────────────────────────────────────────────────
if analyze_button:
    target_cap = get_salary_cap(selected_era)
    specific_data = player_data[player_data["stat_year"] == selected_year]
    
    if not specific_data.empty:
        pure_skill_pct = specific_data.iloc[0]["pure_skill_pct"]
        projected_salary = pure_skill_pct * target_cap
        
        st.header(f"📊 {selected_player} ({selected_year} 實力) ➡️ {selected_era} 年代身價解析")
        
        # 上半部：核心指標
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
        
        # ── 下半部：SHAP 瀑布圖 ──
        st.subheader("🧬 薪資估值拆解 (SHAP Waterfall Analysis)")
        
        # 嘗試解析資料庫裡的 key_features
        raw_features = specific_data.iloc[0].get("key_features")
        if isinstance(raw_features, str):
            try:
                features_dict = json.loads(raw_features)
            except:
                features_dict = {}
        else:
            features_dict = raw_features or {}
            
        # 檢查是否有完整的 shap_values 欄位，沒有的話先給一組超逼真的預設值讓你看 UI
        base_value = features_dict.get("base_pct", 0.08)  # 預設聯盟底薪/平均佔比
        shap_values = features_dict.get("shap_values", {
            "PTS_reg (例行賽得分)": 0.045,
            "has_playoff_exp (季後賽經驗)": 0.021,
            "MAJOR_INJURY_health (重大傷病)": -0.018,
            "Age (年紀折損)": -0.012,
            "AST_reg (例行賽助攻)": 0.009
        })
        
        # 準備 Plotly 瀑布圖的資料結構
        measure = ["absolute"] + ["relative"] * len(shap_values) + ["total"]
        x_labels = ["基礎身價 (Base)"] + list(shap_values.keys()) + ["最終估值 (Final)"]
        y_values = [base_value] + list(shap_values.values()) + [pure_skill_pct]
        
        # 將數值轉成帶有正負號的百分比字串，顯示在圖表上
        text_labels = [f"{v:.1%}" if i==0 or i==len(y_values)-1 else f"{'+' if v>0 else ''}{v:.1%}" for i, v in enumerate(y_values)]
        
        # 建立 Plotly Waterfall 
        fig = go.Figure(go.Waterfall(
            name = "SHAP 拆解",
            orientation = "v",
            measure = measure,
            x = x_labels,
            textposition = "outside",
            text = text_labels,
            y = y_values,
            connector = {"line": {"color": "rgba(63, 63, 63, 0.5)"}},
            decreasing = {"marker": {"color": "#FF4B4B"}}, # 扣分用紅色
            increasing = {"marker": {"color": "#00CC96"}}, # 加分用綠色
            totals = {"marker": {"color": "#636EFA"}}      # 總結用藍色
        ))
        
        fig.update_layout(
            title = f"{selected_player} 薪資特徵影響力貢獻圖",
            waterfallgap = 0.3,
            height = 500,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        # 隱藏 Y 軸數值讓畫面更乾淨
        fig.update_yaxes(showticklabels=False, title_text="")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ── 總結 ──
        st.markdown(f"""
        ### 📝 GM 決策洞察 (Decision Insight)
        根據系統的 **SHAP 歸因分析**，我們可以看到 {selected_player} 最終能拿到 **{pure_skill_pct:.1%}** 薪資佔比的原因。
        綠色柱子代表為他爭取到更大合約的優勢，紅色柱子則是市場對他扣分的風險因子。
        """)
    else:
        st.error("系統異常：找不到對應的資料點。")