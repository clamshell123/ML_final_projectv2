import streamlit as st
import requests
import plotly.graph_objects as go

# 設定後端 API 的位址
API_URL = "http://127.0.0.1:8000"

# --- 頁面基本設定 ---
st.set_page_config(
    page_title="NBA 跨時空薪資模擬器", 
    page_icon="🏀", 
    layout="wide"
)

st.title("🏀 NBA 跨時空 Moneyball 薪資模擬器")
st.markdown("透過 XGBoost 與 SHAP 剝離歷史市場雜訊，將球員的「純粹籃球實力」無縫轉換至現代或未來的薪資體系。")
st.divider()

# --- 核心邏輯：CBA 10% 薪資帽預測 ---
def get_salary_cap(target_year):
    """
    根據目標年份回傳薪資帽。
    套用 CBA 規則：未來年份每年平滑增長 10%。
    """
    # 近年真實薪資帽基準 (USD)
    base_caps = {
        2022: 123655000,
        2023: 136021000,
        2024: 141000000 
    }
    
    if target_year in base_caps:
        return base_caps[target_year]
    elif target_year > 2024:
        # 計算未來年份的 10% 複利增長
        years_ahead = target_year - 2024
        return base_caps[2024] * (1.10 ** years_ahead)
    else:
        # 若選擇更早的年份，這裡為簡化示範，固定回傳 2022 的數值
        # 實務上可擴充完整的歷史薪資帽字典
        return 100000000 

# --- 1. 從後端獲取可用的球員名單 ---
@st.cache_data(ttl=60)
def fetch_available_players():
    try:
        response = requests.get(f"{API_URL}/players")
        if response.status_code == 200:
            # 回傳格式為 [{"player_name": "...", "stat_year": 1993}, ...]
            return response.json().get("players", [])
        return []
    except Exception as e:
        st.error(f"❌ 無法連線至後端 API，請確認 Uvicorn 伺服器是否已啟動。({e})")
        return []

players_data = fetch_available_players()

# --- 2. 建立側邊欄/輸入介面 ---
st.sidebar.header("⚙️ 模擬器參數設定")

if not players_data:
    st.sidebar.warning("目前資料庫中沒有可用的球員資料。")
else:
    # (1) 提取不重複的球員名單並排序
    unique_players = sorted(list(set([p['player_name'] for p in players_data])))
    selected_player = st.sidebar.selectbox("1️⃣ 球員名稱", unique_players)
    
    # (2) 根據選定的球員，動態過濾出他擁有的數據年份
    raw_years = [p['stat_year'] for p in players_data if p['player_name'] == selected_player]
    available_years = sorted(list(set(raw_years)), reverse=True)
    
    selected_year = st.sidebar.selectbox("2️⃣ 球員實力年分", available_years)
    
    # (3) 預估年代 (支援當前賽季至未來 5 年)
    st.sidebar.markdown("---")
    target_eras = list(range(2024, 2031))
    selected_era = st.sidebar.selectbox(
        "3️⃣ 預估年代 (Target Era)", 
        target_eras,
        help="選擇未來的年份，系統將自動套用 CBA 規定的 10% 薪資帽平滑增長率進行計算。"
    )
    
    st.sidebar.markdown("---")
    analyze_button = st.sidebar.button("🚀 進行跨時空估值", use_container_width=True)

    # --- 3. 處理估值邏輯與視覺化 ---
    if analyze_button:
        # 計算對應年代的薪資帽
        target_cap = get_salary_cap(selected_era)
        
        with st.spinner(f"正在從 Supabase 雲端資料庫調閱 {selected_player} ({selected_year}) 的實力特徵..."):
            try:
                payload = {"player_name": selected_player, "target_year": selected_year}
                response = requests.post(f"{API_URL}/predict", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    pure_skill_pct = data["valuation"]["pure_skill_pct"]
                    
                    # 核心公式：絕對美金 = 預測佔比 * 目標年代薪資帽
                    projected_salary = pure_skill_pct * target_cap
                    
                    # --- 顯示結果層 ---
                    st.header(f"📊 {selected_player} ({selected_year} 實力) ➡️ {selected_era} 年代身價解析")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("🎯 **AI 判定：純粹籃球實力佔比**")
                        st.metric(
                            label="剔除市場雜訊後的薪資帽佔比", 
                            value=f"{pure_skill_pct:.1%}",
                            help="這是 XGBoost 結合 SHAP 剝離大球市、鳥權等干擾後，該球員憑實力應得的合約佔比。"
                        )
                        
                    with col2:
                        st.success(f"💰 **跨時空換算：{selected_era} 賽季年薪**")
                        st.metric(
                            label=f"在 ${target_cap/1000000:.1f}M 薪資帽下的絕對薪資", 
                            value=f"${projected_salary:,.0f}",
                            delta=f"套用 CBA 10% 規則" if selected_era > 2024 else None,
                            delta_color="normal"
                        )
                    
                    st.divider()
                    
                    # --- 說明區塊 ---
                    st.markdown(f"""
                    ### 📝 GM 決策洞察 (Decision Insight)
                    根據系統的 **A/B 群組剝離分析 (Pure Skill Extraction)**，{selected_player} 在 {selected_year} 賽季展現出的場上影響力（Group A 特徵），讓他絕對值得佔據球隊 **{pure_skill_pct:.1%}** 的薪資空間。
                    
                    如果將他的實力搬到 **{selected_era}** 賽季（該年預估薪資帽為 **${target_cap:,.0f}**），
                    球隊應該為他準備一份年均薪大約為 **${projected_salary:,.0f}** 的合約，這才是符合 Moneyball 邏輯、不被市場情緒綁架的真實驗證定價。
                    """)
                    
                else:
                    st.error(f"後端發生錯誤: {response.json().get('detail', '未知錯誤')}")
            
            except Exception as e:
                st.error(f"無法取得預測結果：{e}")