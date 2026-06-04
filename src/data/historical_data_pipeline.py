import pandas as pd
import numpy as np
import re
import logging
import unicodedata
import os

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 階段一：資料清洗 (Data Cleaning)
# ==========================================
def clean_player_names(df: pd.DataFrame, name_col: str = 'Player') -> pd.DataFrame:
    """執行名稱正規化 (Regex + Unicode 去重音 + 亂碼防禦)"""
    df = df.copy()

    def normalize_name(name):
        if pd.isna(name): return name
        name = str(name).lower()
        
        # 🚨 [亂碼修復] 攔截東歐姓氏常出現的亂碼特徵，以及全明星星號
        name = name.replace('*', '')
        name = name.replace('viä', 'vic')
        name = name.replace('kiä', 'kic')
        name = name.replace('äiä', 'cic')
        name = name.replace('å ', 's ')
        
        # 將 Unicode 字元分解，並過濾掉附加符號
        name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
        
        # 移除了重音後，再過濾掉標點符號與特殊字元
        name = re.sub(r'[^\w\s]', '', name)
        # 移除 Jr., Sr., III 等後綴
        name = re.sub(r' (jr|sr|ii|iii|iv)$', '', name)
        
        return name.strip()

    df[name_col] = df[name_col].apply(normalize_name)
    
    # 手動 mapping 處理綽號與亂碼特例
    name_mapping = {
        'nicolas claxton': 'nic claxton',
        'marcus morris sr': 'marcus morris',
        'kelly oubre': 'kelly oubre jr',
        'nene hilario': 'nene',
        'luc richard mbah a moute': 'luc mbah a moute',
        'marcus georgeshunt': 'marcus georges hunt',
        'dario aria': 'dario saric',
        'luka donaia': 'luka doncic',
        'ishmael smith': 'ish smith',
        'jose barea': 'jj barea',
        'louis williams': 'lou williams',
        'mohamed bamba': 'mo bamba',
        'ishmail wainright': 'ish wainright',
        'herb jones': 'herbert jones',
        'juancho hernangomez': 'juan hernangomez',
        'vincent poirier': 'vince poirier',
        'pj dozier': 'p j dozier',
        'michael porter': 'michael porter jr'
    }
    df[name_col] = df[name_col].replace(name_mapping)
    return df


def resolve_traded_players(df: pd.DataFrame) -> pd.DataFrame:
    """處理季中交易球員 (2TM, 3TM, TOT)，保留總數據並換上季末球隊"""
    df = df.copy()
    multi_team_labels = ['TOT', '2TM', '3TM', '4TM', '5TM']
    indices_to_keep = []
    
    for (player, year), group in df.groupby(['Player', 'year']):
        tot_rows = group[group['Team'].isin(multi_team_labels)]
        
        if not tot_rows.empty:
            tot_idx = tot_rows.index[0]
            team_rows = group[~group['Team'].isin(multi_team_labels)]
            
            if not team_rows.empty:
                # 抓取該季最後效力的球隊
                final_team = team_rows['Team'].iloc[-1]
                df.loc[tot_idx, 'Team'] = final_team
                
            indices_to_keep.append(tot_idx)
        else:
            indices_to_keep.extend(group.index.tolist())
            
    df_cleaned = df.loc[indices_to_keep].reset_index(drop=True)
    return df_cleaned


def clean_stats_data(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """專屬 B-Ref 統計數據的清洗邏輯 (強化型別防護)"""
    df = df.copy()
    df = clean_player_names(df, 'Player')

    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player']

    # 確保 year 欄位絕對是整數
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    # 🚨 [修正重點]：在篩選出場數之前，先解決季中交易球員！
    if 'Team' in df.columns:
        df = resolve_traded_players(df)

    id_cols = ['Player', 'Team', 'Pos', 'Awards', 'Season', 'Type', 'year']
    numeric_cols = [col for col in df.columns if col not in id_cols]
    
    for col in numeric_cols:
        if col in df.columns:
            # 移除字串中可能干擾轉換的星號、逗號或空白
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(r'[\*\,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 🚨 在交易合併後，才執行出場數過濾，確保主力不被誤刪
    if 'G' in df.columns:
        df = df[df['G'] >= min_games]

    rate_cols = [col for col in df.columns if '%' in col]
    if rate_cols:
        df[rate_cols] = df[rate_cols].fillna(0)

    return df


# ==========================================
# 階段二：資料合併與時間加權 (針對歷史推論)
# ==========================================
def compute_weighted_stats(df: pd.DataFrame, player: str, target_year: int, default_weights: list = [0.2, 0.3, 0.5]) -> pd.Series:
    """回傳球員在 target_year 及前兩年的加權平均數據 (安全加權版)"""
    id_cols = ['Player', 'year', 'Team', 'Pos', 'Age'] 
    stat_cols = [c for c in df.columns if c not in id_cols]
    numeric_stat_cols = df[stat_cols].select_dtypes(include=[np.number]).columns
    
    years_needed = [target_year - 2, target_year - 1, target_year]
    available_data = []

    for y in years_needed:
        sub = df[(df['Player'] == player) & (df['year'] == y)]
        if not sub.empty:
            series_data = sub[numeric_stat_cols].iloc[0].copy()
            series_data.name = None
            available_data.append((y, series_data))
        else:
            available_data.append((y, None))

    if available_data[-1][1] is None:
        return pd.Series([np.nan] * len(numeric_stat_cols), index=numeric_stat_cols)

    valid_weights = [default_weights[i] for i in range(3) if available_data[i][1] is not None]
    weight_sum = sum(valid_weights)

    weighted = pd.Series(0.0, index=numeric_stat_cols)
    for i, (y, data_series) in enumerate(available_data):
        if data_series is not None:
            normalized_weight = default_weights[i] / weight_sum
            weighted = weighted.add(data_series * normalized_weight, fill_value=0)
            
    return weighted


def merge_historical_stats(stats_reg_df: pd.DataFrame, stats_playoff_df: pd.DataFrame) -> pd.DataFrame:
    """以例行賽出現的 (Player, Year) 為基準進行資料合併"""
    enriched_rows = []
    unique_player_years = stats_reg_df[['Player', 'year']].drop_duplicates()

    for _, row in unique_player_years.iterrows():
        player = row['Player']
        yr = int(row['year'])
        
        # 🚨 [修正重點]：抓出 Age 的同時，一併抓出 Team 和 Pos 供模型編碼使用
        age_sub = stats_reg_df[(stats_reg_df['Player'] == player) & (stats_reg_df['year'] == yr)]
        if not age_sub.empty:
            age = age_sub['Age'].iloc[0]
            team = age_sub['Team'].iloc[0]
            pos = age_sub['Pos'].iloc[0]
        else:
            age = np.nan
            team = 'Unknown'
            pos = 'Unknown'

        wt_reg = compute_weighted_stats(stats_reg_df, player, yr)
        wt_playoff = compute_weighted_stats(stats_playoff_df, player, yr)

        if wt_playoff.isna().all():
            has_playoff_exp = 0 
            wt_playoff = wt_reg.copy()
        else:
            has_playoff_exp = 1 

        wt_reg = wt_reg.add_suffix('_reg')
        wt_playoff = wt_playoff.add_suffix('_playoff')

        new_row = {
            'Player': player,
            'Team': team,  # 補上 Team
            'Pos': pos,    # 補上 Pos
            'year': yr,
            'age': age,
            'has_playoff_exp': has_playoff_exp,
            'is_retained': 0  # 填入 dummy 雜訊，模型推論必須要有此欄位
        }
        
        for col in wt_reg.index:
            new_row[col] = wt_reg[col]
        for col in wt_playoff.index:
            new_row[col] = wt_playoff[col]

        enriched_rows.append(new_row)

    return pd.DataFrame(enriched_rows)


# ==========================================
# 階段三：特徵工程 (Feature Engineering)
# ==========================================
def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """建立有意義的進階互動特徵"""
    df = df.copy()

    if 'VORP_reg' in df.columns and 'VORP_playoff' in df.columns:
        df['VORP_Elevation'] = np.where(
            df['has_playoff_exp'] == 1,
            df['VORP_playoff'] - df['VORP_reg'],
            0 
        )
        
    if 'BPM_reg' in df.columns and 'BPM_playoff' in df.columns:
         df['BPM_Elevation'] = np.where(
            df['has_playoff_exp'] == 1,
            df['BPM_playoff'] - df['BPM_reg'],
            0
        )

    if 'MP_reg' in df.columns and 'MP_playoff' in df.columns:
        total_mp_reg = df['MP_reg'] * df['G_reg']
        total_mp_playoff = df['MP_playoff'] * df['G_playoff']
        df['Playoff_MP_Ratio'] = np.where(
            total_mp_reg + total_mp_playoff > 0,
            total_mp_playoff / (total_mp_reg + total_mp_playoff),
            0
        )

    df = df.fillna(0)
    return df


# ==========================================
# 主程式管線執行器 (Pipeline Executor)
# ==========================================
def run_data_pipeline(raw_reg_stats: pd.DataFrame, raw_playoff_stats: pd.DataFrame) -> pd.DataFrame:
    """
    執行歷史資料工程管線，純粹產出推論用的特徵矩陣 (X)
    """
    logger.info("開始執行 NBA 歷史資料工程管線...")
    
    logger.info("1/3 執行資料清洗...")
    # 🚨 [修正重點]：常規賽與季後賽門檻分流
    clean_reg = clean_stats_data(raw_reg_stats, min_games=10)
    clean_playoff = clean_stats_data(raw_playoff_stats, min_games=2)
    
    logger.info("2/3 執行時間加權與資料合併...")
    merged_df = merge_historical_stats(clean_reg, clean_playoff)
    
    logger.info("3/3 執行進階特徵工程...")
    final_df = create_interaction_features(merged_df)
    
    # 🚨 [修正重點]：將 Team 和 Pos 納入元資料保護區，確保它們出現在最前面
    meta_cols = ['Player', 'Team', 'Pos', 'year', 'age', 'has_playoff_exp', 'is_retained']
    feature_cols = [c for c in final_df.columns if c not in meta_cols]
    final_df = final_df[meta_cols + sorted(feature_cols)]
    
    logger.info(f"✅ 管線執行完成！最終產出資料維度: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    logger.info("載入原始資料中...")
    
    # 請確保此路徑與你的專案結構一致
    raw_reg = pd.read_csv('../../data/raw/history_stats_reg.csv', encoding='utf-8')
    raw_playoff = pd.read_csv('../../data/raw/history_stats_playoff.csv', encoding='utf-8')

    # 執行管線
    final_dataset = run_data_pipeline(raw_reg, raw_playoff)
    
    # 將處理好的特徵矩陣存檔，準備餵給 batch_inference.py
    output_path = '../../data/processed/historical_legends_features.csv'
    
    # 確保資料夾存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    final_dataset.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"已將推論用特徵資料庫存至: {output_path}")