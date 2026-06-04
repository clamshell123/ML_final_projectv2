import pandas as pd
import numpy as np
import re
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 階段一：資料清洗 (Data Cleaning)
# ==========================================
def clean_player_names(df: pd.DataFrame, name_col: str = 'Player') -> pd.DataFrame:
    """執行名稱正規化 (Regex)"""
    df = df.copy()

    def normalize_name(name):
        if pd.isna(name):
            return name
        name = str(name).lower()
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r' (jr|sr|ii|iii|iv)$', '', name)
        return name.strip()

    df[name_col] = df[name_col].apply(normalize_name)
    
    name_mapping = {
        'nicolas claxton': 'nic claxton',
        'marcus morris sr': 'marcus morris',
        'kelly oubre': 'kelly oubre jr' 
    }
    df[name_col] = df[name_col].replace(name_mapping)
    return df

def clean_stats_data(df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """專屬 B-Ref 統計數據的清洗邏輯"""
    df = df.copy()
    df = clean_player_names(df, 'Player')

    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player']

    id_cols = ['Player', 'Team', 'Pos', 'Awards', 'Season', 'Type', 'year']
    numeric_cols = [col for col in df.columns if col not in id_cols]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
    """回傳球員在 target_year 及前兩年的加權平均數據 (支援動態權重補缺)"""
    id_cols = ['Player', 'year', 'Team', 'Pos', 'Age'] 
    stat_cols = [c for c in df.columns if c not in id_cols]
    numeric_stat_cols = df[stat_cols].select_dtypes(include=[np.number]).columns
    
    years_needed = [target_year - 2, target_year - 1, target_year]
    available_data = []

    # 1. 收集這三年內確實存在的數據
    for y in years_needed:
        sub = df[(df['Player'] == player) & (df['year'] == y)]
        if not sub.empty:
            available_data.append((y, sub[numeric_stat_cols].iloc[0]))
        else:
            available_data.append((y, None))

    # 2. 如果目標年份 (target_year) 本身就沒打球，直接回傳 NaN
    if available_data[-1][1] is None:
        return pd.Series([np.nan] * len(numeric_stat_cols), index=numeric_stat_cols)

    # 3. 動態重新分配權重
    # 抓出有資料的對應權重，並計算總和
    valid_weights = [default_weights[i] for i in range(3) if available_data[i][1] is not None]
    weight_sum = sum(valid_weights)

    weighted = pd.Series(0.0, index=numeric_stat_cols)
    for i, (y, data_series) in enumerate(available_data):
        if data_series is not None:
            # 正規化權重 (例如只有前一年跟當年，原本 0.3 和 0.5 會放大為 0.375 與 0.625)
            normalized_weight = default_weights[i] / weight_sum
            weighted += normalized_weight * data_series
            
    return weighted

def merge_historical_stats(stats_reg_df: pd.DataFrame, stats_playoff_df: pd.DataFrame) -> pd.DataFrame:
    """直接以例行賽出現的 (Player, Year) 為基準進行資料合併，不再依賴合約表"""
    enriched_rows = []
    
    # 取得所有獨一無二的 (球員, 年份) 組合
    unique_player_years = stats_reg_df[['Player', 'year']].drop_duplicates()

    for _, row in unique_player_years.iterrows():
        player = row['Player']
        yr = int(row['year'])
        
        age_sub = stats_reg_df[(stats_reg_df['Player'] == player) & (stats_reg_df['year'] == yr)]
        age = age_sub['Age'].iloc[0] if not age_sub.empty else np.nan

        wt_reg = compute_weighted_stats(stats_reg_df, player, yr)
        wt_playoff = compute_weighted_stats(stats_playoff_df, player, yr)

        if wt_playoff.isna().all():
            has_playoff_exp = 0 
            wt_playoff = wt_reg.copy()
        else:
            has_playoff_exp = 1 

        wt_reg = wt_reg.add_suffix('_reg')
        wt_playoff = wt_playoff.add_suffix('_playoff')

        # [修正重點]：移除 YRS 與 Cap_Pct，加入 is_retained 預設值防止模型報錯
        new_row = {
            'Player': player,
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
    clean_reg = clean_stats_data(raw_reg_stats)
    clean_playoff = clean_stats_data(raw_playoff_stats)
    
    logger.info("2/3 執行時間加權與資料合併...")
    merged_df = merge_historical_stats(clean_reg, clean_playoff)
    
    logger.info("3/3 執行進階特徵工程...")
    final_df = create_interaction_features(merged_df)
    
    # 重新排序，將目標拿掉，留下元資料在最前面
    meta_cols = ['Player', 'year', 'age', 'has_playoff_exp', 'is_retained']
    feature_cols = [c for c in final_df.columns if c not in meta_cols]
    final_df = final_df[meta_cols + sorted(feature_cols)]
    
    logger.info(f"✅ 管線執行完成！最終產出資料維度: {final_df.shape}")
    return final_df

if __name__ == "__main__":
    # 讀取剛剛用爬蟲抓下來的歷史資料
    logger.info("載入原始資料中...")
    raw_reg = pd.read_csv('../../data/raw/history_stats_reg.csv')
    raw_playoff = pd.read_csv('../../data/raw/history_stats_playoff.csv')

    # 執行管線
    final_dataset = run_data_pipeline(raw_reg, raw_playoff)
    
    # 將處理好的特徵矩陣存檔，準備餵給 batch_inference.py
    output_path = '../../data/processed/historical_legends_features.csv'
    final_dataset.to_csv(output_path, index=False)
    logger.info(f"已將推論用特徵資料庫存至: {output_path}")