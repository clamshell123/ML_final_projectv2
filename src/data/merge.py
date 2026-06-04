import pandas as pd
from src.data.clean import clean_standard_names

def preprocess_injury_history(injury_df):
    """預處理傷病事件並聚合至球員賽季層級"""
    injury_df = injury_df.copy()
    
    # 依據 Kaggle 實際欄位名稱 'Name' 進行清洗與欄位重命名
    injury_df = clean_standard_names(injury_df, 'Name')
    injury_df.rename(columns={'Name': 'player'}, inplace=True)
    
    # 轉換時間 (加上 format='mixed' 避免歐美日期格式混雜報錯)
    injury_df['injury_date'] = pd.to_datetime(injury_df['Date'], format='mixed')
    
    # 計算對應 NBA 賽季年份規則 (10月以後算下一季)
    injury_df['season'] = injury_df['injury_date'].apply(lambda d: d.year + 1 if d.month >= 10 else d.year)
    
    # 比對傷病嚴重度
    injury_df['injury_type_lower'] = injury_df['Notes'].astype(str).str.lower()
    injury_df['is_severe_injury'] = injury_df['injury_type_lower'].str.contains('acl|achilles|patellar|tears|fracture').astype(int)
    
    # 依據 ['player', 'season'] 聚合統計次數
    injury_season_agg = injury_df.groupby(['player', 'season']).agg(
        total_injuries=('injury_type_lower', 'count'),
        had_severe_injury=('is_severe_injury', 'max')
    ).reset_index()
    
    return injury_season_agg

def preprocess_biometric_stats(bio_df):
    """預處理全體球員身體資料庫 (只保留每位球員最新的一筆特徵)"""
    bio_df = bio_df.copy()
    
    # 清洗姓名與轉換欄位名稱
    bio_df = clean_standard_names(bio_df, 'player_name')
    bio_df.rename(columns={'player_name': 'player'}, inplace=True)
    
    # 將賽季降序排序，確保最新年份排在最上面
    bio_df = bio_df.sort_values(by='season', ascending=False)
    
    # 🌟 核心修正點：去掉 season 限制，同球員只保留最新的一筆身高體重
    bio_unique = bio_df.drop_duplicates(subset=['player'], keep='first')
    
    # 保留建模需要的特徵
    keep_cols = ['player', 'player_height', 'player_weight', 'draft_year', 'draft_number']
    return bio_unique[keep_cols]

def merge_nba_pipeline(nba_stats_df, injury_df, bio_df):
    """主資料串接核心整合流水線 (時間外推無損版)"""
    main_df = nba_stats_df.copy()
    
    # 1. 清洗主表姓名並標準化
    main_df = clean_standard_names(main_df, 'Player')
    main_df.rename(columns={'Player': 'player', 'Year': 'season'}, inplace=True)
    main_df['season'] = main_df['season'].astype(int)
    
    # 2. 執行次要特徵預處理
    processed_injury = preprocess_injury_history(injury_df)
    processed_bio = preprocess_biometric_stats(bio_df) # 取得球員不變的身高體重字典
    
    # 3. 串接傷病數據 (左連接)
    merged_df = pd.merge(main_df, processed_injury, on=['player', 'season'], how='left')
    merged_df['total_injuries'] = merged_df['total_injuries'].fillna(0).astype(int)
    merged_df['had_severe_injury'] = merged_df['had_severe_injury'].fillna(0).astype(int)
    
    # 4. 🌟 核心修正點：改用 how='left' 且只對齊 'player'
    # 這樣 2024, 2025, 2026 年度的老球員數據，就能完美自動繼承他們原有的身高體重！
    final_df = pd.merge(merged_df, processed_bio, on='player', how='left')
    
    return final_df