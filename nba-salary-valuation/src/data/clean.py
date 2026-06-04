import pandas as pd
import numpy as np

def clean_standard_names(df, player_col):
    """統一常規球員名稱格式 (First Last)，移除空格與歐陸重音"""
    df = df.copy()
    df[player_col] = df[player_col].astype(str).str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
    df[player_col] = df[player_col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    return df

def convert_imperial_to_metric(df):
    """將體測指標從美制 (英吋/磅) 轉換為公制 (公分/公斤)"""
    df = df.copy()
    inch_columns = ['HGT', 'WNGSPN', 'STNDRCH', 'HANDL']
    for col in inch_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') * 2.54
            df[col] = df[col].round(1)
    if 'WGT' in df.columns:
        df['WGT'] = pd.to_numeric(df['WGT'], errors='coerce') * 0.45359237
        df['WGT'] = df['WGT'].round(1)
    return df