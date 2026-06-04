import pandas as pd
import numpy as np
import shap
import pickle
import os

class SalaryValuationEngine:
    def __init__(self, models_dir='../../models'):
        """載入已經訓練好的模型與 Explainer"""
        # 載入定價模型
        with open(os.path.join(models_dir, 'pricing_model.pkl'), 'rb') as f:
            self.pricing_model = pickle.load(f)
            
        # 載入年限模型
        with open(os.path.join(models_dir, 'duration_model.pkl'), 'rb') as f:
            self.duration_model = pickle.load(f)
            
        # 載入 SHAP Explainer
        with open(os.path.join(models_dir, 'shap_explainer.pkl'), 'rb') as f:
            self.explainer = pickle.load(f)
            
        # 實務上可以存成 config.json 讀取
        self.group_a = ['age', 'has_playoff_exp', 'VORP_reg', 'BPM_reg', 'TS%_reg', 'VORP_Elevation'] # 示意特徵
        self.group_b = ['is_retained']
        
        # 歷史薪資帽對照表 (用於換算美金)
        self.salary_cap_dict = {
            2021: 112414000,
            2022: 123655000,
            2023: 136021000,
            2024: 141000000
        }

    def predict_player_value(self, player_name: str, target_year: int, X_database: pd.DataFrame, ids_database: pd.DataFrame):
        """
        前端呼叫的主要 API 接口
        """
        player_name = player_name.lower().strip()
        
        # 1. 從資料庫中尋找該球員當年的特徵向量
        match_idx = np.where((ids_database['Player'] == player_name) & (ids_database['year'] == target_year))[0]
        
        if len(match_idx) == 0:
            return {"error": f"找不到 {player_name} 在 {target_year} 年的資料。"}
            
        idx = match_idx[0]
        player_features = X_database.iloc[[idx]]
        
        # 2. 預測合約年限
        predicted_yrs = self.duration_model.predict(player_features)[0]
        
        # 3. 執行 SHAP 分析
        shap_vals_obj = self.explainer(player_features)
        player_shap = shap_vals_obj.values[0]
        base_value = self.explainer.expected_value
        
        # 3.1 動態定義 group_a 和 group_b
        self.group_b = ['is_retained']
        # 扣除掉 group_b 的欄位，剩下的全部都是純實力特徵 (group_a)
        self.group_a = [col for col in X_database.columns if col not in self.group_b]
        
        # 分離實力與雜訊 (這兩行維持不變)
        skill_impact = sum([player_shap[X_database.columns.get_loc(f)] for f in self.group_a if f in X_database.columns])
        noise_impact = sum([player_shap[X_database.columns.get_loc(f)] for f in self.group_b if f in X_database.columns])
        
        pure_skill_pct = base_value + skill_impact
        total_pred_pct = pure_skill_pct + noise_impact
        
        # 4. 換算成真實美金 (USD)
        cap_limit = self.salary_cap_dict.get(target_year, 136021000) # 找不到就預設 2023 年
        
        pure_skill_salary = pure_skill_pct * cap_limit
        total_market_salary = total_pred_pct * cap_limit
        noise_premium = noise_impact * cap_limit

        # 5. 萃取 Top 5 SHAP 影響力特徵
        # 將特徵名稱與算出來的 SHAP 值配對
        feature_names = X_database.columns.tolist()
        shap_dict = dict(zip(feature_names, player_shap))
        
        # 按照「絕對值」排序，找出對薪資影響力最大的前 5 個特徵
        sorted_features = sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)
        
        # 轉成字典，並確保數值是標準的 float (避免 JSON 序列化報錯)
        top_5_shap = {k: float(v) for k, v in sorted_features[:5]}
        # ==========================================

        # 5. 回傳給前端的 JSON 格式資料
        return {
            "player": player_name.title(),
            "year": target_year,
            "salary_cap_used": cap_limit,
            "predicted_years": int(predicted_yrs),
            "valuation": {
                "base_pct": float(base_value),
                "pure_skill_pct": float(pure_skill_pct),
                "market_noise_pct": float(noise_impact),
                "total_pct": float(total_pred_pct),
                "shap_values": top_5_shap  # <=== 把 Top 5 塞進這裡！
            },
            "money_usd": {
                "pure_skill_salary": int(pure_skill_salary),
                "market_total_salary": int(total_market_salary),
                "noise_premium": int(noise_premium)
            }
        }