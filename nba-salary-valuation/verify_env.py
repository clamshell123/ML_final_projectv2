"""
Environment Verification Script - Phase 1
驗證所有必要的套件已安裝且可運作

使用方式:
    python verify_env.py
"""

import sys
import importlib
from pathlib import Path

def check_package(package_name, import_name=None):
    """檢查套件是否可導入"""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name:20s} → {version}")
        return True
    except ImportError:
        print(f"❌ {package_name:20s} → NOT INSTALLED")
        return False


def main():
    """主驗證函數"""
    print("\n" + "="*60)
    print("🔍 Phase 1: 環境驗證")
    print("="*60 + "\n")
    
    # 核心套件
    print("📦 核心數據處理套件:")
    packages_core = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
    ]
    for pkg, imp in packages_core:
        check_package(pkg, imp)
    
    print("\n🤖 機器學習 & 優化:")
    packages_ml = [
        ('xgboost', 'xgboost'),
        ('shap', 'shap'),
        ('optuna', 'optuna'),
        ('mord', 'mord'),
    ]
    for pkg, imp in packages_ml:
        check_package(pkg, imp)
    
    print("\n📊 視覺化 & 前端:")
    packages_viz = [
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]
    for pkg, imp in packages_viz:
        check_package(pkg, imp)
    
    print("\n🛠️ 工具 & 工具:")
    packages_tools = [
        ('jupyter', 'jupyter'),
        ('beautifulsoup4', 'bs4'),
        ('requests', 'requests'),
    ]
    for pkg, imp in packages_tools:
        check_package(pkg, imp)
    
    print("\n" + "="*60)
    print("✅ 環境驗證完成!")
    print("="*60 + "\n")
    
    print("📝 後續步驟:")
    print("   1. 若有缺失套件，執行: pip install -r requirements.txt")
    print("   2. 若使用 conda，執行: conda env create -f environment.yml")
    print("   3. 執行: python download_data.py (下載資料)")
    print("   4. 進行 Phase 2 資料清理\n")


if __name__ == '__main__':
    main()
