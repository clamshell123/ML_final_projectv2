"""
Simple Exploratory Data Analysis for NBA Salary Valuation System
Statistical analysis without plotting dependencies
"""

import pandas as pd
import numpy as np
import json
import os

def load_and_prepare_data():
    """Load the engineered features dataset"""
    df = pd.read_csv('data/processed/featured_nba_data.csv')

    # Separate feature types
    target_col = 'pct_of_cap'
    id_cols = ['Player', 'year']

    # Identify feature columns (exclude target and IDs)
    feature_cols = [col for col in df.columns
                   if col not in id_cols + [target_col, 'salary_cap', 'Cap_Pct']]

    # Get Group A and Group B features from attributes if available
    try:
        group_a_features = df.attrs.get('group_a_features', [])
        group_b_features = df.attrs.get('group_b_features', [])
    except:
        # Fallback: infer feature groups based on name patterns
        group_a_indicators = ['VORP', 'BPM', 'WS', 'PER', 'TS%', 'eFG%', '3PAr', 'FTr',
                             'USG%', 'AST%', 'TRB%', 'AST', 'TRB', 'ORB%', 'DRB%',
                             'STL%', 'BLK%', 'TOV%', 'MP', 'G', 'GS', 'PTS', 'FG',
                             'FTA', 'FT', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                             'Age', 'Height', 'Weight']
        group_b_indicators = ['YRS', 'is_retained', 'Bird', 'cap_space', 'payroll',
                             'draft', 'option']

        group_a_features = [col for col in feature_cols
                           if any(indicator in col for indicator in group_a_indicators)]
        group_b_features = [col for col in feature_cols
                           if any(indicator in col for indicator in group_b_indicators)]

        # Ensure we don't double-count
        group_b_features = [col for col in group_b_features if col not in group_a_features]

    return df, target_col, id_cols, feature_cols, group_a_features, group_b_features

def basic_statistics(df, target_col):
    """Calculate and display basic statistics"""
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of unique players: {df['Player'].nunique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Number of player-seasons: {len(df)}")
    print()

    print("Target variable (pct_of_cap) statistics:")
    target_stats = df[target_col].describe()
    print(target_stats)
    print()

    # Additional target stats
    print("Additional target variable insights:")
    print(f"  Skewness: {df[target_col].skew():.4f}")
    print(f"  Kurtosis: {df[target_col].kurtosis():.4f}")
    print(f"  90th percentile: {df[target_col].quantile(0.90):.4f}")
    print(f"  95th percentile: {df[target_col].quantile(0.95):.4f}")
    print(f"  99th percentile: {df[target_col].quantile(0.99):.4f}")
    print()

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values: {missing_count}")
    if missing_count > 0:
        print("Top 10 columns with missing values:")
        missing_cols = df.isnull().sum().sort_values(ascending=False).head(10)
        for col, count in missing_cols.items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    print()

def feature_group_analysis(df, group_a_features, group_b_features, feature_cols):
    """Analyze the engineered feature groups"""
    print("="*60)
    print("FEATURE GROUP ANALYSIS")
    print("="*60)

    # Filter to features that actually exist
    existing_group_a = [f for f in group_a_features if f in df.columns]
    existing_group_b = [f for f in group_b_features if f in df.columns]
    other_features = [f for f in feature_cols
                     if f not in existing_group_a and f not in existing_group_b]

    print(f"Group A (Pure Skill) features: {len(existing_group_a)}")
    print(f"Group B (Market Noise) features: {len(existing_group_b)}")
    print(f"Other/Uncategorized features: {len(other_features)}")
    print()

    if len(existing_group_a) > 0:
        print("Sample Group A features (first 15):")
        for i, feat in enumerate(existing_group_a[:15], 1):
            print(f"  {i:2d}. {feat}")
        if len(existing_group_a) > 15:
            print(f"     ... and {len(existing_group_a) - 15} more")
        print()

    if len(existing_group_b) > 0:
        print("Group B features:")
        for i, feat in enumerate(existing_group_b, 1):
            print(f"  {i:2d}. {feat}")
        print()

def correlation_with_target(df, feature_cols, target_col, group_a_features, group_b_features):
    """Analyze correlations with target variable"""
    print("="*60)
    print("CORRELATION WITH TARGET VARIABLE")
    print("="*60)

    # Get numeric features
    numeric_features = []
    for col in feature_cols:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            # Check if column has sufficient non-null values
            if df[col].notnull().sum() > len(df) * 0.5:  # At least 50% non-null
                numeric_features.append(col)

    if len(numeric_features) == 0:
        print("No sufficient numeric features found for correlation analysis")
        return

    print(f"Analyzing correlations for {len(numeric_features)} numeric features")
    print()

    # Calculate correlations
    correlations = {}
    for feature in numeric_features:
        # Calculate Pearson correlation, ignoring NaN pairs
        valid_data = df[[feature, target_col]].dropna()
        if len(valid_data) > 10:  # Need reasonable sample size
            corr = valid_data[feature].corr(valid_data[target_col])
            correlations[feature] = corr

    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(),
                                key=lambda x: abs(x[1]),
                                reverse=True)

    print("Top 20 features by absolute correlation with pct_of_cap:")
    print("-" * 50)
    for i, (feature, corr) in enumerate(sorted_correlations[:20], 1):
        print(f"{i:2d}. {feature:<30} {corr:>8.4f}")

    print()
    print("Bottom 10 features by absolute correlation with pct_of_cap:")
    print("-" * 50)
    for i, (feature, corr) in enumerate(sorted_correlations[-10:], 1):
        print(f"{i:2d}. {feature:<30} {corr:>8.4f}")
    print()

    # Group-wise analysis
    existing_group_a = [f for f in group_a_features if f in df.columns]
    existing_group_b = [f for f in group_b_features if f in df.columns]

    if len(existing_group_a) > 0:
        group_a_numeric = [f for f in existing_group_a if f in correlations]
        if len(group_a_numeric) > 0:
            group_a_avgs = np.mean([abs(correlations[f]) for f in group_a_numeric])
            print(f"Group A average |correlation|: {group_a_avgs:.4f}")
            print(f"   (based on {len(group_a_numeric)} features)")

    if len(existing_group_b) > 0:
        group_b_numeric = [f for f in existing_group_b if f in correlations]
        if len(group_b_numeric) > 0:
            group_b_avgs = np.mean([abs(correlations[f]) for f in group_b_numeric])
            print(f"Group B average |correlation|: {group_b_avgs:.4f}")
            print(f"   (based on {len(group_b_numeric)} features)")
    print()

    # Save correlations to file
    corr_df = pd.DataFrame(list(correlations.items()),
                          columns=['feature', 'correlation'])
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False)
    corr_df.to_csv('data/processed/feature_target_correlations.csv', index=False)
    print("Full correlation results saved to: data/processed/feature_target_correlations.csv")
    print()

def temporal_analysis(df, target_col):
    """Analyze trends over time"""
    print("="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)

    yearly_stats = df.groupby('year').agg(
        mean_pct=(target_col, 'mean'),
        std_pct=(target_col, 'std'),
        median_pct=(target_col, 'median'),
        count=(target_col, 'count'),
        min_pct=(target_col, 'min'),
        max_pct=(target_col, 'max')
    ).reset_index()

    print("Yearly statistics for pct_of_cap:")
    print(yearly_stats.round(4))
    print()

    # Year-over-year changes
    yearly_stats['yoy_change'] = yearly_stats['mean_pct'].pct_change()
    print("Year-over-year % change in mean pct_of_cap:")
    for _, row in yearly_stats.iterrows():
        if pd.notnull(row['yoy_change']):
            print(f"  {int(row['year'])}: {row['yoy_change']:+.2%}")
    print()

    # Contract count trends
    print("Contract volume by year:")
    for _, row in yearly_stats.iterrows():
        print(f"  {int(row['year'])}: {int(row['count'])} contracts")
    print()

def contract_analysis(df):
    """Analyze contract-specific features"""
    print("="*60)
    print("CONTRACT FEATURES ANALYSIS")
    print("="*60)

    contract_features = ['YRS', 'is_retained']
    existing_contract = [f for f in contract_features if f in df.columns]

    if len(existing_contract) > 0:
        print("Contract features statistics:")
        print(df[existing_contract].describe())
        print()

        if 'YRS' in df.columns:
            print("Contract length (YRS) distribution:")
            yrs_counts = df['YRS'].value_counts().sort_index()
            for yrs, count in yrs_counts.items():
                print(f"  {yrs} year contracts: {count:3d} ({count/len(df)*100:5.1f}%)")
            print()

        if 'is_retained' in df.columns:
            retained_counts = df['is_retained'].value_counts()
            print("Player retention status:")
            for status, count in retained_counts.items():
                status_label = "Retained with same team" if status == 1 else "Changed teams/New contract"
                print(f"  {status_label}: {count:3d} ({count/len(df)*100:5.1f}%)")
            print()

            # Retention by contract length
            if 'YRS' in df.columns:
                print("Retention rate by contract length:")
                retention_by_yrs = df.groupby('YRS')['is_retained'].agg(['mean', 'count'])
                for yrs, row in retention_by_yrs.iterrows():
                    if row['count'] >= 5:  # Only show if reasonable sample size
                        print(f"  {yrs} year contracts: {row['mean']:.1%} retained ({int(row['count'])} contracts)")
                print()

def outliers_and_extremes(df, target_col):
    """Analyze outliers and extreme values"""
    print("="*60)
    print("OUTLIERS AND EXTREME VALUES ANALYSIS")
    print("="*60)

    # Target variable extremes using IQR
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]

    print(f"Target variable outliers (IQR method):")
    print(f"  Range: [{Q1:.4f}, {Q3:.4f}], IQR: {IQR:.4f}")
    print(f"  Outlier bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
    print()

    if len(outliers) > 0:
        print("Top 10 highest pct_of_cap contracts (outliers and extremes):")
        top_10 = df.nlargest(10, target_col)[['Player', 'year', target_col, 'YRS', 'is_retained']]
        for _, row in top_10.iterrows():
            retained_str = "Retained" if row['is_retained'] == 1 else "New Team"
            print(f"  {row['Player']} ({int(row['year'])}): {row[target_col]:.4f} "
                  f"({int(row['YRS'])}yr, {retained_str})")

        print()
        print("Bottom 10 lowest pct_of_cap contracts:")
        bottom_10 = df.nsmallest(10, target_col)[['Player', 'year', target_col, 'YRS', 'is_retained']]
        for _, row in bottom_10.iterrows():
            retained_str = "Retained" if row['is_retained'] == 1 else "New Team"
            print(f"  {row['Player']} ({int(row['year'])}): {row[target_col]:.4f} "
                  f"({int(row['YRS'])}yr, {retained_str})")
    print()

    # Check for extreme values in key performance metrics
    print("Extreme values in key performance metrics:")
    key_metrics = ['VORP', 'BPM', 'WS', 'PER', 'TS%', 'USG%']
    available_metrics = [m for m in key_metrics if m in df.columns]

    for metric in available_metrics:
        if df[metric].notnull().sum() > 10:
            q99 = df[metric].quantile(0.99)
            q01 = df[metric].quantile(0.01)
            max_val = df[metric].max()
            min_val = df[metric].min()

            print(f"  {metric}:")
            print(f"    1st percentile: {q01:.3f}")
            print(f"    99th percentile: {q99:.3f}")
            print(f"    Actual range: [{min_val:.3f}, {max_val:.3f}]")
    print()

def generate_summary_report(df, target_col, id_cols, feature_cols,
                           group_a_features, group_b_features):
    """Generate a summary report of the EDA"""
    print("="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)

    # Create a summary dictionary
    summary = {
        "dataset_info": {
            "shape": list(df.shape),
            "unique_players": int(df['Player'].nunique()),
            "year_range": [int(df['year'].min()), int(df['year'].max())],
            "total_player_seasons": int(len(df))
        },
        "target_variable": {
            "name": target_col,
            "mean": float(df[target_col].mean()),
            "median": float(df[target_col].median()),
            "std": float(df[target_col].std()),
            "min": float(df[target_col].min()),
            "max": float(df[target_col].max()),
            "q1": float(df[target_col].quantile(0.25)),
            "q3": float(df[target_col].quantile(0.75)),
            "skewness": float(df[target_col].skew()),
            "kurtosis": float(df[target_col].kurtosis())
        },
        "feature_groups": {
            "group_a_pure_skill_count": len([f for f in group_a_features if f in df.columns]),
            "group_b_market_noise_count": len([f for f in group_b_features if f in df.columns]),
            "total_features": len(feature_cols)
        },
        "data_quality": {
            "total_missing_values": int(df.isnull().sum().sum()),
            "missing_percentage": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        }
    }

    # Save summary as JSON
    with open('data/processed/eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("Summary report saved to: data/processed/eda_summary.json")
    print()

def main():
    """Main EDA function"""
    print("Starting Exploratory Data Analysis for NBA Salary Valuation System")
    print("="*70)

    # Load data
    df, target_col, id_cols, feature_cols, group_a_features, group_b_features = load_and_prepare_data()

    # Run analyses
    basic_statistics(df, target_col)
    feature_group_analysis(df, group_a_features, group_b_features, feature_cols)
    correlation_with_target(df, feature_cols, target_col, group_a_features, group_b_features)
    temporal_analysis(df, target_col)
    contract_analysis(df)
    outliers_and_extremes(df, target_col)
    generate_summary_report(df, target_col, id_cols, feature_cols,
                           group_a_features, group_b_features)

    print("="*60)
    print("EDA ANALYSIS COMPLETE")
    print("="*60)
    print("Generated files:")
    print("  - data/processed/eda_summary.json")
    print("  - data/processed/feature_target_correlations.csv")
    print("  - data/processed/eda_summary_statistics.csv (from describe)")

if __name__ == "__main__":
    main()