"""
Exploratory Data Analysis for NBA Salary Valuation System
Analyzes the engineered features dataset to understand distributions, correlations, and key insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterfilterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
        # Fallback: try to infer from column names or use all features
        group_a_features = []
        group_b_features = []

    return df, target_col, id_cols, feature_cols, group_a_features, group_b_features

def basic_statistics(df, target_col):
    """Calculate and display basic statistics"""
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)

    print(f"Dataset shape: {df.shape}")
    print(f"Number of players: {df['Player'].nunique()}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print()

    print("Target variable (pct_of_cap) statistics:")
    print(df[target_col].describe())
    print()

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"Total missing values: {missing_count}")
    if missing_count > 0:
        print("Missing values by column:")
        missing_cols = df.isnull().sum()[df.isnull().sum() > 0]
        print(missing_cols)
    print()

def distribution_analysis(df, target_col, feature_cols, n_samples=10):
    """Analyze distributions of key features"""
    print("="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)

    # Target variable distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.hist(df[target_col], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Target Variable: pct_of_cap Distribution')
    plt.xlabel('pct_of_cap')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 2)
    plt.boxplot(df[target_col])
    plt.title('Target Variable: pct_of_cap Boxplot')
    plt.ylabel('pct_of_cap')

    # Select some key features for distribution analysis
    key_features = []
    # Add some Group A features if available
    group_a_in_data = [f for f in df.columns if any(x in f for x in ['VORP', 'BPM', 'WS', 'PER', 'TS%'])]
    key_features.extend(group_a_in_data[:3])

    # Add some engineered features
    engineered_in_data = [f for f in df.columns if any(x in f for x in ['_per_36', '_ratio', '_interaction'])]
    key_features.extend(engineered_in_data[:3])

    # Add some basic stats
    basic_in_data = [f for f in df.columns if any(x in f for x in ['PTS', 'TRB', 'AST'])]
    key_features.extend(basic_in_data[:2])

    # Remove duplicates and limit
    key_features = list(dict.fromkeys(key_features))[:6]

    for i, feature in enumerate(key_features):
        if feature in df.columns:
            plt.subplot(2, 3, i+3)
            plt.hist(df[feature].dropna(), bins=25, edgecolor='black', alpha=0.7)
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('data/processed/eda_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics for key features
    print("Key Features Statistics:")
    key_features_with_target = key_features + [target_col]
    key_features_with_target = [f for f in key_features_with_target if f in df.columns]
    print(df[key_features_with_target].describe())
    print()

def correlation_analysis(df, feature_cols, target_col):
    """Analyze correlations between features and target"""
    print("="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Select numeric features for correlation
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_features) > 0:
        # Calculate correlation matrix
        corr_matrix = df[numeric_features + [target_col]].corr()

        # Get correlations with target
        target_correlations = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)

        print(f"Top 10 features most correlated with {target_col}:")
        print(target_correlations.head(10))
        print()

        print(f"Top 10 features least correlated with {target_col}:")
        print(target_correlations.tail(10))
        print()

        # Plot correlation heatmap for top features
        top_features = target_correlations.head(15).index.tolist()
        top_features.append(target_col)

        plt.figure(figsize=(12, 10))
        sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Heatmap: Top Features & Target')
        plt.tight_layout()
        plt.savefig('data/processed/eda_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("No numeric features found for correlation analysis")
    print()

def group_analysis(df, group_a_features, group_b_features, target_col):
    """Analyze Group A vs Group B features"""
    print("="*60)
    print("GROUP A/B ANALYSIS")
    print("="*60)

    # Filter to features that actually exist in the dataframe
    existing_group_a = [f for f in group_a_features if f in df.columns]
    existing_group_b = [f for f in group_b_features if f in df.columns]

    print(f"Group A (Pure Skill) features found: {len(existing_group_a)}")
    print(f"Group B (Market Noise) features found: {len(existing_group_b)}")

    if len(existing_group_a) > 0:
        print("\nGroup A features:")
        for feat in existing_group_a[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(existing_group_a) > 10:
            print(f"  ... and {len(existing_group_a) - 10} more")

    if len(existing_group_b) > 0:
        print("\nGroup B features:")
        for feat in existing_group_b:
            print(f"  - {feat}")

    # Calculate average absolute correlation with target for each group
    if len(existing_group_a) > 0 and len(existing_group_b) > 0:
        # Get numeric features from each group
        group_a_numeric = [f for f in existing_group_a
                          if f in df.columns and df[f].dtype in ['float64', 'int64']]
        group_b_numeric = [f for f in existing_group_b
                          if f in df.columns and df[f].dtype in ['float64', 'int64']]

        if len(group_a_numeric) > 0:
            group_a_corr = df[group_a_numeric + [target_col]].corr()[target_col].drop(target_col).abs().mean()
            print(f"\nGroup A average |correlation| with target: {group_a_corr:.4f}")

        if len(group_b_numeric) > 0:
            group_b_corr = df[group_b_numeric + [target_col]].corr()[target_col].drop(target_col).abs().mean()
            print(f"Group B average |correlation| with target: {group_b_corr:.4f}")
    print()

def temporal_analysis(df, target_col):
    """Analyze trends over time"""
    print("="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)

    # Yearly trends
    yearly_stats = df.groupby('year')[target_col].agg(['mean', 'std', 'count']).reset_index()

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(yearly_stats['year'], yearly_stats['mean'], marker='o', linewidth=2, markersize=4)
    plt.fill_between(yearly_stats['year'],
                     yearly_stats['mean'] - yearly_stats['std'],
                     yearly_stats['mean'] + yearly_stats['std'],
                     alpha=0.3)
    plt.title('Average pct_of_cap by Year (with ±1 std)')
    plt.xlabel('Year')
    plt.ylabel('Mean pct_of_cap')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(yearly_stats['year'], yearly_stats['count'], alpha=0.7, edgecolor='black')
    plt.title('Number of Contracts by Year')
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('data/processed/eda_temporal_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Yearly statistics:")
    print(yearly_stats.round(4))
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

        # YRS distribution
        if 'YRS' in df.columns:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.hist(df['YRS'], bins=range(1, int(df['YRS'].max())+2),
                     edgecolor='black', alpha=0.7, align='left')
            plt.title('Contract Length (YRS) Distribution')
            plt.xlabel('Years')
            plt.ylabel('Frequency')
            plt.xticks(range(1, int(df['YRS'].max())+1))

            plt.subplot(1, 2, 2)
            yr_state = df['is_retained'].value_counts() if 'is_retained' in df.columns else pd.Series()
            plt.pie(yr_state.values, labels=[f'Not Retained ({yr_state.get(0,0)})',
                                             f'Retained ({yr_state.get(1,0)})'] if len(yr_state) > 0 else ['No Data'],
                    autopct='%1.1f%%', startangle=90)
            plt.title('Player Retention Status')

            plt.tight_layout()
            plt.savefig('data/processed/eda_contract_features.png', dpi=300, bbox_inches='tight')
            plt.show()
    print()

def position_analysis(df):
    """Analyze position features if available"""
    print("="*60)
    print("POSITION ANALYSIS (if available)")
    print("="*60)

    # Look for position columns
    pos_cols = [col for col in df.columns if col.startswith('Pos_')]

    if len(pos_cols) > 0:
        print(f"Position columns found: {pos_cols}")
        # Calculate percentage of players at each position
        pos_percentages = {}
        for pos_col in pos_cols:
            if pos_col in df.columns:
                pos_percentages[pos_col.replace('Pos_', '')] = (df[pos_col].mean() * 100)

        if pos_percentages:
            print("Position percentages:")
            for pos, pct in pos_percentages.items():
                print(f"  {pos}: {pct:.1f}%")

            # Plot
            plt.figure(figsize=(10, 6))
            positions = list(pos_percentages.keys())
            percentages = list(pos_percentages.values())
            plt.bar(positions, percentages, alpha=0.7, edgecolor='black')
            plt.title('Percentage of Players by Position')
            plt.xlabel('Position')
            plt.ylabel('Percentage (%)')
            plt.ylim(0, max(percentages)*1.2 if percentages else 100)

            # Add value labels on bars
            for i, pct in enumerate(percentages):
                plt.text(i, pct + max(percentages)*0.01, f'{pct:.1f}%',
                        ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig('data/processed/eda_position_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
    else:
        print("No position columns found in the dataset")
    print()

def outliers_analysis(df, target_col, feature_cols):
    """Analyze outliers in key features"""
    print("="*60)
    print("OUTLIERS ANALYSIS")
    print("="*60)

    # Check for outliers in target variable using IQR method
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]

    print(f"Target variable outliers (IQR method):")
    print(f"  Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
    print(f"  Lower bound: {lower_bound:.4f}, Upper bound: {upper_bound:.4f}")
    print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

    if len(outliers) > 0:
        print(f"  Outlier values: {sorted(outliers[target_col].tolist())}")

        # Show top 5 outlier contracts
        top_outliers = outliers.nlargest(5, target_col)[['Player', 'year', target_col]] if len(outliers) >= 5 else outliers[['Player', 'year', target_col]]
        print(f"  Top outlier contracts:")
        for _, row in top_outliers.iterrows():
            print(f"    {row['Player']} ({row['year']}): {row[target_col]:.4f}")

    print()

    # Also check for extreme values in some key features
    key_features = []
    group_a_in_data = [f for f in df.columns if any(x in f for x in ['VORP', 'BPM', 'WS'])]
    key_features.extend(group_a_in_data[:3])

    for feature in key_features:
        if feature in df.columns:
            Q1_feat = df[feature].quantile(0.25)
            Q3_feat = df[feature].quantile(0.75)
            IQR_feat = Q3_feat - Q1_feat
            lower_feat = Q1_feat - 1.5 * IQR_feat
            upper_feat = Q3_feat + 1.5 * IQR_feat

            outliers_feat = df[(df[feature] < lower_feat) | (df[feature] > upper_feat)]
            print(f"{feature} outliers: {len(outliers_feat)} ({len(outliers_feat)/len(df)*100:.1f}%)")
    print()

def generate_eda_report(df, target_col, id_cols, feature_cols,
                       group_a_features, group_b_features):
    """Generate a comprehensive EDA report"""
    print("="*60)
    print("GENERATING EDA REPORT")
    print("="*60)

    # Run all analyses
    basic_statistics(df, target_col)
    distribution_analysis(df, target_col, feature_cols)
    correlation_analysis(df, feature_cols, target_col)
    group_analysis(df, group_a_features, group_b_features, target_col)
    temporal_analysis(df, target_col)
    contract_analysis(df)
    position_analysis(df)
    outliers_analysis(df, target_col, feature_cols)

    print("="*60)
    print("EDA COMPLETE - Plots saved to data/processed/")
    print("="*60)

def main():
    """Main EDA function"""
    print("Starting Exploratory Data Analysis for NBA Salary Valuation System")
    print("="*70)

    # Load data
    df, target_col, id_cols, feature_cols, group_a_features, group_b_features = load_and_prepare_data()

    # Generate comprehensive report
    generate_eda_report(df, target_col, id_cols, feature_cols,
                       group_a_features, group_b_features)

    # Save summary statistics to file
    summary_stats = df.describe()
    summary_stats.to_csv('data/processed/eda_summary_statistics.csv')
    print("Summary statistics saved to: data/processed/eda_summary_statistics.csv")

    # Save correlation matrix
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_features) > 0:
        corr_matrix = df[numeric_features + [target_col]].corr()
        corr_matrix.to_csv('data/processed/eda_correlation_matrix.csv')
        print("Correlation matrix saved to: data/processed/eda_correlation_matrix.csv")

    print("\nEDA completed successfully!")

if __name__ == "__main__":
    main()