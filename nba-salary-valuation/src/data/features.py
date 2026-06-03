"""
Feature Engineering Pipeline for NBA Salary Valuation System
Implements A/B grouping (Pure Skill vs Market Noise) and feature transformations
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def define_ab_groups() -> Tuple[List[str], List[str]]:
    """
    Define Group A (Pure Skill/On-Court) and Group B (External Market Noise) features.

    Based on instruction.md and README.md:
    - Group A: On-court skill (VORP, age, injury, games played, position)
    - Group B: Market noise (Bird rights, team cap space, draft pick, contract options)

    Returns:
        Tuple of (group_a_features, group_b_features)
    """
    # Group A: Pure basketball skills and player status
    group_a_features = [
        # Advanced impact metrics
        'VORP',           # Value Over Replacement Player
        'BPM',            # Box Plus/Minus
        'WS',             # Win Shares
        'WS/48',          # Win Shares per 48 minutes
        'OBPM',           # Offensive Box Plus/Minus
        'DBPM',           # Defensive Box Plus/Minus

        # Efficiency and shooting
        'TS%',            # True Shooting Percentage
        'eFG%',           # Effective Field Goal Percentage
        '3PAr',           # 3-Point Attempt Rate
        'FTr',            # Free Throw Rate
        'FG%',            # Field Goal Percentage
        '3P%',            # 3-Point Percentage
        'FT%',            # Free Throw Percentage

        # Usage and role
        'USG%',           # Usage Percentage
        'AST%',           # Assist Percentage
        'AST',            # Assists per game
        'TRB%',           # Rebound Percentage
        'TRB',            # Rebounds per game
        'ORB%',           # Offensive Rebound Percentage
        'DRB%',           # Defensive Rebound Percentage
        'STL%',           # Steal Percentage
        'BLK%',           # Block Percentage
        'TOV%',           # Turnover Percentage

        # Playing time and durability
        'MP',             # Minutes Per Game
        'G',              # Games Played
        'GS',             # Games Started

        # Basic counting stats (per game)
        'PTS',            # Points
        'FG',             # Field Goals Made
        'FGA',            # Field Goals Attempted
        '3P',             # 3-Pointers Made
        '3PA',            # 3-Pointers Attempted
        'FT',             # Free Throws Made
        'FTA',            # Free Throws Attempted
        'ORB',            # Offensive Rebounds
        'DRB',            # Defensive Rebounds
        'AST',            # Assists
        'STL',            # Steals
        'BLK',            # Blocks
        'TOV',            # Turnovers
        'PF',             # Personal Fouls

        # Physical attributes
        'Age',            # Player Age
        'Height',         # Player Height (if available)
        'Weight',         # Player Weight (if available)

        # Position encoding (will be handled separately)
        'Pos_C',          # Center
        'Pos_PF',         # Power Forward
        'Pos_PG',         # Point Guard
        'Pos_SF',         # Small Forward
        'Pos_SG',         # Shooting Guard
    ]

    # Group B: External market and contract factors
    group_b_features = [
        # Contract-specific features (from scraping)
        'YRS',            # Years of contract
        'is_retained',    # Whether player was retained by same team

        # Team context features (would need team-level data)
        # These would typically come from team salary data:
        # 'team_cap_space',      # Team's available cap space
        # 'team_payroll_ratio',  # Team payroll as % of cap
        # 'bird_rights',         # Bird rights flag
        # 'draft_pick',          # Draft pick number (for rookie contracts)
        # 'contract_options',    # Team/player options in contract

        # For now, we'll use available proxies and note that
        # some Group B features require additional team-level data
    ]

    logger.info(f"Defined {len(group_a_features)} Group A (Skill) features")
    logger.info(f"Defined {len(group_b_features)} Group B (Market Noise) features")

    return group_a_features, group_b_features


def create_position_dummies(df: pd.DataFrame, pos_col: str = 'Pos') -> pd.DataFrame:
    """
    Create one-hot encoded position features.

    Args:
        df: Input DataFrame
        pos_col: Column name containing position information

    Returns:
        DataFrame with position dummy variables added
    """
    df = df.copy()

    if pos_col not in df.columns:
        logger.warning(f"Position column '{pos_col}' not found")
        return df

    # Clean position data
    df[pos_col] = df[pos_col].astype(str).str.strip().str.upper()

    # Create position dummies for the 5 standard positions
    positions = ['C', 'PF', 'PG', 'SF', 'SG']
    for pos in positions:
        col_name = f'Pos_{pos}'
        df[col_name] = (df[pos_col] == pos).astype(int)

    logger.info(f"Created position dummy variables for positions: {positions}")
    return df


def calculate_a_b_scores(
    df: pd.DataFrame,
    group_a_features: List[str],
    group_b_features: List[str],
    shap_values_dict: Optional[Dict[str, np.ndarray]] = None
) -> pd.DataFrame:
    """
    Calculate pure skill and market noise scores.

    According to instruction.md:
    - pure_skill_pct = base + shap_A_sum
    - market_noise_pct = shap_B_sum

    However, for feature engineering preparation, we separate the features
    so they can be used with SHAP later.

    Args:
        df: Input DataFrame with features
        group_a_features: List of Group A feature column names
        group_b_features: List of Group B feature column names
        shap_values_dict: Optional pre-calculated SHAP values by feature

    Returns:
        DataFrame with Group A and Group B feature subsets ready for SHAP
    """
    df = df.copy()

    # Filter to only include features that actually exist in the dataframe
    available_group_a = [f for f in group_a_features if f in df.columns]
    available_group_b = [f for f in group_b_features if f in df.columns]

    missing_group_a = set(group_a_features) - set(available_group_a)
    missing_group_b = set(group_b_features) - set(available_group_b)

    if missing_group_a:
        logger.warning(f"Missing Group A features: {missing_group_a}")
    if missing_group_b:
        logger.warning(f"Missing Group B features: {missing_group_b}")

    logger.info(f"Available Group A features: {len(available_group_a)}")
    logger.info(f"Available Group B features: {len(available_group_b)}")

    # Store the feature lists for later use in modeling/SHAP
    df.attrs['group_a_features'] = available_group_a
    df.attrs['group_b_features'] = available_group_b

    # If SHAP values are provided, calculate the scores
    if shap_values_dict is not None:
        # Sum SHAP values for each group
        shap_a_sum = np.zeros(len(df))
        shap_b_sum = np.zeros(len(df))

        for feature in available_group_a:
            if feature in shap_values_dict:
                shap_a_sum += shap_values_dict[feature]

        for feature in available_group_b:
            if feature in shap_values_dict:
                shap_b_sum += shap_values_dict[feature]

        # According to instruction.md:
        # pure_skill_pct = base + shap_A_sum
        # market_noise_pct = shap_B_sum
        #
        # Note: The "base" would be the expected value (mean) from the model
        # For now, we'll store the SHAP sums and let the modeling step add the base

        df['shap_A_sum'] = shap_a_sum
        df['shap_B_sum'] = shap_b_sum

        logger.info("Calculated SHAP A/B sums from provided SHAP values")

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meaningful interaction features for basketball valuation.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()

    # Age-related interactions
    if 'Age' in df.columns and 'MP' in df.columns:
        df['Age_MP_interaction'] = df['Age'] * df['MP']
        logger.info("Created Age x MP interaction")

    # Efficiency interactions
    if 'TS%' in df.columns and 'USG%' in df.columns:
        df['TS_USG_ratio'] = df['TS%'] / (df['USG%'] + 0.001)  # Avoid division by zero
        logger.info("Created TS%/USG% ratio")

    # Experience/durability
    if 'G' in df.columns and 'Age' in df.columns:
        df['Games_per_Year'] = df['G'] / (df['Age'] - 18 + 0.001)  # Rough approximation of years played
        logger.info("Created Games per Year feature")

    # Per-minute rates (if we have counting stats and minutes)
    if 'MP' in df.columns and (df['MP'] > 0).any():
        rate_stats = ['PTS', 'TRB', 'AST', 'STL', 'BLK']
        for stat in rate_stats:
            if stat in df.columns:
                # Avoid division by zero
                mp_safe = df['MP'].replace(0, 0.1)
                df[f'{stat}_per_36'] = df[stat] / (mp_safe / 36) * 36
                logger.info(f"Created {stat} per 36 minutes")

    return df


def prepare_features_for_modeling(
    df: pd.DataFrame,
    target_col: str = 'pct_of_cap',
    id_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare the final feature set for modeling.

    Args:
        df: Input DataFrame with all features
        target_col: Name of target variable column
        id_cols: Columns to exclude from features (identifiers)

    Returns:
        Tuple of (features_DataFrame, target_Series, group_a_features, group_b_features)
    """
    df = df.copy()

    if id_cols is None:
        id_cols = ['Player', 'year', 'Season']  # Common identifier columns

    # Get target variable
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    target = df[target_col]

    # Get feature columns (exclude target and id columns)
    exclude_cols = set(id_cols + [target_col])
    if 'salary_cap' in df.columns:
        exclude_cols.add('salary_cap')  # This is used for conversion, not modeling

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    features = df[feature_cols]

    # Get A/B groups if they were stored as attributes
    group_a_features = df.attrs.get('group_a_features', [])
    group_b_features = df.attrs.get('group_b_features', [])

    # Filter to only include features that exist in our feature set
    group_a_features = [f for f in group_a_features if f in features.columns]
    group_b_features = [f for f in group_b_features if f in features.columns]

    logger.info(f"Prepared {len(feature_cols)} features for modeling")
    logger.info(f"Group A features available for modeling: {len(group_a_features)}")
    logger.info(f"Group B features available for modeling: {len(group_b_features)}")

    return features, target, group_a_features, group_b_features


def engineer_nba_features(
    merged_data_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline for NBA data.

    Args:
        merged_data_path: Path to merged NBA data CSV
        output_path: Optional path to save engineered features

    Returns:
        DataFrame with engineered features ready for modeling
    """
    logger.info("Starting NBA feature engineering pipeline")

    # Load merged data
    df = pd.read_csv(merged_data_path)
    logger.info(f"Loaded merged data with shape: {df.shape}")

    # Step 1: Create position dummies
    pos_col = 'Pos' if 'Pos' in df.columns else None
    if pos_col is not None:
        df = create_position_dummies(df, pos_col)
    else:
        logger.warning("Position column 'Pos' not found in data")

    # Step 2: Define A/B groups
    group_a_features, group_b_features = define_ab_groups()

    # Step 3: Prepare features for modeling (this stores the group definitions)
    features, target, _, _ = prepare_features_for_modeling(df)

    # Step 4: Create interaction features
    df = create_interaction_features(df)

    # Step 5: Update attributes with final feature lists (after interactions)
    # Re-calculate available features after engineering
    available_features = [col for col in df.columns
                         if col not in ['Player', 'year', 'Season', 'pct_of_cap', 'salary_cap']]

    # Update stored feature lists
    df.attrs['group_a_features'] = [f for f in group_a_features if f in available_features]
    df.attrs['group_b_features'] = [f for f in group_b_features if f in available_features]

    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Group A features: {len(df.attrs['group_a_features'])}")
    logger.info(f"Group B features: {len(df.attrs['group_b_features'])}")

    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Engineered features saved to {output_path}")

    return df


if __name__ == "__main__":
    # Example usage when run directly
    import os

    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    processed_dir = os.path.join(data_dir, 'processed')

    merged_file = os.path.join(processed_dir, 'merged_nba_data.csv')
    output_file = os.path.join(processed_dir, 'featured_nba_data.csv')

    # Run feature engineering
    try:
        featured_data = engineer_nba_features(merged_file, output_file)
        print("\nFeature engineering completed successfully!")
        print(f"Shape: {featured_data.shape}")
        print(f"Group A features: {len(featured_data.attrs.get('group_a_features', []))}")
        print(f"Group B features: {len(featured_data.attrs.get('group_b_features', []))}")
        print(f"\nFeature columns: {list(featured_data.columns)}")
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise