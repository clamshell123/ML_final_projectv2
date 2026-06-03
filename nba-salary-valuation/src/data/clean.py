"""
Data Cleaning Pipeline for NBA Salary Valuation System
Handles name standardization, missing values, and data quality issues
"""

import pandas as pd
import numpy as np
import re
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_player_names(df: pd.DataFrame, name_col: str = 'Player') -> pd.DataFrame:
    """
    Standardize player names by removing accents, normalizing spacing, etc.

    Args:
        df: Input DataFrame
        name_col: Column name containing player names

    Returns:
        DataFrame with cleaned player names
    """
    df = df.copy()

    # Remove extra whitespace
    df[name_col] = df[name_col].str.strip()

    # Common name variations mapping (add more as needed)
    name_mapping = {
        # Add known problematic names here
        # 'Nikola Jokić': 'Nikola Jokic',
        # 'José Calderón': 'Jose Calderon',
    }

    # Apply known mappings
    if name_mapping:
        df[name_col] = df[name_col].replace(name_mapping)

    # Remove any remaining special characters that might cause issues
    # Keep letters, spaces, periods, apostrophes, and hyphens
    df[name_col] = df[name_col].apply(
        lambda x: re.sub(r"[^a-zA-Z\s\.\'\-]", "", str(x)) if pd.notnull(x) else x
    )

    logger.info(f"Cleaned player names in column '{name_col}'")
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: dict = None,
    numeric_fill: str = 'median',
    categorical_fill: str = 'mode'
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Args:
        df: Input DataFrame
        strategy: Dictionary mapping column names to fill strategies
        numeric_fill: Strategy for numeric columns ('median', 'mean', 'zero')
        categorical_fill: Strategy for categorical columns ('mode', 'constant')

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    if strategy is None:
        strategy = {}

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target variable and key identifiers from automatic processing
    protected_cols = ['Player', 'Season', 'year', 'Cap_Pct']
    numeric_cols = [col for col in numeric_cols if col not in protected_cols]
    categorical_cols = [col for col in categorical_cols if col not in protected_cols]

    # Handle numeric columns
    for col in numeric_cols:
        if col in strategy:
            # Use custom strategy
            if strategy[col] == 'drop':
                df = df.dropna(subset=[col])
            elif strategy[col] == 'zero':
                df[col] = df[col].fillna(0)
            elif strategy[col] == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            elif strategy[col] == 'backward_fill':
                df[col] = df[col].fillna(method='bfill')
        else:
            # Use default numeric strategy
            if numeric_fill == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif numeric_fill == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif numeric_fill == 'zero':
                df[col] = df[col].fillna(0)

    # Handle categorical columns
    for col in categorical_cols:
        if col in strategy:
            # Use custom strategy
            if strategy[col] == 'drop':
                df = df.dropna(subset=[col])
            elif strategy[col] == 'constant':
                df[col] = df[col].fillna('Unknown')
            elif strategy[col] == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            elif strategy[col] == 'backward_fill':
                df[col] = df[col].fillna(method='bfill')
        else:
            # Use default categorical strategy
            if categorical_fill == 'mode':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('Unknown')
            elif categorical_fill == 'constant':
                df[col] = df[col].fillna('Unknown')

    logger.info(f"Handled missing values using {numeric_fill} for numeric, {categorical_fill} for categorical")
    return df


def filter_season_range(df: pd.DataFrame, season_col: str = 'Season', min_year: int = 2017, max_year: int = 2024) -> pd.DataFrame:
    """
    Filter data to specified season range.

    Args:
        df: Input DataFrame
        season_col: Column name containing season/year information
        min_year: Minimum year to keep (inclusive)
        max_year: Maximum year to keep (inclusive)

    Returns:
        Filtered DataFrame
    """
    df = df.copy()

    if season_col in df.columns:
        initial_count = len(df)
        df = df[(df[season_col] >= min_year) & (df[season_col] <= max_year)]
        filtered_count = len(df)
        logger.info(f"Filtered seasons {min_year}-{max_year}: {initial_count} -> {filtered_count} rows")

    return df


def remove_duplicate_player_seasons(df: pd.DataFrame, player_col: str = 'Player', season_col: str = 'Season') -> pd.DataFrame:
    """
    Remove duplicate player-season entries, keeping the first occurrence.

    Args:
        df: Input DataFrame
        player_col: Column name for player identifier
        season_col: Column name for season/year

    Returns:
        DataFrame with duplicates removed
    """
    df = df.copy()

    initial_count = len(df)
    df = df.drop_duplicates(subset=[player_col, season_col], keep='first')
    final_count = len(df)

    if initial_count != final_count:
        logger.info(f"Removed {initial_count - final_count} duplicate player-season entries")

    return df


def clean_contract_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specific cleaning for contract data.

    Args:
        df: Contract DataFrame with columns Player, YRS, is_retained, year, Cap_Pct

    Returns:
        Cleaned contract DataFrame
    """
    df = df.copy()

    # Clean player names
    df = clean_player_names(df, 'Player')

    # Ensure year is integer
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')

    # Ensure Cap_Pct is numeric and in reasonable range
    if 'Cap_Pct' in df.columns:
        df['Cap_Pct'] = pd.to_numeric(df['Cap_Pct'], errors='coerce')
        # Cap reasonable values (0% to 50% of cap is reasonable for NBA)
        df['Cap_Pct'] = df['Cap_Pct'].clip(lower=0, upper=0.50)

    # Ensure YRS is positive integer
    if 'YRS' in df.columns:
        df['YRS'] = pd.to_numeric(df['YRS'], errors='coerce').clip(lower=1, upper=10)

    # Ensure is_retained is binary
    if 'is_retained' in df.columns:
        df['is_retained'] = pd.to_numeric(df['is_retained'], errors='coerce').fillna(0).astype(int)
        df['is_retained'] = df['is_retained'].clip(lower=0, upper=1)

    logger.info("Cleaned contract data")
    return df


def clean_stats_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specific cleaning for player statistics data.

    Args:
        df: Statistics DataFrame from Basketball Reference

    Returns:
        Cleaned statistics DataFrame
    """
    df = df.copy()

    # Clean player names
    df = clean_player_names(df, 'Player')

    # Remove rows where Player is actually the header (should already be done, but double-check)
    if 'Player' in df.columns:
        df = df[df['Player'] != 'Player']

    # Convert Season to integer if it exists
    if 'Season' in df.columns:
        df['Season'] = pd.to_numeric(df['Season'], errors='coerce').astype('Int64')

    # Identify numeric columns (excluding identifier columns)
    id_cols = ['Player', 'Team', 'Pos', 'Awards', 'Season', 'Type']
    numeric_cols = [col for col in df.columns if col not in id_cols]

    # Convert numeric columns, coercing errors to NaN
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info("Cleaned stats data")
    return df


def clean_salary_cap_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Specific cleaning for salary cap data.

    Args:
        df: Salary cap DataFrame

    Returns:
        Cleaned salary cap DataFrame
    """
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Handle common column name variations
    column_mapping = {
        'Salary_Cap': 'salary_cap',
        'salary_cap_usd': 'salary_cap',
        'year': 'season',
        'Year': 'season'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})

    # Ensure required columns exist
    required_cols = ['season', 'salary_cap']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in salary cap data: {missing_cols}")

    # Convert to appropriate types
    if 'season' in df.columns:
        df['season'] = pd.to_numeric(df['season'], errors='coerce').astype('Int64')

    if 'salary_cap' in df.columns:
        df['salary_cap'] = pd.to_numeric(df['salary_cap'], errors='coerce')
        # Remove any negative or zero values (shouldn't exist but safety check)
        df['salary_cap'] = df['salary_cap'].clip(lower=1000000)  # Minimum $1M cap

    logger.info("Cleaned salary cap data")
    return df


def validate_data_quality(df: pd.DataFrame, dataset_name: str = "Dataset") -> dict:
    """
    Perform basic data quality validation.

    Args:
        df: DataFrame to validate
        dataset_name: Name for logging purposes

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }

    logger.info(f"{dataset_name} validation: {validation_results['total_rows']} rows, "
                f"{validation_results['total_columns']} columns, "
                f"{validation_results['missing_values'].sum()} missing values")

    return validation_results


if __name__ == "__main__":
    # Example usage
    print("Data cleaning module for NBA Salary Valuation System")
    print("Import and use the functions in your pipeline")