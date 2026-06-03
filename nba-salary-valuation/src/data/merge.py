"""
Data Merging Pipeline for NBA Salary Valuation System
Combines contract data, player statistics, and salary cap information
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_contract_stats(
    contract_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    contract_player_col: str = 'Player',
    contract_year_col: str = 'year',
    stats_player_col: str = 'Player',
    stats_season_col: str = 'Season',
    stats_type_col: str = 'Type',
    stats_type_filter: str = 'Regular',
    weighted_years: List[int] = [0, 1, 2],
    weights: List[float] = [0.5, 0.3, 0.2]
) -> pd.DataFrame:
    """
    Merge contract data with player statistics using weighted historical averages.

    For each contract year, we look back at the previous N seasons and apply weights:
    - Current year (t): weight 0.5
    - Previous year (t-1): weight 0.3
    - Two years prior (t-2): weight 0.2

    Args:
        contract_df: DataFrame with contract information (Player, year, Cap_Pct, etc.)
        stats_df: DataFrame with player statistics (Player, Season, various stats)
        contract_player_col: Column name for player in contract data
        contract_year_col: Column name for year in contract data
        stats_player_col: Column name for player in stats data
        stats_season_col: Column name for season in stats data
        stats_type_col: Column name for type (Regular/Playoffs)
        stats_type_filter: Value to filter for stats type (usually 'Regular')
        weighted_years: List of year offsets [0, -1, -2] for t, t-1, t-2
        weights: Corresponding weights [0.5, 0.3, 0.2]

    Returns:
        Merged DataFrame with weighted statistics features
    """
    logger.info("Starting contract-stats merge with weighted historical averages")

    # Filter stats to regular season only
    if stats_type_col in stats_df.columns:
        stats_regular = stats_df[stats_df[stats_type_col] == stats_type_filter].copy()
    else:
        stats_regular = stats_df.copy()
        logger.warning(f"No {stats_type_col} column found, assuming all data is regular season")

    # Prepare result list
    merged_rows = []

    # Process each contract
    for _, contract_row in contract_df.iterrows():
        player_name = contract_row[contract_player_col]
        contract_year = int(contract_row[contract_year_col])

        # Initialize weighted stats dictionary
        weighted_stats = {}

        # Get stats for each weighted year
        total_weight = 0

        for year_offset, weight in zip(weighted_years, weights):
            target_year = contract_year + year_offset  # year_offset is negative or 0

            # Get player stats for this year
            player_stats = stats_regular[
                (stats_regular[stats_player_col] == player_name) &
                (stats_regular[stats_season_col] == target_year)
            ]

            if len(player_stats) > 0:
                # Take the first row if multiple entries (shouldn't happen with proper data)
                player_stats = player_stats.iloc[0]

                # Add weighted stats
                for col in player_stats.index:
                    if col not in [stats_player_col, stats_season_col, stats_type_col]:
                        if pd.notnull(player_stats[col]):
                            # Try to convert to numeric, skip if not possible
                            try:
                                numeric_val = float(player_stats[col])
                                stat_name = f"{col}_t{year_offset}" if year_offset != 0 else col
                                if stat_name not in weighted_stats:
                                    weighted_stats[stat_name] = 0
                                weighted_stats[stat_name] += weight * numeric_val
                                total_weight += weight
                            except (ValueError, TypeError):
                                # Skip non-numeric columns like Team, Pos, etc.
                                continue

        # Normalize weights if we didn't get full data (e.g., for rookie years)
        if total_weight > 0:
            # Normalize so weights sum to 1.0
            for stat_name in weighted_stats:
                weighted_stats[stat_name] /= total_weight
        else:
            # No stats found - this shouldn't happen for established players
            logger.warning(f"No stats found for {player_name} around year {contract_year}")
            # Fill with zeros or league averages would go here
            pass

        # Create merged row
        merged_row = contract_row.to_dict()
        merged_row.update(weighted_stats)
        merged_rows.append(merged_row)

    # Create final DataFrame
    result_df = pd.DataFrame(merged_rows)

    logger.info(f"Merged {len(result_df)} contracts with weighted statistics")
    return result_df


def merge_salary_cap(
    merged_df: pd.DataFrame,
    salary_cap_df: pd.DataFrame,
    year_col: str = 'year',
    cap_year_col: str = 'season',
    cap_value_col: str = 'salary_cap'
) -> pd.DataFrame:
    """
    Merge salary cap data into the main dataset.

    Args:
        merged_df: Main DataFrame with year column
        salary_cap_df: DataFrame with season and salary_cap columns
        year_col: Column name for year in merged data
        cap_year_col: Column name for season in salary cap data
        cap_value_col: Column name for salary cap value

    Returns:
        DataFrame with salary_cap column added
    """
    logger.info("Merging salary cap data")

    # Ensure we have the required columns
    if cap_year_col not in salary_cap_df.columns:
        # Try to find year column
        possible_year_cols = [col for col in salary_cap_df.columns if 'year' in col.lower() or 'season' in col.lower()]
        if possible_year_cols:
            cap_year_col = possible_year_cols[0]
        else:
            logger.error(f"Could not find year/season column in salary cap data. Columns: {list(salary_cap_df.columns)}")
            return merged_df

    if cap_value_col not in salary_cap_df.columns:
        # Try to find salary cap column
        possible_cap_cols = [col for col in salary_cap_df.columns if 'cap' in col.lower() or 'salary' in col.lower()]
        if possible_cap_cols:
            cap_value_col = possible_cap_cols[0]
        else:
            logger.error(f"Could not find salary cap column. Columns: {list(salary_cap_df.columns)}")
            return merged_df

    # Prepare salary cap data for merging
    cap_data = salary_cap_df[[cap_year_col, cap_value_col]].copy()
    cap_data[cap_year_col] = pd.to_numeric(cap_data[cap_year_col], errors='coerce')
    cap_data = cap_data.rename(columns={cap_year_col: year_col, cap_value_col: 'salary_cap'})

    # Merge with main data
    result_df = pd.merge(
        merged_df,
        cap_data,
        on=year_col,
        how='left',
        validate='m:1'  # Many contracts to one salary cap per year
    )

    logger.info(f"Added salary_cap column. Missing values: {result_df['salary_cap'].isnull().sum()}")
    return result_df


def create_pct_of_cap_target(
    df: pd.DataFrame,
    pct_col: str = 'Cap_Pct',
    target_col: str = 'pct_of_cap'
) -> pd.DataFrame:
    """
    Ensure we have a proper pct_of_cap target variable.
    Since we already scraped Cap_Pct, we just rename/validate it.

    Args:
        df: DataFrame with Cap_Pct column
        pct_col: Name of the percentage column
        target_col: Desired name for target variable

    Returns:
        DataFrame with properly named target column
    """
    logger.info("Creating pct_of_cap target variable")

    df = df.copy()

    if pct_col in df.columns:
        # Rename to standard target variable name
        df[target_col] = df[pct_col]

        # Validate range (0 to 0.5 or 0% to 50% is reasonable)
        min_val = df[target_col].min()
        max_val = df[target_col].max()

        if min_val < 0 or max_val > 0.5:
            logger.warning(f"pct_of_cap values outside expected range [0, 0.5]: [{min_val:.4f}, {max_val:.4f}]")

        logger.info(f"Target variable '{target_col}' created. Range: [{df[target_col].min():.4f}, {df[target_col].max():.4f}]")
    else:
        logger.error(f"Could not find {pct_col} column to create target variable")

    return df


def merge_nba_data(
    contract_path: str,
    stats_path: str,
    salary_cap_path: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Complete pipeline to merge all NBA data sources.

    Args:
        contract_path: Path to contract CSV file
        stats_path: Path to player statistics CSV file
        salary_cap_path: Path to salary cap CSV file
        output_path: Optional path to save merged data

    Returns:
        Fully merged DataFrame ready for feature engineering
    """
    logger.info("Starting complete NBA data merge pipeline")

    # Load data
    logger.info(f"Loading contract data from {contract_path}")
    contract_df = pd.read_csv(contract_path)

    logger.info(f"Loading stats data from {stats_path}")
    stats_df = pd.read_csv(stats_path)

    logger.info(f"Loading salary cap data from {salary_cap_path}")
    salary_cap_df = pd.read_csv(salary_cap_path)

    # Clean data (basic cleaning - more detailed cleaning in clean.py)
    contract_df = contract_df.copy()
    stats_df = stats_df.copy()
    salary_cap_df = salary_cap_df.copy()

    # Merge contract with stats (using weighted historical averages)
    logger.info("Step 1: Merging contract data with weighted player statistics")
    merged_df = merge_contract_stats(contract_df, stats_df)

    # Merge with salary cap data
    logger.info("Step 2: Adding salary cap information")
    merged_df = merge_salary_cap(merged_df, salary_cap_df)

    # Create target variable
    logger.info("Step 3: Creating target variable")
    merged_df = create_pct_of_cap_target(merged_df)

    # Reorder columns to put identifiers and target up front
    id_cols = ['Player', 'year']
    target_col = ['pct_of_cap'] if 'pct_of_cap' in merged_df.columns else []
    cap_col = ['salary_cap'] if 'salary_cap' in merged_df.columns else []

    # Get remaining feature columns
    feature_cols = [col for col in merged_df.columns
                   if col not in id_cols + target_col + cap_col]

    # Reorder
    column_order = id_cols + target_col + cap_col + feature_cols
    # Only include columns that actually exist
    column_order = [col for col in column_order if col in merged_df.columns]
    merged_df = merged_df[column_order]

    logger.info(f"Merge complete. Final dataset shape: {merged_df.shape}")
    logger.info(f"Columns: {list(merged_df.columns)}")

    # Save if output path provided
    if output_path:
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to {output_path}")

    return merged_df


if __name__ == "__main__":
    # Example usage when run directly
    import os

    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')

    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    contract_file = os.path.join(raw_dir, 'player_Cap_pct_2017_2024.csv')
    stats_file = os.path.join(raw_dir, 'Stats_reg_2014_2024.csv')  # Using regular season stats
    cap_file = os.path.join(raw_dir, 'Salaery_cap_2017_2026.csv')
    output_file = os.path.join(processed_dir, 'merged_nba_data.csv')

    # Run merge
    try:
        merged_data = merge_nba_data(contract_file, stats_file, cap_file, output_file)
        print("\nMerge completed successfully!")
        print(f"Shape: {merged_data.shape}")
        print(f"Columns: {list(merged_data.columns)}")
        print(f"\nFirst few rows:")
        print(merged_data.head())
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        raise