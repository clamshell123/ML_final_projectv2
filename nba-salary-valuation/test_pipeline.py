"""
Test script to run the complete NBA data pipeline:
1. Data cleaning
2. Data merging
3. Feature engineering
"""

import os
import sys
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_pipeline():
    """Run the complete data pipeline test"""

    logger.info("="*60)
    logger.info("TESTING NBA SALARY VALUATION DATA PIPELINE")
    logger.info("="*60)

    # Define paths
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')

    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Define file paths
    contract_file = os.path.join(raw_dir, 'player_Cap_pct_2017_2024.csv')
    stats_file = os.path.join(raw_dir, 'Stats_reg_2014_2024.csv')
    cap_file = os.path.join(raw_dir, 'Salaery_cap_2017_2026.csv')

    # Output paths
    merged_file = os.path.join(processed_dir, 'merged_nba_data.csv')
    featured_file = os.path.join(processed_dir, 'featured_nba_data.csv')

    try:
        # Step 1: Import and test cleaning functions
        logger.info("Step 1: Testing data cleaning functions")
        from src.data.clean import (
            clean_player_names,
            handle_missing_values,
            clean_contract_data,
            clean_stats_data,
            clean_salary_cap_data
        )

        # Load and test cleaning on small samples
        if os.path.exists(contract_file):
            contract_sample = pd.read_csv(contract_file).head(10)
            contract_cleaned = clean_contract_data(contract_sample)
            logger.info(f"Contract cleaning test: {contract_sample.shape} -> {contract_cleaned.shape}")

        if os.path.exists(stats_file):
            stats_sample = pd.read_csv(stats_file).head(10)
            stats_cleaned = clean_stats_data(stats_sample)
            logger.info(f"Stats cleaning test: {stats_sample.shape} -> {stats_cleaned.shape}")

        if os.path.exists(cap_file):
            cap_sample = pd.read_csv(cap_file).head(10)
            cap_cleaned = clean_salary_cap_data(cap_sample)
            logger.info(f"Salary cap cleaning test: {cap_sample.shape} -> {cap_cleaned.shape}")

        # Step 2: Test merging functions
        logger.info("\nStep 2: Testing data merging functions")
        from src.data.merge import (
            merge_contract_stats,
            merge_salary_cap,
            create_pct_of_cap_target,
            merge_nba_data
        )

        # Run full merge if files exist
        if all(os.path.exists(f) for f in [contract_file, stats_file, cap_file]):
            logger.info("Running full data merge...")
            merged_data = merge_nba_data(
                contract_path=contract_file,
                stats_path=stats_file,
                salary_cap_path=cap_file,
                output_path=merged_file
            )
            logger.info(f"Merge completed. Shape: {merged_data.shape}")
        else:
            missing = [f for f in [contract_file, stats_file, cap_file] if not os.path.exists(f)]
            logger.error(f"Missing files for merge test: {missing}")
            return False

        # Step 3: Test feature engineering
        logger.info("\nStep 3: Testing feature engineering functions")
        from src.data.features import (
            define_ab_groups,
            create_position_dummies,
            engineer_nba_features
        )

        # Run full feature engineering
        if os.path.exists(merged_file):
            logger.info("Running feature engineering...")
            featured_data = engineer_nba_features(
                merged_data_path=merged_file,
                output_path=featured_file
            )
            logger.info(f"Feature engineering completed. Shape: {featured_data.shape}")

            # Report on feature groups
            group_a = featured_data.attrs.get('group_a_features', [])
            group_b = featured_data.attrs.get('group_b_features', [])
            logger.info(f"Group A (Pure Skill) features: {len(group_a)}")
            logger.info(f"Group B (Market Noise) features: {len(group_b)}")

            if len(group_a) > 0:
                logger.info(f"Sample Group A features: {group_a[:5]}")
            if len(group_b) > 0:
                logger.info(f"Sample Group B features: {group_b[:5]}")

        else:
            logger.error("Merged file not found for feature engineering test")
            return False

        # Step 4: Final validation
        logger.info("\nStep 4: Final validation")
        if os.path.exists(featured_file):
            final_data = pd.read_csv(featured_file)
            logger.info(f"Final dataset shape: {final_data.shape}")
            logger.info(f"Columns: {list(final_data.columns)}")

            # Check for target variable
            if 'pct_of_cap' in final_data.columns:
                target_min = final_data['pct_of_cap'].min()
                target_max = final_data['pct_of_cap'].max()
                logger.info(f"Target variable 'pct_of_cap' range: [{target_min:.4f}, {target_max:.4f}]")

                # Validate range is reasonable (0 to 0.5)
                if target_min >= 0 and target_max <= 0.5:
                    logger.info("✓ Target variable range is reasonable")
                else:
                    logger.warning("⚠ Target variable range may be outside expected bounds")
            else:
                logger.error("✗ Target variable 'pct_of_cap' not found")
                return False

            # Check that we have both feature groups
            if len(final_data.attrs.get('group_a_features', [])) > 0:
                logger.info("✓ Group A features present")
            else:
                logger.warning("⚠ No Group A features found")

            if len(final_data.attrs.get('group_b_features', [])) > 0:
                logger.info("✓ Group B features present")
            else:
                logger.info("ℹ Group B features limited (expected without team-level data)")
        else:
            logger.error("✗ Featured file not found")
            return False

        logger.info("\n" + "="*60)
        logger.info("PIPELINE TEST COMPLETED SUCCESSFULLY ✓")
        logger.info("="*60)
        logger.info(f"Output files:")
        logger.info(f"  - Merged data: {merged_file}")
        logger.info(f"  - Featured data: {featured_file}")
        return True

    except Exception as e:
        logger.error(f"Pipeline test failed with error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)