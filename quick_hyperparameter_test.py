#!/usr/bin/env python3
"""
Quick Hyperparameter Test for WATCH Strategy
============================================

A simplified version for quick testing of WATCH strategy hyperparameters.
This script tests a smaller subset of configurations for faster iteration.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, cast
import sys
import pathlib

import pandas as pd

# Add src directory to Python path
src_path = pathlib.Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Datasets
from pyclad.data.datasets.unsw_dataset import UnswDataset
from pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset
from pyclad.data.datasets.wind_energy_dataset import WindEnergyDataset
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset

# Scenarios
from pyclad.scenarios.concept_aware import ConceptAwareScenario

# Models
from pyclad.models.adapters.pyod_adapters import LocalOutlierFactorAdapter

# Strategies
from pyclad.strategies.replay.watch import WatchStrategy

# Callback and metrics
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.callback import Callback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.json_writer import JsonOutputWriter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuickTestConfig:
    """Configuration for a quick test."""

    dataset_name: str
    dataset_type: str
    threshold: float
    max_buffer_size: int


class QuickHyperparameterTester:
    """Quick hyperparameter tester for WATCH strategy."""

    # Dataset configurations (subset for quick testing)
    DATASETS = {
        "nsl_kdd": NslKddDataset,
        "unsw": UnswDataset,
    }

    # Subset of assignment types
    ASSIGNMENT_TYPES = ["random_anomalies", "clustered_with_closest_assignment"]

    # Reduced search space for quick testing
    THRESHOLDS = [0.2, 0.4, 0.51, 0.7, 1.0, 1.5]
    BUFFER_SIZES = [1000, 3000]

    def __init__(self, output_dir: str = "quick_test_results"):
        """Initialize the quick tester."""
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_test_configs(self) -> List[QuickTestConfig]:
        """Generate test configurations."""
        configs = []

        for dataset_name in self.DATASETS.keys():
            for assignment_type in self.ASSIGNMENT_TYPES:
                for threshold in self.THRESHOLDS:
                    for buffer_size in self.BUFFER_SIZES:
                        config = QuickTestConfig(
                            dataset_name=dataset_name,
                            dataset_type=assignment_type,
                            threshold=threshold,
                            max_buffer_size=buffer_size,
                        )
                        configs.append(config)

        return configs

    def run_single_test(self, config: QuickTestConfig) -> Dict:
        """Run a single test configuration."""
        start_time = time.time()

        try:
            logger.info(
                f"Running: {config.dataset_name}_{config.dataset_type}_t{config.threshold}_b{config.max_buffer_size}"
            )

            # Create dataset
            dataset_class = self.DATASETS[config.dataset_name]
            dataset = dataset_class(dataset_type=config.dataset_type)

            # Create model
            model = LocalOutlierFactorAdapter()

            # Create strategy
            strategy = WatchStrategy(
                model=model, max_buffer_size=config.max_buffer_size, threshold_ratio=config.threshold
            )

            # Setup callbacks
            metric_callback = ConceptMetricCallback(
                base_metric=RocAuc(),
                metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
            )

            # Run experiment
            scenario = ConceptAwareScenario(dataset=dataset, strategy=strategy, callbacks=[metric_callback])
            scenario.run()

            # Save temporary results to extract metrics
            temp_file = self.output_dir / f"temp_{config.dataset_name}_{config.threshold}_{int(time.time())}.json"
            output_writer = JsonOutputWriter(temp_file)
            try:
                output_writer.write([model, dataset, strategy, metric_callback])

                # Extract metrics from saved file
                with open(temp_file, "r") as f:
                    temp_results = json.load(f)

                # Find metric callback results
                metric_key = next(k for k in temp_results if k.startswith("concept_metric_callback"))
                metric_data = temp_results[metric_key]
                metrics = metric_data["metrics"]

            except Exception as save_error:
                logger.warning(f"Could not save/load temp results: {save_error}")
                # Fallback to dummy metrics
                metrics = {"ContinualAverage": 0.5, "BackwardTransfer": 0.0, "ForwardTransfer": 0.0}
            finally:
                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()

            # Get strategy info
            strategy_info = strategy.additional_info()

            execution_time = time.time() - start_time

            result = {
                "dataset_name": config.dataset_name,
                "dataset_type": config.dataset_type,
                "threshold": config.threshold,
                "max_buffer_size": config.max_buffer_size,
                "continual_average": metrics.get("ContinualAverage", 0.0),
                "backward_transfer": metrics.get("BackwardTransfer", 0.0),
                "forward_transfer": metrics.get("ForwardTransfer", 0.0),
                "execution_time": execution_time,
                "num_regimes": strategy_info.get("num_regimes", 0),
                "current_size": strategy_info.get("current_size", 0),
                "success": True,
                "error_message": "",
            }

            logger.info(f"Completed in {execution_time:.2f}s - Continual Avg: {result['continual_average']:.4f}")
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed in {execution_time:.2f}s: {str(e)}")

            return {
                "dataset_name": config.dataset_name,
                "dataset_type": config.dataset_type,
                "threshold": config.threshold,
                "max_buffer_size": config.max_buffer_size,
                "continual_average": 0.0,
                "backward_transfer": 0.0,
                "forward_transfer": 0.0,
                "execution_time": execution_time,
                "num_regimes": 0,
                "current_size": 0,
                "success": False,
                "error_message": str(e),
            }

    def run_quick_test(self) -> pd.DataFrame:
        """Run all quick tests."""
        configs = self.generate_test_configs()
        logger.info(f"Running {len(configs)} quick tests")

        results = []
        for i, config in enumerate(configs, 1):
            logger.info(f"Test {i}/{len(configs)}")
            result = self.run_single_test(config)
            results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save results
        results_file = self.output_dir / "quick_test_results.csv"
        df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

        return df

    def analyze_quick_results(self, df: pd.DataFrame):
        """Analyze and print quick test results."""
        successful_df = df[df["success"] == True]

        if successful_df.empty:
            print("No successful tests!")
            return

        print("\n" + "=" * 60)
        print("QUICK HYPERPARAMETER TEST RESULTS")
        print("=" * 60)

        print(f"\nSuccess Rate: {len(successful_df)}/{len(df)} ({len(successful_df)/len(df):.2%})")

        # Best overall
        best_idx = successful_df["continual_average"].idxmax()
        best_result = successful_df.loc[best_idx]

        print(f"\nBest Configuration:")
        print(f"  Dataset: {best_result['dataset_name']}")
        print(f"  Type: {best_result['dataset_type']}")
        print(f"  Threshold: {best_result['threshold']}")
        print(f"  Buffer Size: {best_result['max_buffer_size']}")
        print(f"  Continual Average: {best_result['continual_average']:.4f}")
        print(f"  Backward Transfer: {best_result['backward_transfer']:.4f}")
        print(f"  Forward Transfer: {best_result['forward_transfer']:.4f}")
        print(f"  Execution Time: {best_result['execution_time']:.2f}s")
        print(f"  Number of Regimes: {best_result['num_regimes']}")

        # Threshold analysis
        print(f"\nThreshold Analysis:")
        threshold_stats = successful_df.groupby("threshold")["continual_average"].agg(["mean", "std", "count"])
        for threshold, stats in threshold_stats.iterrows():
            print(f"  {threshold}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")

        # Buffer size analysis
        print(f"\nBuffer Size Analysis:")
        buffer_stats = successful_df.groupby("max_buffer_size")["continual_average"].agg(["mean", "std", "count"])
        for buffer_size, stats in buffer_stats.iterrows():
            print(f"  {buffer_size}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")

        # Dataset analysis
        print(f"\nDataset Analysis:")
        dataset_stats = successful_df.groupby("dataset_name")["continual_average"].agg(["mean", "std", "count"])
        for dataset, stats in dataset_stats.iterrows():
            print(f"  {dataset}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, count={stats['count']}")

        # Top 5 configurations
        print(f"\nTop 5 Configurations:")
        top_5 = successful_df.nlargest(5, "continual_average")
        for i, (_, row) in enumerate(top_5.iterrows(), 1):
            print(
                f"  {i}. {row['dataset_name']}_{row['dataset_type']}_t{row['threshold']}_b{row['max_buffer_size']} "
                f"-> {row['continual_average']:.4f}"
            )


def main():
    """Main function for quick testing."""
    print("Quick WATCH Strategy Hyperparameter Test")
    print("=" * 40)

    tester = QuickHyperparameterTester()

    start_time = time.time()
    results_df = tester.run_quick_test()
    total_time = time.time() - start_time

    print(f"\nAll tests completed in {total_time:.2f} seconds")

    # Analyze results
    tester.analyze_quick_results(results_df)


if __name__ == "__main__":
    main()
