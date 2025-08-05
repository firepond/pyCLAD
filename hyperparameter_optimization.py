#!/usr/bin/env python3
"""
Hyperparameter Optimization for WATCH Strategy
==============================================

This script performs hyperparameter optimization for the WATCH strategy across
different datasets and dataset assignment types using multiprocessing for speed.

The script tests:
- 4 datasets: UNSW, NSL-KDD, Wind Energy, Energy Plants
- 3 assignment types: clustered_with_closest_assignment, random_anomalies, clustered_with_random_assignment
- Various threshold values for the WATCH strategy

Results are saved to JSON files and a summary CSV is generated.
"""

import json
import logging
import multiprocessing as mp
import pathlib
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Any, Optional
import sys

import numpy as np
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
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.json_writer import JsonOutputWriter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    dataset_name: str
    dataset_type: str
    threshold: float
    max_buffer_size: int
    experiment_id: str


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    config: ExperimentConfig
    continual_average: float
    backward_transfer: float
    forward_transfer: float
    execution_time: float
    memory_usage: float
    num_regimes: int
    success: bool
    error_message: str = ""


class HyperparameterOptimizer:
    """Hyperparameter optimizer for WATCH strategy."""

    # Dataset configurations
    DATASETS = {
        "unsw": UnswDataset,
        "nsl_kdd": NslKddDataset,
        "wind_energy": WindEnergyDataset,
        "energy_plants": EnergyPlantsDataset,
    }

    # Dataset assignment types
    ASSIGNMENT_TYPES = [
        "clustered_with_closest_assignment",
        "random_anomalies",
        "clustered_with_random_assignment",
    ]

    # Hyperparameter search space
    THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.51, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    BUFFER_SIZES = [1000, 2000, 3000, 5000]

    def __init__(
        self,
        output_dir: str = "hyperparameter_results",
        n_processes: Optional[int] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            output_dir: Directory to save results
            n_processes: Number of processes to use (defaults to CPU count - 1)
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        logger.info(f"Using {self.n_processes} processes for optimization")

    def generate_experiment_configs(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations."""
        configs = []

        for dataset_name, assignment_type, threshold, buffer_size in product(
            self.DATASETS.keys(),
            self.ASSIGNMENT_TYPES,
            self.THRESHOLDS,
            self.BUFFER_SIZES,
        ):
            experiment_id = (
                f"{dataset_name}_{assignment_type}_t{threshold}_b{buffer_size}"
            )

            config = ExperimentConfig(
                dataset_name=dataset_name,
                dataset_type=assignment_type,
                threshold=threshold,
                max_buffer_size=buffer_size,
                experiment_id=experiment_id,
            )
            configs.append(config)

        logger.info(f"Generated {len(configs)} experiment configurations")
        return configs

    def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment with the given configuration.

        Args:
            config: Experiment configuration

        Returns:
            ExperimentResult with the experiment outcomes
        """
        start_time = time.time()

        try:
            # Create dataset
            dataset_class = self.DATASETS[config.dataset_name]
            dataset = dataset_class(dataset_type=config.dataset_type)

            # Create model
            model = LocalOutlierFactorAdapter()

            # Create strategy
            strategy = WatchStrategy(
                model=model,
                max_buffer_size=config.max_buffer_size,
                threshold_ratio=config.threshold,
            )

            # Setup callbacks
            callbacks = [
                ConceptMetricCallback(
                    base_metric=RocAuc(),
                    metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
                ),
                TimeEvaluationCallback(),
                MemoryUsageCallback(),
            ]

            # Run experiment
            scenario = ConceptAwareScenario(
                dataset=dataset, strategy=strategy, callbacks=callbacks
            )
            scenario.run()

            # Save detailed results
            output_file = self.output_dir / f"{config.experiment_id}.json"
            output_writer = JsonOutputWriter(output_file)
            output_writer.write([model, dataset, strategy, *callbacks])

            # Extract metrics
            with open(output_file, "r") as f:
                results = json.load(f)

            # Find metric callback results
            metric_key = next(
                k for k in results if k.startswith("concept_metric_callback")
            )
            metric_data = results[metric_key]
            metrics = metric_data["metrics"]

            # Extract time and memory info
            time_key = next(
                k for k in results if k.startswith("time_evaluation_callback")
            )
            time_data = results[time_key]

            memory_key = next(
                k for k in results if k.startswith("memory_usage_callback")
            )
            memory_data = results[memory_key]

            # Get strategy info
            strategy_info = strategy.additional_info()

            execution_time = time.time() - start_time

            return ExperimentResult(
                config=config,
                continual_average=metrics.get("ContinualAverage", 0.0),
                backward_transfer=metrics.get("BackwardTransfer", 0.0),
                forward_transfer=metrics.get("ForwardTransfer", 0.0),
                execution_time=execution_time,
                memory_usage=memory_data.get("peak_memory_mb", 0.0),
                num_regimes=strategy_info.get("num_regimes", 0),
                success=True,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Experiment {config.experiment_id} failed: {str(e)}")

            return ExperimentResult(
                config=config,
                continual_average=0.0,
                backward_transfer=0.0,
                forward_transfer=0.0,
                execution_time=execution_time,
                memory_usage=0.0,
                num_regimes=0,
                success=False,
                error_message=str(e),
            )

    def run_optimization(self) -> pd.DataFrame:
        """
        Run the full hyperparameter optimization.

        Returns:
            DataFrame with all results
        """
        configs = self.generate_experiment_configs()

        logger.info(
            f"Starting hyperparameter optimization with {len(configs)} experiments"
        )

        # Run experiments in parallel
        with mp.Pool(processes=self.n_processes) as pool:
            results = pool.map(self.run_single_experiment, configs)

        # Convert results to DataFrame
        results_data = []
        for result in results:
            results_data.append(
                {
                    "dataset_name": result.config.dataset_name,
                    "dataset_type": result.config.dataset_type,
                    "threshold": result.config.threshold,
                    "max_buffer_size": result.config.max_buffer_size,
                    "experiment_id": result.config.experiment_id,
                    "continual_average": result.continual_average,
                    "backward_transfer": result.backward_transfer,
                    "forward_transfer": result.forward_transfer,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "num_regimes": result.num_regimes,
                    "success": result.success,
                    "error_message": result.error_message,
                }
            )

        df = pd.DataFrame(results_data)

        # Save summary results
        summary_file = self.output_dir / "hyperparameter_optimization_summary.csv"
        df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary results to {summary_file}")

        return df

    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze optimization results and find best configurations.

        Args:
            df: Results DataFrame

        Returns:
            Dictionary with analysis results
        """
        # Filter successful experiments
        successful_df = df[df["success"] == True].copy()

        if successful_df.empty:
            logger.warning("No successful experiments found!")
            return {}

        analysis = {}

        # Overall best configuration
        best_overall = successful_df.loc[successful_df["continual_average"].idxmax()]
        analysis["best_overall"] = {
            "config": best_overall.to_dict(),
            "metric": "continual_average",
        }

        # Best per dataset
        analysis["best_per_dataset"] = {}
        for dataset in successful_df["dataset_name"].unique():
            dataset_df = successful_df[successful_df["dataset_name"] == dataset]
            best_for_dataset = dataset_df.loc[dataset_df["continual_average"].idxmax()]
            analysis["best_per_dataset"][dataset] = best_for_dataset.to_dict()

        # Best per dataset type
        analysis["best_per_dataset_type"] = {}
        for dataset_type in successful_df["dataset_type"].unique():
            type_df = successful_df[successful_df["dataset_type"] == dataset_type]
            best_for_type = type_df.loc[type_df["continual_average"].idxmax()]
            analysis["best_per_dataset_type"][dataset_type] = best_for_type.to_dict()

        # Threshold analysis
        threshold_analysis = (
            successful_df.groupby("threshold")
            .agg(
                {
                    "continual_average": ["mean", "std", "max"],
                    "backward_transfer": ["mean", "std"],
                    "forward_transfer": ["mean", "std"],
                    "num_regimes": ["mean", "std"],
                    "execution_time": ["mean", "std"],
                }
            )
            .round(4)
        )

        analysis["threshold_analysis"] = threshold_analysis.to_dict()

        # Buffer size analysis
        buffer_analysis = (
            successful_df.groupby("max_buffer_size")
            .agg(
                {
                    "continual_average": ["mean", "std", "max"],
                    "backward_transfer": ["mean", "std"],
                    "forward_transfer": ["mean", "std"],
                    "memory_usage": ["mean", "std"],
                    "execution_time": ["mean", "std"],
                }
            )
            .round(4)
        )

        analysis["buffer_size_analysis"] = buffer_analysis.to_dict()

        # Success rate analysis
        total_experiments = len(df)
        successful_experiments = len(successful_df)
        analysis["success_rate"] = {
            "total_experiments": total_experiments,
            "successful_experiments": successful_experiments,
            "success_rate": successful_experiments / total_experiments,
        }

        # Save analysis
        analysis_file = self.output_dir / "optimization_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        logger.info(f"Saved analysis results to {analysis_file}")
        return analysis

    def print_summary(self, analysis: Dict[str, Any]):
        """Print a summary of the optimization results."""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("=" * 80)

        if not analysis:
            print("No successful experiments found!")
            return

        # Success rate
        success_info = analysis["success_rate"]
        print(
            f"\nSuccess Rate: {success_info['successful_experiments']}/{success_info['total_experiments']} "
            f"({success_info['success_rate']:.2%})"
        )

        # Best overall configuration
        best_config = analysis["best_overall"]["config"]
        print(f"\nBest Overall Configuration:")
        print(f"  Dataset: {best_config['dataset_name']}")
        print(f"  Assignment Type: {best_config['dataset_type']}")
        print(f"  Threshold: {best_config['threshold']}")
        print(f"  Buffer Size: {best_config['max_buffer_size']}")
        print(f"  Continual Average: {best_config['continual_average']:.4f}")
        print(f"  Backward Transfer: {best_config['backward_transfer']:.4f}")
        print(f"  Forward Transfer: {best_config['forward_transfer']:.4f}")

        # Best per dataset
        print(f"\nBest Configuration per Dataset:")
        for dataset, config in analysis["best_per_dataset"].items():
            print(f"  {dataset.upper()}:")
            print(
                f"    Threshold: {config['threshold']}, Buffer: {config['max_buffer_size']}"
            )
            print(f"    Continual Average: {config['continual_average']:.4f}")

        # Threshold recommendations
        threshold_analysis = analysis["threshold_analysis"]
        continual_avg_means = threshold_analysis["continual_average"]["mean"]
        best_threshold = max(continual_avg_means, key=continual_avg_means.get)
        print(
            f"\nRecommended Threshold: {best_threshold} (avg continual average: {continual_avg_means[best_threshold]:.4f})"
        )

        # Buffer size recommendations
        buffer_analysis = analysis["buffer_size_analysis"]
        buffer_avg_means = buffer_analysis["continual_average"]["mean"]
        best_buffer = max(buffer_avg_means, key=buffer_avg_means.get)
        print(
            f"Recommended Buffer Size: {best_buffer} (avg continual average: {buffer_avg_means[best_buffer]:.4f})"
        )


def main():
    """Main function to run the hyperparameter optimization."""
    print("Starting WATCH Strategy Hyperparameter Optimization")
    print("=" * 50)

    # Create optimizer
    optimizer = HyperparameterOptimizer(
        output_dir="watch_hyperparameter_results",
        n_processes=None,  # Use default (CPU count - 1)
    )

    # Run optimization
    start_time = time.time()
    results_df = optimizer.run_optimization()
    total_time = time.time() - start_time

    print(f"\nOptimization completed in {total_time:.2f} seconds")

    # Analyze results
    analysis = optimizer.analyze_results(results_df)

    # Print summary
    optimizer.print_summary(analysis)

    print(f"\nDetailed results saved to: {optimizer.output_dir}")
    print("Files created:")
    print(f"  - hyperparameter_optimization_summary.csv")
    print(f"  - optimization_analysis.json")
    print(f"  - Individual experiment results: *.json")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method("spawn", force=True)
    main()
