import pathlib

# pyclad import is in src/pyclad, so we need to adjust the import paths accordingly
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# Datasets
from pyclad.data.datasets.unsw_dataset import UnswDataset
from pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset
from pyclad.data.datasets.wind_energy_dataset import WindEnergyDataset
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset

# Scenarios
from pyclad.scenarios.concept_aware import ConceptAwareScenario

from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer

# Models
from pyclad.models.adapters.pyod_adapters import LocalOutlierFactorAdapter

# Strategies
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy
from pyclad.strategies.replay.candi import CandiStrategy

# Additional imports for replay strategies
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.selection.random import RandomSelection

# Callback and metrics
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.output.json_writer import JsonOutputWriter
import time
import sys
import os

# Configuration
DATASETS = {
    "wind": WindEnergyDataset,
    "unsw": UnswDataset,
    "energy": EnergyPlantsDataset,
    "nsl-kdd": NslKddDataset,
}

DATASET_TYPES = [
    "random_anomalies",
    "clustered_with_closest_assignment",
    "clustered_with_random_assignment",
]

max_size = 2000

STRATEGIES = {
    "watch_percentile": lambda model: CandiStrategy(
        model,
        max_buffer_size=max_size,
        threshold_ratio=0.5,
        warm_up_period=2,
        threshold_cal_index=1,
        resize_new_regime=True,
    ),
}


print("Setup complete.")

results_file = open("experiment_results.txt", "w")


def run_experiments():
    for dataset_name, dataset_class in DATASETS.items():
        for dataset_type in DATASET_TYPES:
            print(
                f"Running experiments for {dataset_name} - {dataset_type}",
                file=results_file,
            )
            try:
                dataset = dataset_class(dataset_type=dataset_type)
            except Exception as e:
                print(
                    f"Could not load dataset {dataset_name} with type {dataset_type}. Error: {e}",
                    file=results_file,
                )
                continue

            for strategy_name, strategy_builder in STRATEGIES.items():
                start_time = time.time()
                print(f"  with strategy: {strategy_name}", file=results_file)
                model = LocalOutlierFactorAdapter()
                strategy = strategy_builder(model)

                callbacks = [
                    ConceptMetricCallback(
                        base_metric=RocAuc(),
                        metrics=[
                            ContinualAverage(),
                            BackwardTransfer(),
                            ForwardTransfer(),
                        ],
                    ),
                    TimeEvaluationCallback(),
                    MemoryUsageCallback(),
                ]
                scenario = ConceptAwareScenario(
                    dataset, strategy=strategy, callbacks=callbacks
                )
                scenario.run()

                # callbacks[0].print_continual_average()
                continual_average = callbacks[0].get_continual_average()
                print(f"  Continual Average: {continual_average}", file=results_file)

                end_time = time.time()
                print(
                    f"  Time taken: {end_time - start_time:.2f} seconds",
                    file=results_file,
                )

            print("-" * 20, file=results_file)

        print(f"Finished experiments for {dataset_name}.", file=results_file)
        print("=" * 40, file=results_file)


run_experiments()
print("All experiments finished.", file=results_file)
