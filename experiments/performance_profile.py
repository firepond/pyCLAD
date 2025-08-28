import os
import time
import sys
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib

sys.path.append("../src/")

from pyclad.data import dataset
from pyclad.models.lof import LOFModel
from pyclad.strategies.replay.buffers.adaptive_balanced import (
    AdaptiveBalancedReplayBuffer,
)
from pyclad.strategies.replay.selection.random import RandomSelection

# set pyclad import path to "../src/pyclad"
import pyclad

# Datasets
from pyclad.data.datasets.unsw_dataset import UnswDataset
from pyclad.data.datasets.nsl_kdd_dataset import NslKddDataset
from pyclad.data.datasets.wind_energy_dataset import WindEnergyDataset
from pyclad.data.datasets.energy_plants_dataset import EnergyPlantsDataset


# Scenarios
from pyclad.scenarios.concept_aware import ConceptAwareScenario

# Models
from pyclad.models.adapters.pyod_adapters import LocalOutlierFactorAdapter

from pyclad.models.c_wrappers.fogml_lof import FogMLLOFModel


# Callback and metrics
from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.forward_transfer import ForwardTransfer
from pyclad.output.json_writer import JsonOutputWriter
from pyclad.strategies.replay.candi import CandiStrategy
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy

# dataset = WindEnergyDataset(dataset_type="random_anomalies")
# dataset = NslKddDataset(dataset_type="random_anomalies")
DATASETS = {
    "energy": EnergyPlantsDataset,
    "nsl-kdd": NslKddDataset,
    "unsw": UnswDataset,
    "wind": WindEnergyDataset,
}

# possible scenarios: "clustered_with_closest_assignment", "random_anomalies", "clustered_with_random_assignment"
DATASET_TYPE = [
    "random_anomalies",
    "clustered_with_closest_assignment",
    "clustered_with_random_assignment",
]

n_neighbors = 5
max_size = 1000
MODELS = {"FogMLLOFModel": FogMLLOFModel, "LOFModel": LOFModel}  # init in loop


results_file = open("results.txt", "w+")


for dataset_name, dataset_class in DATASETS.items():
    for model_name, model in MODELS.items():
        for dataset_type in DATASET_TYPE:
            print(
                f"Running experiment for {dataset_name} with {model_name} in scenario {dataset_type}"
            )
            results_file.write(
                f"Running experiment for {dataset_name} with {model_name} in scenario {dataset_type}\n"
            )
            if model_name == "FogMLLOFModel":
                model = FogMLLOFModel(k=n_neighbors)
            else:
                model = LOFModel(n_neighbors=n_neighbors)
            dataset = dataset_class(dataset_type=dataset_type)
            strategy = ReplayEnhancedStrategy(
                model,
                AdaptiveBalancedReplayBuffer(
                    selection_method=RandomSelection(), max_size=max_size
                ),
            )

            # Run the experiment
            callbacks = [
                ConceptMetricCallback(
                    base_metric=RocAuc(),
                    metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
                ),
                TimeEvaluationCallback(),
            ]
            scenario = ConceptAwareScenario(
                dataset=dataset, strategy=strategy, callbacks=callbacks
            )
            start_time = time.time()
            scenario.run()
            end_time = time.time()

            print(f"Time taken: {end_time - start_time:.2f} seconds")
            results_file.write(f"Time taken: {end_time - start_time:.2f} seconds\n")

            callbacks[0].print_continual_average()
            metrics = callbacks[0].get_metrics()
            for metric_name, metric_value in metrics.items():
                results_file.write(f"{metric_name}: {metric_value}\n")
            # flush
            results_file.flush()
            time.sleep(10)

results_file.close()
