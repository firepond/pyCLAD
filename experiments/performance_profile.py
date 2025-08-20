import os

import sys
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pathlib

sys.path.append("../src/")

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
dataset = NslKddDataset(dataset_type="random_anomalies")

model = FogMLLOFModel()
# model = LOFModel(metric="euclidean")

max_size = 1000

strategy = ReplayEnhancedStrategy(
    model,
    AdaptiveBalancedReplayBuffer(selection_method=RandomSelection(), max_size=max_size),
)

# Run the experiment
callbacks = [
    ConceptMetricCallback(
        base_metric=RocAuc(),
        metrics=[ContinualAverage(), BackwardTransfer(), ForwardTransfer()],
    ),
    TimeEvaluationCallback(),
]
scenario = ConceptAwareScenario(dataset=dataset, strategy=strategy, callbacks=callbacks)
scenario.run()

callbacks[0].print_continual_average()
