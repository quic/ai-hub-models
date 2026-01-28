# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioMapping,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.simulation.callback.multi_callback import MultiCallback
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import (
    KinematicBicycleModel,
)
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.two_stage_controller import (
    TwoStageController,
)
from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
    StepSimulationTimeController,
)
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
from PIL import Image

from qai_hub_models.models.statetransformer.model import (
    MODEL_ASSET_VERSION,
    MODEL_ID,
    STR_SOURCE_REPO_COMMIT,
    STR_SOURCE_REPOSITORY,
)
from qai_hub_models.models.statetransformer.util import save_raster
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    SourceAsRoot,
    load_yaml,
)
from qai_hub_models.utils.inference import OnDeviceModel


def deep_update(original: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively update a nested dictionary with values from another dictionary.

    This function traverses the `update` dictionary and merges its contents into the
    `original` dictionary. If a value in `update` is itself a dictionary, the function
    performs a deep merge into the corresponding dictionary in `original`.

    Parameters
    ----------
    original
        The original dictionary to be updated.
    update
        The dictionary containing updates to apply.

    Returns
    -------
    updated_dict
        The updated dictionary after applying the deep merge.
    """
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class StateTransformerApp:
    """
    Application wrapper for the StateTransformer model.

    This class provides functionality to initialize the model and extract model-ready
    input samples from NuPlan simulation scenarios.
    """

    def __init__(
        self,
        model: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
        | OnDeviceModel,
    ):
        """
        Initialize the StateTransformerApp.

        Parameters
        ----------
        model
            A callable model instance that takes three input tensors and returns a prediction.
        """
        self.model = model

    def extract_model_samples(
        self,
        model_path: str | CachedWebModelAsset | Path,
        data_path: str | CachedWebModelAsset | Path,
        map_path: str | CachedWebModelAsset | Path,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts model-ready input samples from NuPlan simulation scenarios.

        This method performs the following steps:
        1. Loads and filters scenario definitions.
        2. Configures planners with custom overrides.
        3. Simulates each scenario using a motion model and controller.
        4. Extracts planner inputs from the simulations.
        5. Converts planner inputs into model-compatible tensors.

        Parameters
        ----------
        model_path
            Path to the trained model checkpoint directory.
            Example structure:
                |-> checkpoint-66000/
                    |-> model weights and config files
        data_path
            Path to NuPlan scenario data directory.
            Example structure:
                |-> nuplan-v1.1_test/
                    |-> *.db files containing scenario logs
        map_path
            Path to map assets directory.
            Example structure:
                |-> maps/
                    |-> *.gpkg files for different regions
                    |-> nuplan-maps-v1.0.json metadata

        Returns
        -------
        high_res_raster
            High-resolution raster input.
        low_res_raster
            Low-resolution raster input.
        context_actions
            Temporal context actions.
        """
        if isinstance(model_path, CachedWebModelAsset):
            model_path = model_path.fetch(extract=True).joinpath("checkpoint-66000")
        if isinstance(map_path, CachedWebModelAsset):
            map_path = map_path.fetch(extract=True).joinpath("maps")
        if isinstance(data_path, CachedWebModelAsset):
            data_path = data_path.fetch(extract=True).joinpath("nuplan-v1.1_test")

        with SourceAsRoot(
            STR_SOURCE_REPOSITORY,
            STR_SOURCE_REPO_COMMIT,
            MODEL_ID,
            MODEL_ASSET_VERSION,
        ):
            # Load utility functions and scenario map
            from nuplan_simulation.common_utils import (
                get_pacifica_parameters,
                get_scenario_map,
            )

            # Paths to planner config and scenario filter YAMLs
            planner_config_path = f"{os.getcwd()}/tuplan_garage/planning/script/config/simulation/planner/str_closed_planner.yaml"
            scenario_filter_yaml = "nuplan_simulation/test_split_4.yaml"

            # Step 1: Build scenarios
            map_version = "nuplan-maps-v1.0"
            scenario_mapping = ScenarioMapping(
                scenario_map=get_scenario_map(), subsample_ratio_override=0.5
            )
            builder = NuPlanScenarioBuilder(
                str(data_path),
                str(map_path),
                None,
                None,
                map_version,
                scenario_mapping=scenario_mapping,
            )
            params = load_yaml(scenario_filter_yaml)
            scenario_filter = ScenarioFilter(**params)
            worker = SingleMachineParallelExecutor(use_process_pool=False)
            scenarios = builder.get_scenarios(scenario_filter, worker)

            # Step 2: Build planners with overrides
            params_planner = load_yaml(planner_config_path)

            pdm_speed_limit_fraction = "0.2 0.4 0.6 0.8 1.0"
            conservative_factor = 0.7
            comfort_weight = 10.0
            initstable_time = 8
            planner_name = "str_closed_planner"

            planner_override: dict[str, Any] = {
                "str_closed_planner": {
                    "idm_policies": {
                        "speed_limit_fraction": [
                            float(c) for c in pdm_speed_limit_fraction.split(" ")
                        ]
                    },
                    "str_generator": {"model_path": str(model_path)},
                    "conservative_factor": conservative_factor,
                    "comfort_weight": comfort_weight,
                    "initstable_time": initstable_time,
                }
            }
            deep_update(params_planner, planner_override)

            # Instantiate planners for each scenario
            planners = [
                hydra.utils.instantiate(params_planner)[planner_name] for _ in scenarios
            ]

            # Step 3: Build simulations
            tracker = LQRTracker(
                q_longitudinal=[10.0],
                r_longitudinal=[1.0],
                q_lateral=[1.0, 10.0, 0.0],
                r_lateral=[1.0],
                discretization_time=0.1,
                tracking_horizon=10,
                jerk_penalty=1e-4,
                curvature_rate_penalty=1e-2,
                stopping_proportional_gain=0.5,
                stopping_velocity=0.2,
            )
            motion_model = KinematicBicycleModel(get_pacifica_parameters())

            # Create ego controllers and observations
            ego_controllers = [
                TwoStageController(scenario, tracker, motion_model)
                for scenario in scenarios
            ]
            observations = [TracksObservation(scenario) for scenario in scenarios]

            # Create simulation setups
            simulation_setups = [
                SimulationSetup(
                    time_controller=StepSimulationTimeController(scenario),
                    observations=observation,
                    ego_controller=ego_controller,
                    scenario=scenario,
                )
                for scenario, observation, ego_controller in zip(
                    scenarios, observations, ego_controllers, strict=False
                )
            ]

            # Run simulations
            simulations = [
                Simulation(
                    simulation_setup=setup,
                    callback=MultiCallback([]),  # No callbacks needed
                )
                for setup in simulation_setups
            ]

            # Step 4: Get planner inputs
            for i in range(len(simulations)):
                sim_init = simulations[i].initialize()
                planners[i].initialize(sim_init)

            planner_inputs = [sim.get_planner_input() for sim in simulations]

            # Step 5: Convert planner inputs to model samples
            # Only return the first sample
            i = 0
            model_sample = planners[i]._str_generator._inputs_to_model_sample(
                history=planner_inputs[i].history,
                traffic_light_data=list(planner_inputs[i].traffic_light_data),
                map_name=planners[i]._map_api.map_name,
            )

        # Stack samples into batch format
        device = torch.device("cpu")
        return (
            torch.from_numpy(model_sample["high_res_raster"])
            .unsqueeze(0)
            .to(device, dtype=torch.float32),
            torch.from_numpy(model_sample["low_res_raster"])
            .unsqueeze(0)
            .to(device, dtype=torch.float32),
            torch.from_numpy(model_sample["context_actions"])
            .unsqueeze(0)
            .to(device, dtype=torch.float32),
        )

    def post_process(
        self,
        high_res_raster: torch.Tensor,
        low_res_raster: torch.Tensor,
        context_actions: torch.Tensor,
        prediction_generation: dict[str, np.ndarray],
    ) -> Image.Image:
        """
        Post-process model predictions and generate a visualization image.

        This method overlays the predicted trajectory (and optionally key points)
        on the high-resolution raster image using a visualization utility.

        Parameters
        ----------
        high_res_raster
            High-resolution BEV raster image of the scene.
            Shape: (batch_size, 224, 224, 58)
            Dtype: float32

        low_res_raster
            Low-resolution BEV raster image covering a larger spatial range.
            Shape: (batch_size, 224, 224, 58)
            Dtype: float32

        context_actions
            Past ego motion states or control context.
            Shape: (batch_size, 4, 7)
            Dtype: float32

        prediction_generation
            Dictionary containing raw prediction outputs from the model.
            Keys:
                - 'traj_logits': ndarray of shape (batch_size, 80, 4)
                Predicted trajectory logits.
                - 'key_points_logits' (optional): ndarray of shape (batch_size, N, 4)
                Predicted key points for trajectory refinement.

        Returns
        -------
        visualization
            A PIL Image object representing the high-resolution raster with predicted
            trajectory and key points overlaid.
        """
        prepared_data = {
            "high_res_raster": high_res_raster,
            "low_res_raster": low_res_raster,
            "context_actions": context_actions,
        }
        image_dictionary = save_raster(
            inputs=prepared_data,
            sample_index=0,
            prediction_trajectory_by_gen=prediction_generation["traj_logits"][0],
        )

        if image_dictionary is None:
            raise ValueError("save_raster returned None; nothing to render.")

        return Image.fromarray(image_dictionary["high_res_raster"].astype(np.uint8))

    def predict(
        self,
        high_res_raster: torch.Tensor,
        low_res_raster: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> Image.Image:
        """
        Generate trajectory predictions and return a visualization image.

        This method runs the model to produce raw predictions and then applies
        post-processing to render the predicted trajectory on the raster image.

        Parameters
        ----------
        high_res_raster
            High-resolution BEV raster image of the scene.
            Shape: (batch_size, 224, 224, 58)
            Dtype: float32
        low_res_raster
            Low-resolution BEV raster image covering a larger spatial range.
            Shape: (batch_size, 224, 224, 58)
            Dtype: float32
        context_actions
            Past ego motion states or control context.
            Shape: (batch_size, 4, 7)
            Dtype: float32

        Returns
        -------
        visualization
            A PIL Image object representing the high-resolution raster with predicted
            trajectory and key points overlaid.
        """
        traj_logits, traj_scores = self.model(
            high_res_raster, low_res_raster, context_actions
        )
        prediction_generation = {"traj_logits": traj_logits, "traj_scores": traj_scores}
        if isinstance(prediction_generation, dict):
            img: Image.Image = self.post_process(
                high_res_raster, low_res_raster, context_actions, prediction_generation
            )
        return img
