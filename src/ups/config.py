"""Validated, hashable Phase Zero configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The observed two-seed pilot used 0 and 1.  Keep the confirmation seeds
# disjoint so the final cohort is not a rerun of inspected random streams.
FULL_STUDY_SEEDS = (10, 11, 12, 13, 14)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingConfig(StrictModel):
    algorithm: str
    runtime: str = "upstream"
    max_environment_steps: int = Field(gt=0)
    evaluation_interval: int = Field(gt=0)
    evaluation_episodes: int = Field(gt=0)
    minimum_success: float = Field(ge=0, le=1)
    checkpoint_retention: int = Field(ge=20)
    checkpoint_save_interval_seconds: int = Field(gt=0)
    heartbeat_reporting_interval_seconds: int = Field(gt=0)


class BufferConfig(StrictModel):
    sequences: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    fixed_seeds: list[int]


class AnalysisConfig(StrictModel):
    primary_modules: list[str]
    explained_variance: float = Field(gt=0, le=1)
    null_replicates: int = Field(gt=0)
    bootstrap_replicates: int = Field(gt=0)
    practical_effect_size: float = Field(gt=0)
    leave_one_task_out: bool


class AdapterConfig(StrictModel):
    repositories: list[str]
    projections: list[str]


class GateConfig(StrictModel):
    immutable: bool
    null_effect_min_sd: float
    retention_median_min: float
    retention_ci_low_min: float
    action_kl_max: float
    action_agreement_min: float
    feature_cka_min: float
    normalized_value_rmse_max: float


class ExperimentalDesignConfig(StrictModel):
    training_seed_count: int = Field(gt=0)
    environment_families: list[str] = Field(min_length=1)
    evaluation_seed_policy: str


class Phase0Config(StrictModel):
    schema_version: int
    study: str
    artifact_root: Path
    seeds: list[int]
    environments: list[str]
    experimental_design: ExperimentalDesignConfig | None = None
    training: TrainingConfig
    evaluation_buffer: BufferConfig
    analysis: AnalysisConfig
    adapters: AdapterConfig
    gate: GateConfig

    @model_validator(mode="after")
    def fixed_seed_cardinality(self) -> Phase0Config:
        if len(self.evaluation_buffer.fixed_seeds) != len(self.environments):
            raise ValueError("one fixed evaluation seed is required per environment")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("training seeds must be unique")
        if len(set(self.environments)) != len(self.environments):
            raise ValueError("environments must be unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("training seeds must be non-negative")
        if any(seed < 0 for seed in self.evaluation_buffer.fixed_seeds):
            raise ValueError("fixed evaluation seeds must be non-negative")
        if self.study == "phase0":
            if tuple(self.seeds) != FULL_STUDY_SEEDS:
                raise ValueError(
                    "full study requires the five preset training seeds [10, 11, 12, 13, 14]"
                )
            if self.experimental_design is None:
                raise ValueError("full study requires explicit experimental_design metadata")
            if self.experimental_design.training_seed_count != len(self.seeds):
                raise ValueError("experimental design seed count must match training seeds")
        elif self.experimental_design is not None and (
            self.experimental_design.training_seed_count != len(self.seeds)
        ):
            raise ValueError("experimental design seed count must match training seeds")
        eval_episodes = self.training.evaluation_episodes
        eval_ranges = [
            range(seed, seed + eval_episodes) for seed in self.evaluation_buffer.fixed_seeds
        ]
        if any(set(self.seeds).intersection(eval_range) for eval_range in eval_ranges):
            raise ValueError("training and fixed evaluation seed ranges must be disjoint")
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def load_config(path: Path) -> Phase0Config:
    with path.open("rb") as stream:
        raw = yaml.safe_load(stream)
    return Phase0Config.model_validate(raw)
