"""Validated, hashable Phase Zero configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingConfig(StrictModel):
    algorithm: str
    runtime: str = "upstream"
    max_environment_steps: int = Field(gt=0)
    evaluation_interval: int = Field(gt=0)
    evaluation_episodes: int = Field(gt=0)
    minimum_success: float = Field(ge=0, le=1)


class BufferConfig(StrictModel):
    sequences: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    fixed_seeds: list[int]


class AnalysisConfig(StrictModel):
    primary_modules: list[str]
    explained_variance: float = Field(gt=0, le=1)
    null_replicates: int = Field(gt=0)
    bootstrap_replicates: int = Field(gt=0)
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


class Phase0Config(StrictModel):
    schema_version: int
    study: str
    artifact_root: Path
    seeds: list[int]
    environments: list[str]
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
