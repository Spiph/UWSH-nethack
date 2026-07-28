"""Fixed-interface recurrent NLE policy with an explicit analysis registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

NLE_ACTIONS = (
    "north",
    "east",
    "south",
    "west",
    "northeast",
    "southeast",
    "southwest",
    "northwest",
)


class LoRALinear(nn.Linear):
    """Linear layer with an optional canonical low-rank update."""

    lora_a: nn.Parameter | None
    lora_b: nn.Parameter | None

    def __init__(self, in_features: int, out_features: int, rank: int = 0, alpha: float = 1.0):
        super().__init__(in_features, out_features)
        self.rank = rank
        self.alpha = alpha
        if rank:
            self.lora_a = nn.Parameter(torch.empty(rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.lora_a)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def composed_weight(self) -> Tensor:
        if self.lora_a is None or self.lora_b is None:
            return self.weight
        return self.weight + (self.alpha / self.rank) * (self.lora_b @ self.lora_a)

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.linear(input, self.composed_weight(), self.bias)


class LoRAConv2d(nn.Conv2d):
    """Convolution with a flattened-kernel LoRA update."""

    lora_a: nn.Parameter | None
    lora_b: nn.Parameter | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int = 0,
        alpha: float = 1.0,
    ):
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.rank = rank
        self.alpha = alpha
        flat = in_channels * kernel_size * kernel_size
        if rank:
            self.lora_a = nn.Parameter(torch.empty(rank, flat))
            self.lora_b = nn.Parameter(torch.zeros(out_channels, rank))
            nn.init.kaiming_uniform_(self.lora_a)
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def composed_weight(self) -> Tensor:
        if self.lora_a is None or self.lora_b is None:
            return self.weight
        update = (self.alpha / self.rank) * (self.lora_b @ self.lora_a)
        return self.weight + update.reshape_as(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return cast(Tensor, self._conv_forward(input, self.composed_weight(), self.bias))


@dataclass(frozen=True)
class PolicyOutput:
    logits: Tensor
    value: Tensor
    state: Tensor
    features: Tensor


class RecurrentNLEPolicy(nn.Module):
    """Glyph crop + BLStats encoder, 256-unit GRU, separate actor/critic heads."""

    def __init__(
        self,
        glyph_vocab: int = 6000,
        glyph_embedding: int = 32,
        blstats_size: int = 27,
        crop_size: int = 9,
        hidden_size: int = 256,
        lora_rank: int = 0,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.glyph_embedding = nn.Embedding(glyph_vocab, glyph_embedding)
        channels = (glyph_embedding, 64, 64, 64)
        self.cnn = nn.Sequential(
            *[
                layer
                for index in range(3)
                for layer in (
                    LoRAConv2d(channels[index], channels[index + 1], 3, lora_rank),
                    nn.ReLU(),
                )
            ]
        )
        self.encoder = LoRALinear(64 * crop_size * crop_size + blstats_size, hidden_size, lora_rank)
        self.core = nn.GRU(hidden_size, hidden_size)
        self.actor = LoRALinear(hidden_size, hidden_size, lora_rank)
        self.critic = LoRALinear(hidden_size, hidden_size, lora_rank)
        self.policy_head = LoRALinear(hidden_size, len(NLE_ACTIONS), lora_rank)
        self.value_head = LoRALinear(hidden_size, 1, lora_rank)

    @property
    def module_registry(self) -> dict[str, nn.Module]:
        return {
            "encoder.glyph_embedding": self.glyph_embedding,
            "encoder.cnn": self.cnn,
            "encoder.projection": self.encoder,
            "recurrent_core.gru": self.core,
            "actor.trunk": self.actor,
            "critic.trunk": self.critic,
            "heads.policy": self.policy_head,
            "heads.value": self.value_head,
        }

    def forward(
        self, glyphs_crop: Tensor, blstats: Tensor, state: Tensor | None = None
    ) -> PolicyOutput:
        # Inputs are [T,B,H,W] and [T,B,F].
        time, batch = glyphs_crop.shape[:2]
        embedded = self.glyph_embedding(glyphs_crop.long()).permute(0, 1, 4, 2, 3)
        visual = self.cnn(embedded.reshape(time * batch, *embedded.shape[2:]))
        visual = visual.reshape(time, batch, -1)
        features = torch.relu(self.encoder(torch.cat((visual, blstats.float()), dim=-1)))
        recurrent, state = self.core(features, state)
        actor = torch.relu(self.actor(recurrent))
        critic = torch.relu(self.critic(recurrent))
        return PolicyOutput(
            logits=self.policy_head(actor),
            value=self.value_head(critic).squeeze(-1),
            state=state,
            features=recurrent,
        )

    def export_onnx(self, path: str) -> None:
        class Wrapper(nn.Module):
            def __init__(self, policy: RecurrentNLEPolicy):
                super().__init__()
                self.policy = policy

            def forward(self, glyphs: Tensor, blstats: Tensor, state: Tensor) -> tuple[Tensor, ...]:
                output = self.policy(glyphs, blstats, state)
                return output.logits, output.value, output.state, output.features

        sample_glyphs = torch.zeros(1, 1, self.crop_size, self.crop_size, dtype=torch.long)
        sample_stats = torch.zeros(1, 1, self.encoder.in_features - 64 * self.crop_size**2)
        sample_state = torch.zeros(1, 1, self.core.hidden_size)
        torch.onnx.export(
            Wrapper(self),
            (sample_glyphs, sample_stats, sample_state),
            path,
            input_names=["glyphs_crop", "blstats", "state"],
            output_names=["logits", "value", "new_state", "features"],
            dynamic_axes={
                "glyphs_crop": {0: "time", 1: "batch"},
                "blstats": {0: "time", 1: "batch"},
            },
            opset_version=17,
        )


def checkpoint_payload(model: nn.Module, step: int, config_hash: str) -> dict[str, Any]:
    return {"model": model.state_dict(), "step": step, "config_hash": config_hash}
