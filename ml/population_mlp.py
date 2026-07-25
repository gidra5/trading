from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.func import functional_call, vmap

from mlp_model import ExposureMlp


class PopulationExposureMlp(nn.Module):
    """Stack identical ExposureMlp replicas behind one batched CUDA forward."""

    def __init__(self, prototype: ExposureMlp, population_size: int) -> None:
        super().__init__()
        if population_size < 1:
            raise ValueError("population size must be positive")
        self.population_size = population_size
        self.parameter_names = tuple(name for name, _ in prototype.named_parameters())
        self.buffer_names = tuple(name for name, _ in prototype.named_buffers())
        self.stacked_parameters = nn.ParameterList([
            nn.Parameter(torch.stack([
                parameter.detach().clone() for _ in range(population_size)
            ]))
            for parameter in prototype.parameters()
        ])
        for index, buffer in enumerate(prototype.buffers()):
            self.register_buffer(
                f"_stacked_buffer_{index}",
                torch.stack([buffer.detach().clone() for _ in range(population_size)]),
            )

        # The template only supplies module structure to functional_call. Its
        # meta tensors allocate no storage and are intentionally not registered
        # as a child module, so optimizers see only the stacked parameters.
        template = copy.deepcopy(prototype).to(device="meta")
        object.__setattr__(self, "_functional_template", template)

    def train(self, mode: bool = True) -> PopulationExposureMlp:
        super().train(mode)
        self._functional_template.train(mode)
        return self

    def forward(self, features: Tensor) -> Tensor:
        parameters = dict(zip(
            self.parameter_names,
            self.stacked_parameters,
            strict=True,
        ))
        buffers = {
            name: getattr(self, f"_stacked_buffer_{index}")
            for index, name in enumerate(self.buffer_names)
        }

        def apply_member(member_parameters, member_buffers):
            return functional_call(
                self._functional_template,
                (member_parameters, member_buffers),
                (features,),
            )

        # Separate processes previously started from the same seed and consumed
        # identical dropout masks. Preserve that paired-experiment property.
        return vmap(apply_member, randomness="same")(parameters, buffers)

    @torch.no_grad()
    def reset_from(self, prototype: ExposureMlp) -> None:
        prototype_parameters = dict(prototype.named_parameters())
        prototype_buffers = dict(prototype.named_buffers())
        for name, stacked in zip(
            self.parameter_names,
            self.stacked_parameters,
            strict=True,
        ):
            stacked.copy_(prototype_parameters[name].detach().unsqueeze(0))
        for index, name in enumerate(self.buffer_names):
            getattr(self, f"_stacked_buffer_{index}").copy_(
                prototype_buffers[name].detach().unsqueeze(0)
            )

    @torch.no_grad()
    def reset_from_state_dicts(
        self,
        states: list[dict[str, Tensor]],
    ) -> None:
        if len(states) != self.population_size:
            raise ValueError("one parent state is required for every population member")
        expected = set(self.parameter_names) | set(self.buffer_names)
        if any(set(state) != expected for state in states):
            raise ValueError("parent state does not match the population model")
        for name, stacked in zip(
            self.parameter_names,
            self.stacked_parameters,
            strict=True,
        ):
            values = [
                state[name].detach().to(device=stacked.device, dtype=stacked.dtype)
                for state in states
            ]
            stacked.copy_(torch.stack(values))
        for index, name in enumerate(self.buffer_names):
            stacked = getattr(self, f"_stacked_buffer_{index}")
            values = [
                state[name].detach().to(device=stacked.device, dtype=stacked.dtype)
                for state in states
            ]
            stacked.copy_(torch.stack(values))

    def member_state_dict(self, member: int) -> dict[str, Tensor]:
        if not 0 <= member < self.population_size:
            raise IndexError("population member is out of range")
        state = {
            name: parameter[member].detach().cpu().clone()
            for name, parameter in zip(
                self.parameter_names,
                self.stacked_parameters,
                strict=True,
            )
        }
        state.update({
            name: getattr(self, f"_stacked_buffer_{index}")[member].detach().cpu().clone()
            for index, name in enumerate(self.buffer_names)
        })
        return state


def population_clip_grad_norm_(
    parameters,
    population_size: int,
    max_norm: float,
) -> Tensor:
    """Clip every replica independently while retaining stacked parameters."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.zeros(population_size)
    for parameter in parameters:
        if parameter.shape[0] != population_size:
            raise ValueError(
                "every population parameter needs a leading member dimension"
            )
    member_gradients = [
        parameter.grad[member]
        for member in range(population_size)
        for parameter in parameters
    ]
    parameter_count = len(parameters)
    # Treat each member slice as a tensor-list entry. PyTorch's foreach kernels
    # then reduce and scale all 68 network tensors in two multi-tensor launches
    # instead of serializing one reduction and one multiply per parameter.
    partial_norms = torch.stack(
        torch._foreach_norm(member_gradients, 2),
    ).float().view(population_size, parameter_count)
    norm = partial_norms.square().sum(dim=1).sqrt()
    scale = (max_norm / norm.clamp_min(1e-12)).clamp(max=1.0)
    torch._foreach_mul_(
        member_gradients,
        [
            scale[member]
            for member in range(population_size)
            for _ in range(parameter_count)
        ],
    )
    return norm
