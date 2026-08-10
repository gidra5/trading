from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Protocol
import types

import numpy as np

from forecast_model_zoo import MODEL_SPECS, snapshot_dir, source_root


QUANTILE_LEVELS = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)


@dataclass(frozen=True)
class QuantileForecast:
    model_id: str
    point: np.ndarray
    quantiles: np.ndarray
    quantile_levels: np.ndarray = field(default_factory=lambda: QUANTILE_LEVELS.copy())
    latency_seconds: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(self, expected_horizon: int | None = None) -> None:
        if self.point.ndim != 3:
            raise ValueError(f"point must have shape [batch, variate, horizon], got {self.point.shape}")
        expected = (*self.point.shape, len(self.quantile_levels))
        if self.quantiles.shape != expected:
            raise ValueError(f"quantiles must have shape {expected}, got {self.quantiles.shape}")
        if expected_horizon is not None and self.point.shape[-1] != expected_horizon:
            raise ValueError(f"expected horizon {expected_horizon}, got {self.point.shape[-1]}")
        if not np.isfinite(self.point).all() or not np.isfinite(self.quantiles).all():
            raise ValueError("forecast contains non-finite values")
        if np.any(np.diff(self.quantile_levels) <= 0):
            raise ValueError("quantile levels must be strictly increasing")

    @property
    def crossing_fraction(self) -> float:
        differences = np.diff(self.quantiles, axis=-1)
        return float(np.mean(differences < 0.0)) if differences.size else 0.0


class ForecastAdapter(Protocol):
    model_id: str

    def predict(self, contexts: np.ndarray, prediction_length: int) -> QuantileForecast: ...

    def close(self) -> None: ...


def _spec(model_id: str):
    return next(spec for spec in MODEL_SPECS if spec.id == model_id)


def _validate_contexts(contexts: np.ndarray) -> np.ndarray:
    values = np.asarray(contexts, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"contexts must have shape [batch, variate, time], got {values.shape}")
    if values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] < 2:
        raise ValueError(f"contexts are too small: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("contexts contain non-finite values")
    return values


def _median(quantiles: np.ndarray) -> np.ndarray:
    return np.asarray(quantiles[..., 4], dtype=np.float32)


class Chronos2Adapter:
    model_id = "chronos2"

    def __init__(
        self,
        repo_root: Path,
        device: str = "cuda",
        *,
        checkpoint: Path | None = None,
    ) -> None:
        import torch
        from chronos import Chronos2Pipeline

        self._torch = torch
        model_path = checkpoint.resolve() if checkpoint is not None else snapshot_dir(repo_root, _spec(self.model_id))
        if not model_path.is_dir():
            raise FileNotFoundError(f"Chronos-2 snapshot is missing: {model_path}")
        kwargs: dict[str, object] = {"device_map": device}
        self._pipeline = Chronos2Pipeline.from_pretrained(model_path, **kwargs)
        self._checkpoint = model_path

    def predict(self, contexts: np.ndarray, prediction_length: int) -> QuantileForecast:
        values = _validate_contexts(contexts)
        started = time.perf_counter()
        quantiles, _ = self._pipeline.predict_quantiles(
            values,
            prediction_length=prediction_length,
            quantile_levels=QUANTILE_LEVELS.tolist(),
            context_length=values.shape[-1],
            batch_size=max(1, min(256, values.shape[0] * values.shape[1])),
        )
        stacked = np.stack([item.numpy() for item in quantiles]).astype(np.float32, copy=False)
        result = QuantileForecast(
            model_id=self.model_id,
            point=_median(stacked),
            quantiles=stacked,
            latency_seconds=time.perf_counter() - started,
            metadata={
                "multivariate": True,
                "contextLength": values.shape[-1],
                "checkpoint": str(self._checkpoint).replace("\\", "/"),
            },
        )
        result.validate(prediction_length)
        return result

    def close(self) -> None:
        del self._pipeline
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class TiRex2Adapter:
    model_id = "tirex2"

    def __init__(
        self,
        repo_root: Path,
        device: str = "cuda",
        *,
        tta_sign_flip: bool | None = None,
        tta_diff: bool | None = None,
    ) -> None:
        import torch
        from tirex2 import load_model

        self._torch = torch
        self._tta_sign_flip = tta_sign_flip
        self._tta_diff = tta_diff
        self._native_cuda_fallback = sys.platform == "win32" and device.startswith("cuda")
        model_path = snapshot_dir(repo_root, _spec(self.model_id))
        if not model_path.is_dir():
            raise FileNotFoundError(f"TiRex-2 snapshot is missing: {model_path}")
        self._model = load_model(
            model_path,
            device="cpu" if self._native_cuda_fallback else device,
            use_flex_attention=False if sys.platform == "win32" else None,
        )
        if self._native_cuda_fallback:
            # TiRex-2's fused FlashRNN extension is not supported on Windows.
            # Building with device="cpu" selects its official device-agnostic
            # PyTorch kernels; moving the finished module still runs those
            # tensor operations on CUDA without changing checkpoint weights.
            self._model.model.to(device)
        if device.startswith("cuda"):
            torch.set_float32_matmul_precision("high")

    def predict(self, contexts: np.ndarray, prediction_length: int) -> QuantileForecast:
        from tirex2 import TimeseriesType

        values = _validate_contexts(contexts)
        timeseries = [
            TimeseriesType(
                target=self._torch.from_numpy(item),
                past_covariates=None,
                future_covariates=None,
            )
            for item in values
        ]
        kwargs = {}
        if self._tta_sign_flip is not None:
            kwargs["tta_sign_flip"] = self._tta_sign_flip
        if self._tta_diff is not None:
            kwargs["tta_diff"] = self._tta_diff
        started = time.perf_counter()
        forecast = self._model.forecast(
            timeseries,
            prediction_length=prediction_length,
            output_type="numpy",
            batch_size=max(1, min(64, len(timeseries))),
            **kwargs,
        )
        # Native layout is [batch][variate, quantile, horizon].
        stacked = np.stack(forecast).transpose(0, 1, 3, 2).astype(np.float32, copy=False)
        result = QuantileForecast(
            model_id=self.model_id,
            point=_median(stacked),
            quantiles=stacked,
            latency_seconds=time.perf_counter() - started,
            metadata={
                "multivariate": True,
                "contextLength": values.shape[-1],
                "ttaSignFlip": self._tta_sign_flip,
                "ttaDiff": self._tta_diff,
                "kernelBackend": "native-cuda" if self._native_cuda_fallback else "official-default",
            },
        )
        result.validate(prediction_length)
        return result

    def close(self) -> None:
        del self._model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


class FinCastAdapter:
    model_id = "fincast"

    def __init__(self, repo_root: Path, context_length: int, device: str = "cuda") -> None:
        import torch

        spec = _spec(self.model_id)
        source = source_root(repo_root, spec) / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        # FinCast's ffm/__init__.py eagerly imports a JAX-only covariate helper.
        # The released PyTorch decoder does not use it, so load that official
        # module through a minimal package shell instead of adding JAX to the
        # production inference environment.
        if "ffm" not in sys.modules:
            package = types.ModuleType("ffm")
            package.__path__ = [str(source / "ffm")]
            sys.modules["ffm"] = package
        from ffm.pytorch_patched_decoder_MOE import FFMConfig, PatchedTimeSeriesDecoder_MOE

        if context_length < 32 or context_length > 2_048 or context_length % 32 != 0:
            raise ValueError("FinCast context length must be a multiple of 32 from 32 through 2048")
        model_path = snapshot_dir(repo_root, spec) / "v1.pth"
        if not model_path.is_file():
            raise FileNotFoundError(f"FinCast snapshot is missing: {model_path}")
        backend = "gpu" if device.startswith("cuda") else "cpu"
        self._torch = torch
        self._device_type = "cuda" if backend == "gpu" else "cpu"
        self._device = torch.device(device if backend == "gpu" else "cpu")
        self._context_length = context_length
        config = FFMConfig(
            num_layers=50,
            num_heads=16,
            hidden_size=1280,
            intermediate_size=1280,
            patch_len=32,
            horizon_len=128,
            head_dim=80,
            num_experts=4,
            gating_top_n=2,
            use_positional_embedding=False,
        )
        self._model = PatchedTimeSeriesDecoder_MOE(config)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        normalized = {}
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in ("_orig_mod.module.", "_orig_mod.", "module."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    break
            normalized[normalized_key] = value
        self._model.load_state_dict(normalized, strict=True)
        self._model.to(self._device).eval()

    def _decode(
        self,
        input_ts,
        paddings,
        frequency,
        prediction_length: int,
        *,
        autocast_enabled: bool,
    ):
        with self._torch.inference_mode():
            with self._torch.autocast(
                device_type=self._device_type,
                dtype=self._torch.float16,
                enabled=autocast_enabled,
            ):
                _point, full_tensor = self._model.decode(
                    input_ts=input_ts,
                    paddings=paddings,
                    freq=frequency,
                    horizon_len=prediction_length,
                    output_patch_len=128,
                    max_len=self._context_length,
                    return_forecast_on_context=False,
                )
        return full_tensor.float().cpu().numpy()

    def predict(self, contexts: np.ndarray, prediction_length: int) -> QuantileForecast:
        values = _validate_contexts(contexts)
        if values.shape[-1] != self._context_length:
            raise ValueError(f"FinCast adapter expects context {self._context_length}, got {values.shape[-1]}")
        flat = values.reshape(-1, values.shape[-1])
        input_ts = self._torch.from_numpy(flat).to(self._device)
        paddings = self._torch.zeros(
            (len(flat), self._context_length + prediction_length),
            dtype=self._torch.float32,
            device=self._device,
        )
        frequency = self._torch.zeros((len(flat), 1), dtype=self._torch.long, device=self._device)
        started = time.perf_counter()
        autocast_enabled = self._device_type == "cuda"
        full = self._decode(
            input_ts,
            paddings,
            frequency,
            prediction_length,
            autocast_enabled=autocast_enabled,
        )
        fp32_fallback = False
        if not np.isfinite(full).all() and autocast_enabled:
            # Retry pathological batches one multivariate origin at a time in
            # FP32. This preserves the official weights/math and bounds peak
            # memory on 8 GB GPUs.
            fp32_fallback = True
            pieces = []
            variates = values.shape[1]
            for start in range(0, len(flat), variates):
                end = start + variates
                pieces.append(self._decode(
                    input_ts[start:end],
                    paddings[start:end],
                    frequency[start:end],
                    prediction_length,
                    autocast_enabled=False,
                ))
            full = np.concatenate(pieces, axis=0)
        # FinCast emits [flattened batch*variate, horizon, mean+9 quantiles].
        native_quantiles = full[..., 1:]
        stacked = native_quantiles.reshape(
            values.shape[0], values.shape[1], prediction_length, len(QUANTILE_LEVELS)
        ).astype(np.float32, copy=False)
        result = QuantileForecast(
            model_id=self.model_id,
            point=_median(stacked),
            quantiles=stacked,
            latency_seconds=time.perf_counter() - started,
            metadata={
                "multivariate": False,
                "contextLength": values.shape[-1],
                "fp32Fallback": fp32_fallback,
            },
        )
        result.validate(prediction_length)
        return result

    def close(self) -> None:
        del self._model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


def create_adapter(
    model_id: str,
    repo_root: Path,
    *,
    context_length: int,
    device: str,
    chronos_checkpoint: Path | None = None,
) -> ForecastAdapter:
    if model_id == "fincast":
        return FinCastAdapter(repo_root, context_length=context_length, device=device)
    if model_id == "tirex2":
        return TiRex2Adapter(repo_root, device=device)
    if model_id == "chronos2":
        return Chronos2Adapter(repo_root, device=device, checkpoint=chronos_checkpoint)
    raise ValueError(f"unknown model: {model_id}")
