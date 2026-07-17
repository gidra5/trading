#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string last_error;

struct VwKamaParams {
  int32_t efficiency_period;
  int32_t efficiency_volume_period;
  int32_t fast_period;
  int32_t slow_period;
  int32_t volume_period;
  int32_t rate_ema_samples;
  int32_t threshold_samples;
  int32_t acceleration_samples;
  int32_t distance_samples;
  int32_t ema_samples;
  int32_t rsi_samples;
  int32_t dmi_samples;
  int32_t mean_reversion_efficiency_period;
  int32_t mean_reversion_fast_period;
  int32_t mean_reversion_slow_period;
  int32_t mean_reversion_volatility_samples;
  int32_t deadband_mode;
  int32_t agreement_mode;
  int32_t rate_mode;
  int32_t threshold_noise_response;

  float power;
  float volume_cap;
  float volume_power;
  float efficiency_volume_power;
  float deadband_bps_hour;
  float hysteresis_release_ratio;
  float threshold_noise_multiplier;
  float threshold_inverse_max;
  float threshold_inverse_noise_scale;
  float buy_max_fraction;
  float sell_max_fraction;
  float buy_sizing_sigma;
  float sell_sizing_sigma;
  float confirmation_mix;
  float confirmation_min_quality;
  float confirmation_acceleration_weight;
  float confirmation_distance_weight;
  float confirmation_bias;
  float confirmation_ema_threshold;
  float confirmation_ema_weight;
  float confirmation_ema_gate_strength;
  float confirmation_rsi_threshold;
  float confirmation_rsi_weight;
  float confirmation_dmi_weight;
  float confirmation_adx_threshold;
  float mean_reversion_suppression_threshold;
  float mean_reversion_reversal_threshold;
  float signal_friction_fraction;
  float warmup_multiple;
};

static_assert(sizeof(VwKamaParams) == 196, "VwKamaParams ABI changed");

struct VwKamaResult {
  double state_credit;
  double timing_credit;
  double lag_p50_ms;
  double lag_p90_ms;
  double lag_p95_ms;
  double lag_median_signed_ms;
  double elapsed_ms;
  double distillation_weighted_cross_entropy;
  double distillation_weighted_oracle_entropy;
  double distillation_weight;
  double distillation_opportunity;
  int32_t state_count;
  int32_t signal_count;
  int32_t oracle_count;
  int32_t matched_count;
};

static_assert(sizeof(VwKamaResult) == 104, "VwKamaResult ABI changed");

struct DeviceResult {
  float state_credit;
  int32_t signal_count;
  double distillation_weighted_cross_entropy;
  double distillation_weighted_oracle_entropy;
  double distillation_weight;
  double distillation_opportunity;
};

struct Transition {
  int32_t index;
  int32_t state;
};

struct AlignmentScore {
  double credit = 0;
  int32_t count = 0;
  double absolute_lag_ms = 0;
  int32_t node = -1;
};

struct AlignmentNode {
  int32_t previous;
  int32_t candidate_index;
  int32_t oracle_index;
  double lag_ms;
  double timing_credit;
  AlignmentScore score;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t count) { allocate(count); }
  ~DeviceBuffer() { if (data_) cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void allocate(size_t count) {
    if (count == 0) return;
    const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
  }

  T* get() const { return data_; }

 private:
  T* data_ = nullptr;
};

void cuda_check(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
  }
}

__device__ __forceinline__ float clamp01(float value) {
  return fmaxf(0.0f, fminf(1.0f, value));
}

__device__ __forceinline__ float clamp_value(float value, float minimum, float maximum) {
  return fmaxf(minimum, fminf(maximum, value));
}

__device__ __forceinline__ float period_alpha(int period) {
  return 2.0f / (static_cast<float>(period) + 1.0f);
}

__device__ __forceinline__ float ema_update(float previous, float value, float alpha, float& delta) {
  if (previous == 0.0f) {
    delta = 0.0f;
    return value;
  }
  const float next = previous + alpha * (value - previous);
  delta = next - previous;
  return next;
}

__device__ __forceinline__ float smooth(float previous, float value, float alpha) {
  return previous == 0.0f ? value : previous + alpha * (value - previous);
}

__host__ __device__ __forceinline__ int state_from_code(uint8_t code) {
  return code == 1 ? 1 : code == 2 ? -1 : 0;
}

__device__ __forceinline__ uint32_t code_from_state(int state) {
  return state > 0 ? 1u : state < 0 ? 2u : 0u;
}

__device__ __forceinline__ float marked_exposure(
  int direction,
  float fraction,
  float anchor_price,
  float price
) {
  if (direction == 0 || fraction <= 0.0f) return 0.0f;
  if (fraction >= 1.0f) return static_cast<float>(direction > 0 ? 1 : -1);
  const float movement = price / anchor_price;
  const float equity = direction > 0
    ? 1.0f - fraction + fraction * movement
    : 1.0f + fraction - fraction * movement;
  if (equity <= 2.220446e-16f) return static_cast<float>(direction > 0 ? 1 : -1);
  return static_cast<float>(direction) * fraction * movement / equity;
}

__device__ __forceinline__ float agreement_value(
  int mode,
  int direction,
  float fraction,
  float anchor_price,
  float price
) {
  return mode == 1
    ? static_cast<float>(direction) * fraction
    : marked_exposure(direction, fraction, anchor_price, price);
}

__device__ __forceinline__ float agreement_credit(
  int mode,
  int direction,
  float candidate,
  int oracle
) {
  if (mode == 1) {
    const float confidence = clamp01(fabsf(candidate));
    if (direction == oracle) return direction == 0 ? 1.0f : confidence;
    return oracle == 0 ? 1.0f - confidence : 0.0f;
  }
  if (oracle > 0) return clamp01(candidate);
  if (oracle < 0) return clamp01(-candidate);
  return 1.0f - clamp01(fabsf(candidate));
}

__device__ __forceinline__ int candidate_state(
  float rate,
  int current,
  float threshold,
  const VwKamaParams& parameters
) {
  if (parameters.deadband_mode == 1) {
    return rate > threshold ? 1 : rate < -threshold ? -1 : current;
  }
  if (parameters.deadband_mode == 2 && current != 0) {
    const float release = threshold * clamp01(parameters.hysteresis_release_ratio);
    if ((rate > 0.0f) == (current > 0) && fabsf(rate) >= release) return current;
  }
  return rate > threshold ? 1 : rate < -threshold ? -1 : 0;
}

__device__ __forceinline__ bool ema_aligned(int direction, float rate, const VwKamaParams& p) {
  return static_cast<float>(direction) * rate >= -fmaxf(0.0f, p.confirmation_ema_threshold);
}

__device__ __forceinline__ float confirmation_quality(
  int direction,
  float acceleration,
  float overextension,
  float ema_rate,
  float rsi,
  float dmi,
  float adx,
  const VwKamaParams& p
) {
  const float mix = clamp01(p.confirmation_mix);
  if (mix == 0.0f) return 1.0f;
  const float ema_threshold = fmaxf(0.0f, p.confirmation_ema_threshold);
  const float ema_evidence = clamp_value(
    (static_cast<float>(direction) * ema_rate + ema_threshold) / fmaxf(1.0f, ema_threshold),
    -5.0f,
    5.0f
  );
  const float rsi_evidence = (
    static_cast<float>(direction) * (rsi - 50.0f) + fmaxf(0.0f, p.confirmation_rsi_threshold)
  ) / 10.0f;
  const float adx_threshold = fmaxf(0.0f, p.confirmation_adx_threshold);
  const float adx_trust = clamp01((adx - adx_threshold) / fmaxf(1.0f, 100.0f - adx_threshold));
  const float dmi_evidence = static_cast<float>(direction) * dmi / 100.0f * adx_trust;
  const float score = p.confirmation_bias
    + fmaxf(0.0f, p.confirmation_acceleration_weight) * static_cast<float>(direction) * acceleration
    - fmaxf(0.0f, p.confirmation_distance_weight) * static_cast<float>(direction) * overextension
    + fmaxf(0.0f, p.confirmation_ema_weight) * ema_evidence
    + fmaxf(0.0f, p.confirmation_rsi_weight) * rsi_evidence
    + fmaxf(0.0f, p.confirmation_dmi_weight) * dmi_evidence;
  const float logistic = 1.0f / (1.0f + expf(-clamp_value(score, -50.0f, 50.0f)));
  const float gate = ema_aligned(direction, ema_rate, p)
    ? 1.0f
    : 1.0f - clamp01(p.confirmation_ema_gate_strength);
  return (1.0f - mix + mix * logistic) * gate;
}

__device__ __forceinline__ float signal_fraction(int direction, float rate, const VwKamaParams& p) {
  const bool buy = direction > 0;
  const float maximum = clamp01(buy ? p.buy_max_fraction : p.sell_max_fraction);
  const float sigma = buy ? p.buy_sizing_sigma : p.sell_sizing_sigma;
  const float normalized = rate / fmaxf(2.220446e-16f, sigma);
  return maximum * expf(-0.5f * normalized * normalized);
}

__global__ void evaluate_kernel(
  const float* high,
  const float* low,
  const float* close,
  const float* volume,
  const uint8_t* oracle_codes,
  const float* value_means,
  const float* value_second_moments,
  const float* value_entropies,
  const float* value_weights,
  const float* value_opportunities,
  int candle_count,
  int score_start,
  float interval_ms,
  float oracle_friction,
  int value_grid_size,
  float value_grid_minimum,
  float value_grid_maximum,
  float strategy_sigma,
  const VwKamaParams* parameters,
  int candidate_count,
  float* changes,
  int change_stride,
  DeviceResult* results,
  const int32_t* transition_offsets,
  uint32_t* transitions,
  bool write_transitions
) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const VwKamaParams p = parameters[candidate];
  float* change_ring = changes + static_cast<size_t>(candidate) * change_stride * 2;
  float* mean_change_ring = change_ring + change_stride;

  const bool threshold_enabled = p.threshold_noise_response == 1
    ? p.threshold_inverse_max > 0.0f
    : p.threshold_noise_multiplier > 0.0f;
  const bool confirmation_enabled = p.confirmation_mix > 0.0f;
  const bool acceleration_enabled = confirmation_enabled && p.confirmation_acceleration_weight > 0.0f;
  const bool distance_enabled = confirmation_enabled && p.confirmation_distance_weight > 0.0f;
  const bool ema_enabled = (confirmation_enabled && p.confirmation_ema_weight > 0.0f)
    || p.confirmation_ema_gate_strength > 0.0f;
  const bool rsi_enabled = confirmation_enabled && p.confirmation_rsi_weight > 0.0f;
  const bool dmi_enabled = confirmation_enabled && p.confirmation_dmi_weight > 0.0f;
  const bool mean_reversion_enabled = p.mean_reversion_reversal_threshold > 0.0f;

  int base_warmup = max(max(p.efficiency_period + 1, p.slow_period), p.volume_period);
  if (p.efficiency_volume_power > 0.0f) base_warmup = max(base_warmup, p.efficiency_volume_period);
  int required = static_cast<int>(ceilf(static_cast<float>(base_warmup) * fmaxf(1.0f, p.warmup_multiple)));
  required = max(required, static_cast<int>(ceilf(p.rate_ema_samples * p.warmup_multiple)));
  if (threshold_enabled) required = max(required, static_cast<int>(ceilf(p.threshold_samples * p.warmup_multiple)));
  if (acceleration_enabled) required = max(required, static_cast<int>(ceilf(p.acceleration_samples * p.warmup_multiple)));
  if (distance_enabled) required = max(required, static_cast<int>(ceilf(p.distance_samples * p.warmup_multiple)));
  if (ema_enabled) required = max(required, static_cast<int>(ceilf(p.ema_samples * p.warmup_multiple)));
  if (rsi_enabled) required = max(required, static_cast<int>(ceilf(p.rsi_samples * p.warmup_multiple)));
  if (dmi_enabled) required = max(required, static_cast<int>(ceilf(p.dmi_samples * p.warmup_multiple * 2.0f)));
  if (mean_reversion_enabled) {
    required = max(required, static_cast<int>(ceilf(
      max(
        max(p.mean_reversion_efficiency_period + 1, p.mean_reversion_slow_period),
        max(p.volume_period, p.mean_reversion_volatility_samples)
      ) * p.warmup_multiple
    )));
    if (p.efficiency_volume_power > 0.0f) {
      required = max(required, static_cast<int>(ceilf(
        p.efficiency_volume_period * p.warmup_multiple
      )));
    }
  }
  const int feed_start = max(0, score_start - required);

  const float fast_alpha = period_alpha(p.fast_period);
  const float slow_alpha = period_alpha(p.slow_period);
  const float efficiency_volume_alpha = period_alpha(p.efficiency_volume_period);
  const float volume_alpha = period_alpha(p.volume_period);
  const float rate_ema_alpha = period_alpha(p.rate_ema_samples);
  const float threshold_alpha = 2.0f / (p.threshold_samples + 1.0f);
  const float acceleration_noise_alpha = 2.0f / (p.acceleration_samples + 1.0f);
  const float acceleration_alpha = acceleration_noise_alpha;
  const float distance_alpha = 2.0f / (p.distance_samples + 1.0f);
  const float ema_alpha = period_alpha(p.ema_samples);
  const float rsi_alpha = 1.0f / p.rsi_samples;
  const float dmi_alpha = 1.0f / p.dmi_samples;
  const float mean_fast_alpha = period_alpha(p.mean_reversion_fast_period);
  const float mean_slow_alpha = period_alpha(p.mean_reversion_slow_period);
  const float mean_variance_alpha = period_alpha(p.mean_reversion_volatility_samples);

  int change_count = 0;
  int change_position = 0;
  float change_sum = 0.0f;
  float change_noise = 0.0f;
  int mean_change_count = 0;
  int mean_change_position = 0;
  float mean_change_sum = 0.0f;
  float mean_change_noise = 0.0f;
  float previous_price = -1.0f;
  float efficiency_volume_ema = 0.0f;
  float volume_ema = 0.0f;
  float kama = 0.0f;
  float kama_delta = 0.0f;
  float rate_ema = 0.0f;
  float threshold_previous_rate = 0.0f;
  bool threshold_has_previous = false;
  float threshold_noise = 0.0f;
  float acceleration_previous_rate = 0.0f;
  bool acceleration_noise_has_previous = false;
  float acceleration_noise = 0.0f;
  float previous_rate = 0.0f;
  bool has_previous_rate = false;
  float smoothed_rate_change = 0.0f;
  float distance_noise = 0.0f;
  float slow_ema = 0.0f;
  float slow_ema_delta = 0.0f;
  float rsi_previous_price = 0.0f;
  bool rsi_has_previous = false;
  float rsi_gain = 0.0f;
  float rsi_loss = 0.0f;
  float previous_high = 0.0f;
  float previous_low = 0.0f;
  float previous_close = 0.0f;
  bool dmi_has_previous = false;
  float true_range_ema = 0.0f;
  float positive_movement_ema = 0.0f;
  float negative_movement_ema = 0.0f;
  float adx = 0.0f;
  float mean_kama = 0.0f;
  float variance = 0.0f;
  int current = 0;
  int trend = 0;
  float target_fraction = 0.0f;
  float anchor_price = close[feed_start];
  float last_signal_price = 0.0f;
  bool has_last_signal = false;
  float state_credit = 0.0f;
  int signal_count = 0;
  double distillation_weighted_cross_entropy = 0.0;
  double distillation_weighted_oracle_entropy = 0.0;
  double distillation_weight = 0.0;
  double distillation_opportunity = 0.0;
  const bool distillation_enabled = value_grid_size >= 3
    && strategy_sigma > 0.0f
    && value_means
    && value_second_moments
    && value_entropies
    && value_weights
    && value_opportunities;

  for (int index = feed_start; index < candle_count; ++index) {
    const float price = close[index];
    const float candle_volume = volume[index];

    float efficiency_relative_volume = 1.0f;
    if (p.efficiency_volume_power > 0.0f && candle_volume > 0.0f) {
      float unused_delta = 0.0f;
      efficiency_volume_ema = ema_update(
        efficiency_volume_ema,
        candle_volume,
        efficiency_volume_alpha,
        unused_delta
      );
      efficiency_relative_volume = candle_volume / efficiency_volume_ema;
    }
    if (previous_price > 0.0f) {
      const float change = (price - previous_price) * (
        p.efficiency_volume_power == 0.0f
          ? 1.0f
          : powf(efficiency_relative_volume, p.efficiency_volume_power)
      );
      if (change_count >= p.efficiency_period) {
        const float removed = change_ring[change_position];
        change_sum -= removed;
        change_noise -= fabsf(removed);
      }
      change_ring[change_position] = change;
      change_position = (change_position + 1) % p.efficiency_period;
      change_count = min(change_count + 1, p.efficiency_period);
      change_sum += change;
      change_noise += fabsf(change);
      if (mean_reversion_enabled) {
        if (mean_change_count >= p.mean_reversion_efficiency_period) {
          const float removed = mean_change_ring[mean_change_position];
          mean_change_sum -= removed;
          mean_change_noise -= fabsf(removed);
        }
        mean_change_ring[mean_change_position] = change;
        mean_change_position = (mean_change_position + 1) % p.mean_reversion_efficiency_period;
        mean_change_count = min(mean_change_count + 1, p.mean_reversion_efficiency_period);
        mean_change_sum += change;
        mean_change_noise += fabsf(change);
      }
    }
    previous_price = price;
    const float efficiency_ratio = change_count >= p.efficiency_period && change_noise > 0.0f
      ? clamp01(fabsf(change_sum) / change_noise)
      : 0.0f;
    const float mean_efficiency_ratio = mean_reversion_enabled
      && mean_change_count >= p.mean_reversion_efficiency_period
      && mean_change_noise > 0.0f
      ? clamp01(fabsf(mean_change_sum) / mean_change_noise)
      : 0.0f;

    const float relative_volume = candle_volume > 0.0f && volume_ema > 0.0f
      ? clamp_value(candle_volume / volume_ema, 0.0f, p.volume_cap)
      : 1.0f;
    const float effective_efficiency = clamp01(
      efficiency_ratio * powf(relative_volume, p.volume_power)
    );
    const float smoothing = slow_alpha + effective_efficiency * (fast_alpha - slow_alpha);
    const float adaptive_alpha = clamp01(powf(smoothing, p.power));
    kama = ema_update(kama, price, adaptive_alpha, kama_delta);
    if (mean_reversion_enabled) {
      const float mean_effective_efficiency = clamp01(
        mean_efficiency_ratio * powf(relative_volume, p.volume_power)
      );
      const float mean_smoothing = mean_slow_alpha
        + mean_effective_efficiency * (mean_fast_alpha - mean_slow_alpha);
      const float mean_adaptive_alpha = clamp01(powf(mean_smoothing, p.power));
      float unused_delta = 0.0f;
      mean_kama = ema_update(mean_kama, price, mean_adaptive_alpha, unused_delta);
    }
    if (candle_volume > 0.0f) {
      float unused_delta = 0.0f;
      volume_ema = ema_update(volume_ema, candle_volume, volume_alpha, unused_delta);
    }
    const float previous_kama = kama - kama_delta;
    const float relative_rate = kama > 0.0f
      ? kama_delta / kama * 10000.0f * 3600000.0f / interval_ms
      : 0.0f;
    const float log_rate = kama > 0.0f && previous_kama > 0.0f
      ? logf(kama / previous_kama) * 10000.0f * 3600000.0f / interval_ms
      : 0.0f;
    const float raw_rate = p.rate_mode == 1 ? log_rate : relative_rate;
    float unused_rate_delta = 0.0f;
    rate_ema = ema_update(rate_ema, raw_rate, rate_ema_alpha, unused_rate_delta);
    const float rate = rate_ema;

    if (threshold_enabled) {
      if (threshold_has_previous) {
        const float change = fabsf(rate - threshold_previous_rate);
        threshold_noise += threshold_alpha * (change - threshold_noise);
      }
      threshold_previous_rate = rate;
      threshold_has_previous = true;
    }

    float acceleration = 0.0f;
    if (acceleration_enabled) {
      if (acceleration_noise_has_previous) {
        const float change = fabsf(rate - acceleration_previous_rate);
        acceleration_noise += acceleration_noise_alpha * (change - acceleration_noise);
      }
      acceleration_previous_rate = rate;
      acceleration_noise_has_previous = true;
      const float rate_change = has_previous_rate ? rate - previous_rate : 0.0f;
      smoothed_rate_change += acceleration_alpha * (rate_change - smoothed_rate_change);
      previous_rate = rate;
      has_previous_rate = true;
      acceleration = acceleration_noise > 2.220446e-16f
        ? smoothed_rate_change / acceleration_noise
        : 0.0f;
    }

    float overextension = 0.0f;
    if (distance_enabled) {
      const float distance = kama > 0.0f ? (price / kama - 1.0f) * 10000.0f : 0.0f;
      distance_noise += distance_alpha * (fabsf(distance) - distance_noise);
      overextension = distance_noise > 2.220446e-16f ? distance / distance_noise : 0.0f;
    }

    float ema_rate = 0.0f;
    if (ema_enabled) {
      slow_ema = ema_update(slow_ema, price, ema_alpha, slow_ema_delta);
      ema_rate = slow_ema > 0.0f
        ? slow_ema_delta / slow_ema * 10000.0f * 3600000.0f / interval_ms
        : 0.0f;
    }

    float rsi = 50.0f;
    if (rsi_enabled) {
      if (!rsi_has_previous) {
        rsi_previous_price = price;
        rsi_has_previous = true;
      } else {
        const float price_change = price - rsi_previous_price;
        rsi_previous_price = price;
        float unused_delta = 0.0f;
        rsi_gain = ema_update(rsi_gain, fmaxf(0.0f, price_change), rsi_alpha, unused_delta);
        rsi_loss = ema_update(rsi_loss, fmaxf(0.0f, -price_change), rsi_alpha, unused_delta);
      }
      if (rsi_loss <= 0.0f) rsi = rsi_gain > 0.0f ? 100.0f : 50.0f;
      else rsi = 100.0f - 100.0f / (1.0f + rsi_gain / rsi_loss);
    }

    float dmi_direction = 0.0f;
    if (dmi_enabled) {
      if (dmi_has_previous) {
        const float up = high[index] - previous_high;
        const float down = previous_low - low[index];
        const float true_range = fmaxf(
          high[index] - low[index],
          fmaxf(fabsf(high[index] - previous_close), fabsf(low[index] - previous_close))
        );
        true_range_ema = smooth(true_range_ema, true_range, dmi_alpha);
        positive_movement_ema = smooth(
          positive_movement_ema,
          up > down && up > 0.0f ? up : 0.0f,
          dmi_alpha
        );
        negative_movement_ema = smooth(
          negative_movement_ema,
          down > up && down > 0.0f ? down : 0.0f,
          dmi_alpha
        );
        const float positive = true_range_ema > 0.0f
          ? 100.0f * positive_movement_ema / true_range_ema
          : 0.0f;
        const float negative = true_range_ema > 0.0f
          ? 100.0f * negative_movement_ema / true_range_ema
          : 0.0f;
        const float direction_sum = positive + negative;
        const float dx = direction_sum > 0.0f
          ? 100.0f * fabsf(positive - negative) / direction_sum
          : 0.0f;
        adx = smooth(adx, dx, dmi_alpha);
        dmi_direction = positive - negative;
      }
      previous_high = high[index];
      previous_low = low[index];
      previous_close = price;
      dmi_has_previous = true;
    }

    float normalized_mean_distance = 0.0f;
    if (mean_reversion_enabled) {
      float unused_delta = 0.0f;
      const float signed_distance = mean_kama > 0.0f ? price / mean_kama - 1.0f : 0.0f;
      variance = ema_update(
        variance,
        fmaxf(2.220446e-16f, signed_distance * signed_distance),
        mean_variance_alpha,
        unused_delta
      );
      normalized_mean_distance = fabsf(signed_distance)
        / fmaxf(2.220446e-16f, sqrtf(fmaxf(0.0f, variance)));
    }

    const float threshold_adjustment = p.threshold_noise_response == 1
      ? fmaxf(0.0f, p.threshold_inverse_max) / (
        1.0f + fmaxf(0.0f, threshold_noise) / fmaxf(1e-12f, p.threshold_inverse_noise_scale)
      )
      : threshold_noise * fmaxf(0.0f, p.threshold_noise_multiplier);
    const float threshold = p.deadband_bps_hour + threshold_adjustment;
    const int previous_trend = trend;
    trend = candidate_state(rate, trend, threshold, p);
    const bool source_edge = trend != previous_trend;
    const int proposed = !mean_reversion_enabled || trend == 0
      ? trend
      : normalized_mean_distance >= p.mean_reversion_reversal_threshold
        ? -trend
        : normalized_mean_distance >= p.mean_reversion_suppression_threshold ? 0 : trend;
    const bool aligned = proposed == 0 || ema_aligned(proposed, ema_rate, p);
    const int desired = proposed != 0
      && clamp01(p.confirmation_ema_gate_strength) == 1.0f
      && !aligned
      ? 0
      : proposed;
    const float confirmation = desired == 0
      ? 1.0f
      : confirmation_quality(
          desired,
          acceleration,
          overextension,
          ema_rate,
          rsi,
          dmi_direction,
          adx,
          p
        );
    const float quality = desired == 0 ? 1.0f : confirmation;

    if (
      source_edge
      && desired != current
      && (desired == 0 || quality > 0.0f)
      && quality >= clamp01(p.confirmation_min_quality)
      && (!has_last_signal || oracle_friction <= 0.0f || p.signal_friction_fraction <= 0.0f
        || fabsf(price - last_signal_price)
          > last_signal_price * oracle_friction * p.signal_friction_fraction)
    ) {
      const float next_fraction = desired == 0 ? 0.0f : signal_fraction(desired, rate, p) * quality;
      if (index >= score_start) {
        if (write_transitions) {
          transitions[transition_offsets[candidate] + signal_count] =
            (static_cast<uint32_t>(index) << 2u) | code_from_state(desired);
        }
        ++signal_count;
      }
      current = desired;
      target_fraction = next_fraction;
      anchor_price = price;
      last_signal_price = price;
      has_last_signal = true;
    }

    if (index >= score_start && !write_transitions) {
      const float exposure = agreement_value(
        p.agreement_mode,
        current,
        target_fraction,
        anchor_price,
        price
      );
      state_credit += agreement_credit(
        p.agreement_mode,
        current,
        exposure,
        state_from_code(oracle_codes[index])
      );
      if (distillation_enabled) {
        const float center = clamp_value(exposure, value_grid_minimum, value_grid_maximum);
        const float inverse_two_variance = 1.0f / (2.0f * strategy_sigma * strategy_sigma);
        float maximum_log_weight = -3.402823466e38F;
        for (int grid_index = 0; grid_index < value_grid_size; ++grid_index) {
          const float grid_exposure = value_grid_minimum
            + static_cast<float>(grid_index) / static_cast<float>(value_grid_size - 1)
              * (value_grid_maximum - value_grid_minimum);
          const float difference = grid_exposure - center;
          maximum_log_weight = fmaxf(
            maximum_log_weight,
            -difference * difference * inverse_two_variance
          );
        }
        float normalizer = 0.0f;
        for (int grid_index = 0; grid_index < value_grid_size; ++grid_index) {
          const float grid_exposure = value_grid_minimum
            + static_cast<float>(grid_index) / static_cast<float>(value_grid_size - 1)
              * (value_grid_maximum - value_grid_minimum);
          const float difference = grid_exposure - center;
          normalizer += expf(
            -difference * difference * inverse_two_variance - maximum_log_weight
          );
        }
        const float log_normalizer = maximum_log_weight + logf(normalizer);
        const float expected_squared_distance = fmaxf(
          0.0f,
          value_second_moments[index]
            - 2.0f * center * value_means[index]
            + center * center
        );
        const float cross_entropy = log_normalizer
          + expected_squared_distance * inverse_two_variance;
        const float weight = value_weights[index];
        distillation_weighted_cross_entropy += static_cast<double>(weight * cross_entropy);
        distillation_weighted_oracle_entropy += static_cast<double>(weight * value_entropies[index]);
        distillation_weight += static_cast<double>(weight);
        distillation_opportunity += static_cast<double>(value_opportunities[index]);
      }
    }
  }

  if (!write_transitions) {
    results[candidate].state_credit = state_credit;
    results[candidate].signal_count = signal_count;
    results[candidate].distillation_weighted_cross_entropy = distillation_weighted_cross_entropy;
    results[candidate].distillation_weighted_oracle_entropy = distillation_weighted_oracle_entropy;
    results[candidate].distillation_weight = distillation_weight;
    results[candidate].distillation_opportunity = distillation_opportunity;
  }
}

int compare_alignment(const AlignmentScore& left, const AlignmentScore& right) {
  const double tolerance = std::numeric_limits<double>::epsilon() * 32.0
    * std::max({1.0, std::abs(left.credit), std::abs(right.credit)});
  if (left.credit > right.credit + tolerance) return 1;
  if (right.credit > left.credit + tolerance) return -1;
  if (left.count != right.count) return left.count > right.count ? 1 : -1;
  if (left.absolute_lag_ms != right.absolute_lag_ms) {
    return left.absolute_lag_ms < right.absolute_lag_ms ? 1 : -1;
  }
  return 0;
}

AlignmentScore query_alignment(const std::vector<AlignmentScore>& tree, int end) {
  AlignmentScore best;
  for (int index = end; index > 0; index -= index & -index) {
    if (compare_alignment(tree[index], best) > 0) best = tree[index];
  }
  return best;
}

void update_alignment(std::vector<AlignmentScore>& tree, int oracle_index, const AlignmentScore& score) {
  for (int index = oracle_index + 1; index < static_cast<int>(tree.size()); index += index & -index) {
    if (compare_alignment(score, tree[index]) > 0) tree[index] = score;
  }
}

double percentile(std::vector<double> values, double quantile) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const double index = (values.size() - 1) * quantile;
  const size_t lower = static_cast<size_t>(std::floor(index));
  const double fraction = index - lower;
  const double upper = values[std::min(lower + 1, values.size() - 1)];
  return values[lower] * (1.0 - fraction) + upper * fraction;
}

struct AlignmentResult {
  double credit;
  int matched_count;
  std::vector<double> signed_lags;
};

AlignmentResult align_transitions(
  const std::vector<Transition>& candidate,
  const std::vector<Transition>& oracle,
  const double* close_times,
  double match_window_ms,
  double timing_half_life_ms
) {
  std::vector<std::vector<std::pair<int32_t, double>>> by_state(3);
  for (int index = 0; index < static_cast<int>(oracle.size()); ++index) {
    const int bucket = oracle[index].state + 1;
    by_state[bucket].push_back({index, close_times[oracle[index].index]});
  }
  std::vector<AlignmentScore> tree(oracle.size() + 1);
  std::vector<AlignmentNode> nodes;
  for (int candidate_index = 0; candidate_index < static_cast<int>(candidate.size()); ++candidate_index) {
    const Transition& item = candidate[candidate_index];
    const double time = close_times[item.index];
    const auto& targets = by_state[item.state + 1];
    const auto first = std::lower_bound(
      targets.begin(),
      targets.end(),
      time - match_window_ms,
      [](const auto& target, double value) { return target.second < value; }
    );
    const auto end = std::upper_bound(
      targets.begin(),
      targets.end(),
      time + match_window_ms,
      [](double value, const auto& target) { return value < target.second; }
    );
    struct Pending {
      int oracle_index;
      double lag_ms;
      double timing_credit;
      AlignmentScore previous;
      AlignmentScore score;
    };
    std::vector<Pending> pending;
    for (auto target = first; target != end; ++target) {
      const double lag_ms = time - target->second;
      const double timing_credit = std::exp(-std::log(2.0) * std::abs(lag_ms) / timing_half_life_ms);
      const AlignmentScore previous = query_alignment(tree, target->first);
      AlignmentScore score {
        previous.credit + timing_credit,
        previous.count + 1,
        previous.absolute_lag_ms + std::abs(lag_ms),
        -1,
      };
      if (compare_alignment(score, query_alignment(tree, target->first + 1)) <= 0) continue;
      pending.push_back({target->first, lag_ms, timing_credit, previous, score});
    }
    for (const Pending& item_pending : pending) {
      const int node_index = static_cast<int>(nodes.size());
      AlignmentScore score = item_pending.score;
      score.node = node_index;
      nodes.push_back({
        item_pending.previous.node,
        candidate_index,
        item_pending.oracle_index,
        item_pending.lag_ms,
        item_pending.timing_credit,
        score,
      });
      update_alignment(tree, item_pending.oracle_index, score);
    }
  }
  const AlignmentScore best = query_alignment(tree, oracle.size());
  std::vector<double> signed_lags;
  for (int node = best.node; node >= 0; node = nodes[node].previous) {
    signed_lags.push_back(nodes[node].lag_ms);
  }
  std::reverse(signed_lags.begin(), signed_lags.end());
  return {best.credit, static_cast<int>(signed_lags.size()), std::move(signed_lags)};
}

}  // namespace

extern "C" int vw_kama_cuda_device_count() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) {
    last_error = cudaGetErrorString(error);
    return -1;
  }
  last_error.clear();
  return count;
}

extern "C" const char* vw_kama_cuda_device_name(int device) {
  static thread_local std::string name;
  cudaDeviceProp properties {};
  const cudaError_t error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    last_error = cudaGetErrorString(error);
    return nullptr;
  }
  name = properties.name;
  last_error.clear();
  return name.c_str();
}

extern "C" const char* vw_kama_cuda_last_error() {
  return last_error.c_str();
}

extern "C" int vw_kama_cuda_params_size() {
  return sizeof(VwKamaParams);
}

extern "C" int vw_kama_cuda_result_size() {
  return sizeof(VwKamaResult);
}

extern "C" int vw_kama_cuda_evaluate(
  const double* close_times,
  const double* high,
  const double* low,
  const double* close,
  const double* volume,
  const uint8_t* oracle_codes,
  const float* value_means,
  const float* value_second_moments,
  const float* value_entropies,
  const float* value_weights,
  const float* value_opportunities,
  int candle_count,
  int score_start,
  double interval_ms,
  double oracle_friction,
  double match_window_ms,
  double timing_half_life_ms,
  int value_grid_size,
  double value_grid_minimum,
  double value_grid_maximum,
  double strategy_sigma,
  const void* parameter_data,
  int candidate_count,
  void* output_data
) {
  try {
    if (!close_times || !high || !low || !close || !volume || !oracle_codes || !parameter_data || !output_data) {
      throw std::runtime_error("VW-KAMA CUDA received a null input pointer");
    }
    if (candle_count <= 0 || candidate_count <= 0 || score_start < 0 || score_start >= candle_count) {
      throw std::runtime_error("VW-KAMA CUDA received invalid dimensions");
    }
    if (interval_ms <= 0 || match_window_ms < 0 || timing_half_life_ms <= 0) {
      throw std::runtime_error("VW-KAMA CUDA received invalid timing options");
    }
    const bool distillation_enabled = value_grid_size >= 3;
    if (distillation_enabled && (
      !value_means
      || !value_second_moments
      || !value_entropies
      || !value_weights
      || !value_opportunities
      || value_grid_minimum >= 0
      || value_grid_maximum <= 0
      || strategy_sigma <= 0
    )) {
      throw std::runtime_error("VW-KAMA CUDA received invalid value-distillation inputs");
    }
    const auto* parameters = static_cast<const VwKamaParams*>(parameter_data);
    auto* output = static_cast<VwKamaResult*>(output_data);

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) throw std::runtime_error("no CUDA device is available");
    cuda_check(cudaSetDevice(0), "cudaSetDevice");

    std::vector<float> host_high(candle_count);
    std::vector<float> host_low(candle_count);
    std::vector<float> host_close(candle_count);
    std::vector<float> host_volume(candle_count);
    for (int index = 0; index < candle_count; ++index) {
      host_high[index] = static_cast<float>(high[index]);
      host_low[index] = static_cast<float>(low[index]);
      host_close[index] = static_cast<float>(close[index]);
      host_volume[index] = static_cast<float>(volume[index]);
    }

    int max_efficiency_period = 1;
    for (int index = 0; index < candidate_count; ++index) {
      max_efficiency_period = std::max(max_efficiency_period, parameters[index].efficiency_period);
      max_efficiency_period = std::max(
        max_efficiency_period,
        parameters[index].mean_reversion_efficiency_period
      );
    }

    DeviceBuffer<float> device_high(candle_count);
    DeviceBuffer<float> device_low(candle_count);
    DeviceBuffer<float> device_close(candle_count);
    DeviceBuffer<float> device_volume(candle_count);
    DeviceBuffer<uint8_t> device_oracle(candle_count);
    DeviceBuffer<float> device_value_means(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_value_second_moments(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_value_entropies(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_value_weights(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_value_opportunities(distillation_enabled ? candle_count : 0);
    DeviceBuffer<VwKamaParams> device_parameters(candidate_count);
    DeviceBuffer<float> device_changes(
      static_cast<size_t>(candidate_count) * max_efficiency_period * 2
    );
    DeviceBuffer<DeviceResult> device_results(candidate_count);
    cuda_check(cudaMemcpy(device_high.get(), host_high.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy high");
    cuda_check(cudaMemcpy(device_low.get(), host_low.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy low");
    cuda_check(cudaMemcpy(device_close.get(), host_close.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy close");
    cuda_check(cudaMemcpy(device_volume.get(), host_volume.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy volume");
    cuda_check(cudaMemcpy(device_oracle.get(), oracle_codes, candle_count, cudaMemcpyHostToDevice), "copy oracle");
    if (distillation_enabled) {
      cuda_check(cudaMemcpy(device_value_means.get(), value_means, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value means");
      cuda_check(cudaMemcpy(device_value_second_moments.get(), value_second_moments, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value second moments");
      cuda_check(cudaMemcpy(device_value_entropies.get(), value_entropies, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value entropies");
      cuda_check(cudaMemcpy(device_value_weights.get(), value_weights, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value weights");
      cuda_check(cudaMemcpy(device_value_opportunities.get(), value_opportunities, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value opportunities");
    }
    cuda_check(cudaMemcpy(
      device_parameters.get(),
      parameters,
      candidate_count * sizeof(VwKamaParams),
      cudaMemcpyHostToDevice
    ), "copy parameters");

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cuda_check(cudaEventCreate(&start_event), "create CUDA start event");
    cuda_check(cudaEventCreate(&stop_event), "create CUDA stop event");
    cuda_check(cudaEventRecord(start_event), "record CUDA start event");

    const int threads = 128;
    const int blocks = (candidate_count + threads - 1) / threads;
    evaluate_kernel<<<blocks, threads>>>(
      device_high.get(),
      device_low.get(),
      device_close.get(),
      device_volume.get(),
      device_oracle.get(),
      device_value_means.get(),
      device_value_second_moments.get(),
      device_value_entropies.get(),
      device_value_weights.get(),
      device_value_opportunities.get(),
      candle_count,
      score_start,
      static_cast<float>(interval_ms),
      static_cast<float>(oracle_friction),
      value_grid_size,
      static_cast<float>(value_grid_minimum),
      static_cast<float>(value_grid_maximum),
      static_cast<float>(strategy_sigma),
      device_parameters.get(),
      candidate_count,
      device_changes.get(),
      max_efficiency_period,
      device_results.get(),
      nullptr,
      nullptr,
      false
    );
    cuda_check(cudaGetLastError(), "launch VW-KAMA score kernel");

    std::vector<DeviceResult> host_results(candidate_count);
    cuda_check(cudaMemcpy(
      host_results.data(),
      device_results.get(),
      candidate_count * sizeof(DeviceResult),
      cudaMemcpyDeviceToHost
    ), "copy VW-KAMA score results");

    std::vector<int32_t> transition_offsets(candidate_count + 1, 0);
    for (int index = 0; index < candidate_count; ++index) {
      transition_offsets[index + 1] = transition_offsets[index] + host_results[index].signal_count;
    }
    const int total_transitions = transition_offsets.back();
    DeviceBuffer<int32_t> device_transition_offsets(candidate_count);
    DeviceBuffer<uint32_t> device_transitions(total_transitions);
    cuda_check(cudaMemcpy(
      device_transition_offsets.get(),
      transition_offsets.data(),
      candidate_count * sizeof(int32_t),
      cudaMemcpyHostToDevice
    ), "copy VW-KAMA transition offsets");
    if (total_transitions > 0) {
      evaluate_kernel<<<blocks, threads>>>(
        device_high.get(),
        device_low.get(),
        device_close.get(),
        device_volume.get(),
        device_oracle.get(),
        device_value_means.get(),
        device_value_second_moments.get(),
        device_value_entropies.get(),
        device_value_weights.get(),
        device_value_opportunities.get(),
        candle_count,
        score_start,
        static_cast<float>(interval_ms),
        static_cast<float>(oracle_friction),
        value_grid_size,
        static_cast<float>(value_grid_minimum),
        static_cast<float>(value_grid_maximum),
        static_cast<float>(strategy_sigma),
        device_parameters.get(),
        candidate_count,
        device_changes.get(),
        max_efficiency_period,
        device_results.get(),
        device_transition_offsets.get(),
        device_transitions.get(),
        true
      );
      cuda_check(cudaGetLastError(), "launch VW-KAMA transition kernel");
    }
    cuda_check(cudaEventRecord(stop_event), "record CUDA stop event");
    cuda_check(cudaEventSynchronize(stop_event), "synchronize VW-KAMA CUDA kernels");
    float elapsed_ms = 0;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "measure VW-KAMA CUDA kernels");
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    std::vector<uint32_t> host_transitions(total_transitions);
    if (total_transitions > 0) {
      cuda_check(cudaMemcpy(
        host_transitions.data(),
        device_transitions.get(),
        total_transitions * sizeof(uint32_t),
        cudaMemcpyDeviceToHost
      ), "copy VW-KAMA transitions");
    }

    std::vector<Transition> oracle_transitions;
    for (int index = std::max(1, score_start); index < candle_count; ++index) {
      const int previous = state_from_code(oracle_codes[index - 1]);
      const int current = state_from_code(oracle_codes[index]);
      if (current != previous) oracle_transitions.push_back({index, current});
    }

    for (int candidate = 0; candidate < candidate_count; ++candidate) {
      std::vector<Transition> candidate_transitions;
      candidate_transitions.reserve(host_results[candidate].signal_count);
      for (int index = transition_offsets[candidate]; index < transition_offsets[candidate + 1]; ++index) {
        const uint32_t packed = host_transitions[index];
        candidate_transitions.push_back({
          static_cast<int32_t>(packed >> 2u),
          state_from_code(static_cast<uint8_t>(packed & 3u)),
        });
      }
      const AlignmentResult alignment = align_transitions(
        candidate_transitions,
        oracle_transitions,
        close_times,
        match_window_ms,
        timing_half_life_ms
      );
      std::vector<double> absolute_lags(alignment.signed_lags.size());
      std::transform(
        alignment.signed_lags.begin(),
        alignment.signed_lags.end(),
        absolute_lags.begin(),
        [](double value) { return std::abs(value); }
      );
      output[candidate] = {
        static_cast<double>(host_results[candidate].state_credit),
        alignment.credit,
        percentile(absolute_lags, 0.5),
        percentile(absolute_lags, 0.9),
        percentile(absolute_lags, 0.95),
        percentile(alignment.signed_lags, 0.5),
        static_cast<double>(elapsed_ms),
        host_results[candidate].distillation_weighted_cross_entropy,
        host_results[candidate].distillation_weighted_oracle_entropy,
        host_results[candidate].distillation_weight,
        host_results[candidate].distillation_opportunity,
        candle_count - score_start,
        host_results[candidate].signal_count,
        static_cast<int32_t>(oracle_transitions.size()),
        alignment.matched_count,
      };
    }
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}
