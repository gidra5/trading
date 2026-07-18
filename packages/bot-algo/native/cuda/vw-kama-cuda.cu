#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
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
  int32_t strategy_quadratic_volatility_samples;

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
  float strategy_temperature;
  float strategy_quadratic_scale;
  float warmup_multiple;
};

static_assert(sizeof(VwKamaParams) == 208, "VwKamaParams ABI changed");

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
  double strategy_final_equity;
  double oracle_final_equity;
  double strategy_max_drawdown;
  double oracle_max_drawdown;
  double strategy_turnover;
  double oracle_turnover;
  int32_t state_count;
  int32_t signal_count;
  int32_t oracle_count;
  int32_t matched_count;
  double distillation_weighted_strategy_entropy;
  double distillation_weighted_entropy_gap;
  double distillation_state_mutual_information;
  double distillation_oracle_mutual_information;
  double distillation_mixed_loss;
};

static_assert(sizeof(VwKamaResult) == 192, "VwKamaResult ABI changed");

struct DeviceResult {
  float state_credit;
  int32_t signal_count;
  double distillation_weighted_cross_entropy;
  double distillation_weighted_oracle_entropy;
  double distillation_weight;
  double distillation_opportunity;
  double distillation_weighted_strategy_entropy;
  double distillation_weighted_entropy_gap;
  double distillation_state_mutual_information;
  double distillation_oracle_mutual_information;
  double distillation_mixed_loss;
  double strategy_final_equity;
  double oracle_final_equity;
  double strategy_max_drawdown;
  double oracle_max_drawdown;
  double strategy_turnover;
  double oracle_turnover;
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
    if (count <= capacity_) return;
    if (data_) {
      cudaFree(data_);
      data_ = nullptr;
      capacity_ = 0;
    }
    if (count == 0) return;
    const cudaError_t error = cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    capacity_ = count;
  }

  T* get() const { return data_; }
  size_t capacity() const { return capacity_; }

 private:
  T* data_ = nullptr;
  size_t capacity_ = 0;
};

struct CudaFitnessCase {
  static constexpr uint64_t MAGIC = 0x56574b4649544341ULL;
  uint64_t magic = MAGIC;
  int candle_count = 0;
  int score_start = 0;
  float interval_ms = 0.0f;
  int value_holding_period_steps = 1;
  float oracle_friction = 0.0f;
  float quote_lend_rate = 0.0f;
  float quote_borrow_rate = 0.0f;
  float asset_borrow_rate = 0.0f;
  int value_grid_size = 0;
  float value_grid_minimum = 0.0f;
  float value_grid_maximum = 0.0f;
  float maximum_effective_exposure = 250.0f;
  float strategy_quadratic_scale = 0.0f;
  float entropy_gap_lambda = 0.0f;
  float state_mutual_information_lambda = 0.0f;
  float oracle_mutual_information_lambda = 0.0f;
  int oracle_mutual_information_mode = 0;
  int mutual_information_bins = 15;
  bool has_high_low = false;
  double distillation_weight = 0.0;
  double distillation_weighted_oracle_entropy = 0.0;
  size_t resident_bytes = 0;
  DeviceBuffer<float> high;
  DeviceBuffer<float> low;
  DeviceBuffer<float> close;
  DeviceBuffer<float> volume;
  DeviceBuffer<float> value_means;
  DeviceBuffer<float> value_second_moments;
  DeviceBuffer<float> value_entropies;
  DeviceBuffer<float> value_weights;
  DeviceBuffer<float> oracle_bin_probabilities;
  DeviceBuffer<float> strategy_temperatures;
  DeviceBuffer<float> strategy_quadratic_volatilities;
  DeviceBuffer<VwKamaParams> parameters;
  DeviceBuffer<uint64_t> change_offsets;
  DeviceBuffer<float> changes;
  DeviceBuffer<DeviceResult> results;
  DeviceBuffer<float> precise_features;
  DeviceBuffer<double> precise_joint;
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

__device__ __forceinline__ double rebalance_equity_factor(
  float from_exposure,
  float to_exposure,
  float friction
) {
  const float difference = to_exposure - from_exposure;
  if (fabsf(difference) <= 1e-12f) return 1.0;
  if (difference > 0.0f) {
    const double denominator = 1.0 - friction + friction * to_exposure;
    return denominator > 0.0 ? 1.0 - friction * difference / denominator : 0.0;
  }
  const double denominator = 1.0 - friction * to_exposure;
  return denominator > 0.0 ? 1.0 - friction * -difference / denominator : 0.0;
}

__device__ __forceinline__ void update_drawdown(
  double equity,
  double& peak,
  double& max_drawdown
) {
  peak = fmax(peak, equity);
  max_drawdown = fmax(max_drawdown, peak > 0.0 ? 1.0 - equity / peak : 1.0);
}

__device__ __forceinline__ void advance_return(
  float requested_exposure,
  float price,
  float next_price,
  float friction,
  float minimum_exposure,
  float maximum_exposure,
  float maximum_effective_exposure,
  float quote_lend_rate,
  float quote_borrow_rate,
  float asset_borrow_rate,
  double& equity,
  float& current_exposure,
  double& peak,
  double& max_drawdown,
  double& turnover
) {
  if (!(equity > 0.0)) return;
  const float target = clamp_value(requested_exposure, minimum_exposure, maximum_exposure);
  const float change = fabsf(target - current_exposure);
  if (change > 1e-12f) {
    const double rebalance = rebalance_equity_factor(current_exposure, target, friction);
    if (!(rebalance > 0.0)) {
      equity = 0.0;
      max_drawdown = 1.0;
      current_exposure = 0.0f;
      return;
    }
    equity *= rebalance;
    turnover += change;
    update_drawdown(equity, peak, max_drawdown);
  }
  const double quote = 1.0 - target;
  const double asset = target / price;
  const double maintained_quote = quote >= 0.0
    ? quote * (1.0 + quote_lend_rate)
    : quote * (1.0 + quote_borrow_rate);
  const double maintained_asset = asset >= 0.0 ? asset : asset * (1.0 + asset_borrow_rate);
  const double asset_value = maintained_asset * next_price;
  const double marked_equity = maintained_quote + asset_value;
  const double liquidated_asset_value = asset_value >= 0.0
    ? asset_value * (1.0 - friction)
    : asset_value / fmax(1e-12, 1.0 - friction);
  const double liquidation_equity = maintained_quote + liquidated_asset_value;
  const double liquidation_exposure = liquidation_equity > 0.0
    ? liquidated_asset_value / liquidation_equity
    : 0.0;
  if (!(liquidation_equity > 0.0)) {
    equity = 0.0;
    current_exposure = 0.0f;
  } else if (fabs(liquidation_exposure) > maximum_effective_exposure) {
    equity *= liquidation_equity;
    current_exposure = 0.0f;
  } else if (!(marked_equity > 0.0)) {
    equity = 0.0;
    current_exposure = 0.0f;
  } else {
    equity *= marked_equity;
    current_exposure = static_cast<float>(asset_value / marked_equity);
  }
  update_drawdown(equity, peak, max_drawdown);
}

struct OracleHoldingOutcome {
  double equity_factor;
  double exposure;
};

__device__ __forceinline__ double oracle_rebalance_equity_factor(
  double from_exposure,
  double to_exposure,
  double friction
) {
  const double difference = to_exposure - from_exposure;
  if (fabs(difference) <= 2.220446049250313e-16) return 1.0;
  if (difference > 0.0) {
    const double denominator = 1.0 - friction + friction * to_exposure;
    return denominator > 0.0
      ? 1.0 - friction * difference / denominator
      : -INFINITY;
  }
  const double denominator = 1.0 - friction * to_exposure;
  return denominator > 0.0
    ? 1.0 - friction * -difference / denominator
    : -INFINITY;
}

double host_oracle_rebalance_equity_factor(
  double from_exposure,
  double to_exposure,
  double friction
) {
  const double difference = to_exposure - from_exposure;
  if (std::abs(difference) <= std::numeric_limits<double>::epsilon()) return 1.0;
  if (difference > 0.0) {
    const double denominator = 1.0 - friction + friction * to_exposure;
    return denominator > 0.0
      ? 1.0 - friction * difference / denominator
      : -std::numeric_limits<double>::infinity();
  }
  const double denominator = 1.0 - friction * to_exposure;
  return denominator > 0.0
    ? 1.0 - friction * -difference / denominator
    : -std::numeric_limits<double>::infinity();
}

__device__ __forceinline__ OracleHoldingOutcome oracle_holding_outcome(
  double exposure,
  double price,
  double next_price,
  double friction,
  double minimum_exposure,
  double maximum_exposure,
  double maximum_effective_exposure,
  double quote_lend_rate,
  double quote_borrow_rate,
  double asset_borrow_rate
) {
  const double quote = 1.0 - exposure;
  const double asset = exposure / price;
  const double maintained_quote = quote >= 0.0
    ? quote * (1.0 + quote_lend_rate)
    : quote * (1.0 + quote_borrow_rate);
  const double maintained_asset = asset >= 0.0
    ? asset
    : asset * (1.0 + asset_borrow_rate);
  const double asset_value = maintained_asset * next_price;
  const double marked_equity = maintained_quote + asset_value;
  const double liquidated_asset_value = asset_value >= 0.0
    ? asset_value * (1.0 - friction)
    : asset_value / fmax(2.220446049250313e-16, 1.0 - friction);
  const double liquidation_equity = maintained_quote + liquidated_asset_value;
  if (!(liquidation_equity > 0.0) || !isfinite(liquidation_equity)) return {0.0, 0.0};
  const double liquidation_exposure = liquidated_asset_value / liquidation_equity;
  if (fabs(liquidation_exposure) > maximum_effective_exposure) {
    return {liquidation_equity, 0.0};
  }
  if (!(marked_equity > 0.0) || !isfinite(marked_equity)) return {0.0, 0.0};
  return {marked_equity, asset_value / marked_equity};
}

__device__ __forceinline__ double oracle_interpolate(
  const double* values,
  int grid_size,
  double minimum_exposure,
  double maximum_exposure,
  double exposure
) {
  const double raw_position = (exposure - minimum_exposure)
    / (maximum_exposure - minimum_exposure) * (grid_size - 1);
  const double position = fmax(0.0, fmin(static_cast<double>(grid_size - 1), raw_position));
  const int lower = static_cast<int>(floor(position));
  const int upper = min(grid_size - 1, lower + 1);
  const double fraction = position - lower;
  const double left = values[lower];
  const double right = values[upper];
  if (!isfinite(left)) return right;
  if (!isfinite(right)) return left;
  return left * (1.0 - fraction) + right * fraction;
}

__device__ __forceinline__ double oracle_grid_exposure(
  int index,
  int grid_size,
  double minimum_exposure,
  double maximum_exposure
) {
  return minimum_exposure
    + static_cast<double>(index) / (grid_size - 1) * (maximum_exposure - minimum_exposure);
}

__device__ __forceinline__ void synchronize_oracle_grid() {
  if (blockDim.x <= warpSize) {
    __syncwarp(__activemask());
  } else {
    __syncthreads();
  }
}

__device__ __forceinline__ float truncated_exponential_log_normalizer(
  int grid_size,
  float minimum_exposure,
  float maximum_exposure,
  float slope
) {
  if (slope == 0.0f) return logf(static_cast<float>(grid_size));
  const float spacing = (maximum_exposure - minimum_exposure) / (grid_size - 1);
  const float step = fabsf(slope) * spacing;
  const float maximum_log_weight = fmaxf(
    slope * minimum_exposure,
    slope * maximum_exposure
  );
  if (step < 1.0e-4f) {
    const float count = static_cast<float>(grid_size);
    const float mean_index = 0.5f * (count - 1.0f);
    const float variance_index = (count * count - 1.0f) / 12.0f;
    return maximum_log_weight + logf(count)
      - mean_index * step + 0.5f * variance_index * step * step;
  }
  return maximum_log_weight
    + logf(-expm1f(-step * grid_size))
    - logf(-expm1f(-step));
}

__device__ __forceinline__ float quadratic_exponential_log_normalizer(
  int grid_size,
  float minimum_exposure,
  float maximum_exposure,
  float linear_coefficient,
  float quadratic_coefficient
) {
  if (quadratic_coefficient == 0.0f) {
    return truncated_exponential_log_normalizer(
      grid_size,
      minimum_exposure,
      maximum_exposure,
      linear_coefficient
    );
  }
  const float spacing = (maximum_exposure - minimum_exposure) / (grid_size - 1);
  const float minimum_log_weight = linear_coefficient * minimum_exposure
    + quadratic_coefficient * minimum_exposure * minimum_exposure;
  const float maximum_log_weight = linear_coefficient * maximum_exposure
    + quadratic_coefficient * maximum_exposure * maximum_exposure;
  float maximum = fmaxf(minimum_log_weight, maximum_log_weight);
  int peak_index = minimum_log_weight >= maximum_log_weight ? 0 : grid_size - 1;
  if (quadratic_coefficient < 0.0f) {
    const float vertex_position = clamp_value(
      (-linear_coefficient / (2.0f * quadratic_coefficient) - minimum_exposure) / spacing,
      0.0f,
      static_cast<float>(grid_size - 1)
    );
    const int lower = static_cast<int>(floorf(vertex_position));
    const int upper = min(grid_size - 1, lower + 1);
    const float lower_exposure = minimum_exposure + lower * spacing;
    const float upper_exposure = minimum_exposure + upper * spacing;
    const float lower_log_weight = linear_coefficient * lower_exposure
      + quadratic_coefficient * lower_exposure * lower_exposure;
    const float upper_log_weight = linear_coefficient * upper_exposure
      + quadratic_coefficient * upper_exposure * upper_exposure;
    if (lower_log_weight > maximum) {
      maximum = lower_log_weight;
      peak_index = lower;
    }
    if (upper_log_weight > maximum) {
      maximum = upper_log_weight;
      peak_index = upper;
    }

    // Curvature is candidate-independent at each candle, so this fixed-width window keeps
    // every thread in a warp on the same loop count. The skipped tail is below one Float32
    // epsilon of the modal term for grids up to the supported 1,024 points.
    const float index_curvature = -quadratic_coefficient * spacing * spacing;
    const int radius = min(
      grid_size,
      static_cast<int>(ceilf(0.5f + sqrtf(0.25f + 25.0f / index_curvature)))
    );
    const int active_count = min(grid_size, 2 * radius + 1);
    const int start_index = max(0, min(peak_index - radius, grid_size - active_count));
    const float start_exposure = minimum_exposure + start_index * spacing;
    const float second_difference = 2.0f * quadratic_coefficient * spacing * spacing;
    float total = 0.0f;
    float relative_log_weight = linear_coefficient * start_exposure
      + quadratic_coefficient * start_exposure * start_exposure - maximum;
    float delta = linear_coefficient * spacing
      + quadratic_coefficient * (2.0f * start_exposure * spacing + spacing * spacing);
    for (int offset = 0; offset < active_count; ++offset) {
      total += expf(relative_log_weight);
      relative_log_weight += delta;
      delta += second_difference;
    }
    return maximum + logf(total);
  }
  float total = 0.0f;
  for (int index = 0; index < grid_size; ++index) {
    const float exposure = minimum_exposure + index * spacing;
    total += expf(
      linear_coefficient * exposure + quadratic_coefficient * exposure * exposure - maximum
    );
  }
  return maximum + logf(total);
}

struct QuadraticExponentialStatistics {
  float log_normalizer;
  float mean;
  float second_moment;
  float entropy;
};

__device__ __forceinline__ QuadraticExponentialStatistics quadratic_exponential_statistics(
  int grid_size,
  float minimum_exposure,
  float maximum_exposure,
  float linear_coefficient,
  float quadratic_coefficient
) {
  const float spacing = (maximum_exposure - minimum_exposure) / (grid_size - 1);
  if (quadratic_coefficient == 0.0f) {
    const float log_normalizer = truncated_exponential_log_normalizer(
      grid_size,
      minimum_exposure,
      maximum_exposure,
      linear_coefficient
    );
    if (linear_coefficient == 0.0f) {
      const double mean = 0.5 * (minimum_exposure + maximum_exposure);
      const double index_variance = (
        static_cast<double>(grid_size) * grid_size - 1.0
      ) / 12.0;
      const double variance = spacing * spacing * index_variance;
      return {
        log_normalizer,
        static_cast<float>(mean),
        static_cast<float>(mean * mean + variance),
        log_normalizer,
      };
    }
    const double step = fabs(static_cast<double>(linear_coefficient) * spacing);
    if (step >= 1.0e-4) {
      const double ratio = exp(-step);
      const double tail = exp(-step * grid_size);
      const double one_minus_ratio = 1.0 - ratio;
      const double one_minus_tail = 1.0 - tail;
      const double mean_distance = ratio / one_minus_ratio
        - grid_size * tail / one_minus_tail;
      const double variance_distance = ratio / (one_minus_ratio * one_minus_ratio)
        - static_cast<double>(grid_size) * grid_size * tail
          / (one_minus_tail * one_minus_tail);
      const double mean = linear_coefficient > 0.0f
        ? maximum_exposure - spacing * mean_distance
        : minimum_exposure + spacing * mean_distance;
      const double variance = fmax(0.0, spacing * spacing * variance_distance);
      const double second_moment = mean * mean + variance;
      return {
        log_normalizer,
        static_cast<float>(mean),
        static_cast<float>(second_moment),
        fmaxf(0.0f, log_normalizer - linear_coefficient * static_cast<float>(mean)),
      };
    }
  }

  const float minimum_log_weight = linear_coefficient * minimum_exposure
    + quadratic_coefficient * minimum_exposure * minimum_exposure;
  const float maximum_log_weight = linear_coefficient * maximum_exposure
    + quadratic_coefficient * maximum_exposure * maximum_exposure;
  float maximum = fmaxf(minimum_log_weight, maximum_log_weight);
  int peak_index = minimum_log_weight >= maximum_log_weight ? 0 : grid_size - 1;
  int start_index = 0;
  int active_count = grid_size;
  if (quadratic_coefficient < 0.0f) {
    const float vertex_position = clamp_value(
      (-linear_coefficient / (2.0f * quadratic_coefficient) - minimum_exposure) / spacing,
      0.0f,
      static_cast<float>(grid_size - 1)
    );
    const int lower = static_cast<int>(floorf(vertex_position));
    const int upper = min(grid_size - 1, lower + 1);
    const float lower_exposure = minimum_exposure + lower * spacing;
    const float upper_exposure = minimum_exposure + upper * spacing;
    const float lower_log_weight = linear_coefficient * lower_exposure
      + quadratic_coefficient * lower_exposure * lower_exposure;
    const float upper_log_weight = linear_coefficient * upper_exposure
      + quadratic_coefficient * upper_exposure * upper_exposure;
    if (lower_log_weight > maximum) {
      maximum = lower_log_weight;
      peak_index = lower;
    }
    if (upper_log_weight > maximum) {
      maximum = upper_log_weight;
      peak_index = upper;
    }
    const float index_curvature = -quadratic_coefficient * spacing * spacing;
    const int radius = min(
      grid_size,
      static_cast<int>(ceilf(0.5f + sqrtf(0.25f + 25.0f / index_curvature)))
    );
    active_count = min(grid_size, 2 * radius + 1);
    start_index = max(0, min(peak_index - radius, grid_size - active_count));
  }
  const float start_exposure = minimum_exposure + start_index * spacing;
  const float second_difference = 2.0f * quadratic_coefficient * spacing * spacing;
  double total = 0.0;
  double weighted_mean = 0.0;
  double weighted_second_moment = 0.0;
  float relative_log_weight = linear_coefficient * start_exposure
    + quadratic_coefficient * start_exposure * start_exposure - maximum;
  float delta = linear_coefficient * spacing
    + quadratic_coefficient * (2.0f * start_exposure * spacing + spacing * spacing);
  for (int offset = 0; offset < active_count; ++offset) {
    const double weight = exp(static_cast<double>(relative_log_weight));
    const double exposure = start_exposure + offset * spacing;
    total += weight;
    weighted_mean += weight * exposure;
    weighted_second_moment += weight * exposure * exposure;
    relative_log_weight += delta;
    delta += second_difference;
  }
  const float log_normalizer = maximum + logf(static_cast<float>(total));
  const double mean = weighted_mean / total;
  const double second_moment = weighted_second_moment / total;
  return {
    log_normalizer,
    static_cast<float>(mean),
    static_cast<float>(second_moment),
    fmaxf(0.0f, log_normalizer
      - linear_coefficient * static_cast<float>(mean)
      - quadratic_coefficient * static_cast<float>(second_moment)),
  };
}

__device__ __forceinline__ double normalized_gaussian_variance_information(
  double total_variance,
  double conditional_variance,
  int grid_size
) {
  if (total_variance <= 2.220446049250313e-16) return 0.0;
  if (conditional_variance <= 2.220446049250313e-16) return 1.0;
  return fmax(0.0, fmin(
    1.0,
    0.5 * log(total_variance / conditional_variance) / log(static_cast<double>(grid_size))
  ));
}

__device__ __forceinline__ double normalized_gaussian_correlation_information(
  double oracle_mean,
  double oracle_second_moment,
  double strategy_mean,
  double strategy_second_moment,
  double cross_mean,
  int grid_size
) {
  const double oracle_variance = fmax(0.0, oracle_second_moment - oracle_mean * oracle_mean);
  const double strategy_variance = fmax(0.0, strategy_second_moment - strategy_mean * strategy_mean);
  if (oracle_variance <= 2.220446049250313e-16
    || strategy_variance <= 2.220446049250313e-16) return 0.0;
  const double covariance = cross_mean - oracle_mean * strategy_mean;
  const double correlation_squared = fmax(0.0, fmin(
    1.0 - 1.0e-12,
    covariance * covariance / (oracle_variance * strategy_variance)
  ));
  return fmax(0.0, fmin(
    1.0,
    -0.5 * log1p(-correlation_squared) / log(static_cast<double>(grid_size))
  ));
}

__global__ void prepare_value_oracle_holds_kernel(
  const double* prices,
  int price_count,
  int score_start,
  int holding_period_steps,
  int grid_size,
  double minimum_exposure,
  double maximum_exposure,
  double maximum_effective_exposure,
  double friction,
  double quote_lend_rate,
  double quote_borrow_rate,
  double asset_borrow_rate,
  double* holding_values,
  double* endpoint_exposures
) {
  const size_t cell = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int scored_count = price_count - score_start;
  const size_t cell_count = static_cast<size_t>(scored_count) * grid_size;
  if (cell >= cell_count) return;
  const int exposure_index = cell % grid_size;
  const int time = score_start + cell / grid_size;
  const double exposure = oracle_grid_exposure(
    exposure_index,
    grid_size,
    minimum_exposure,
    maximum_exposure
  );
  if (time == price_count - 1) {
    holding_values[cell] = 0.0;
    endpoint_exposures[cell] = exposure;
    return;
  }
  const int endpoint_time = min(price_count - 1, time + holding_period_steps);
  double log_return = 0.0;
  double current_exposure = exposure;
  for (int move = time; move < endpoint_time; ++move) {
    const OracleHoldingOutcome outcome = oracle_holding_outcome(
      current_exposure,
      prices[move],
      prices[move + 1],
      friction,
      minimum_exposure,
      maximum_exposure,
      maximum_effective_exposure,
      quote_lend_rate,
      quote_borrow_rate,
      asset_borrow_rate
    );
    if (!(outcome.equity_factor > 0.0)) {
      log_return = -INFINITY;
      current_exposure = 0.0;
      break;
    }
    log_return += log(outcome.equity_factor);
    current_exposure = outcome.exposure;
  }
  holding_values[cell] = log_return;
  endpoint_exposures[cell] = current_exposure;
}

__global__ void prepare_value_oracle_chains_kernel(
  int price_count,
  int score_start,
  int holding_period_steps,
  int grid_size,
  double minimum_exposure,
  double maximum_exposure,
  double temperature,
  double friction,
  double opportunity_epsilon,
  const double* holding_values,
  const double* endpoint_exposures,
  const double* rebalance_logs,
  bool separable_rebalance_costs,
  const double* sell_logs,
  const double* buy_logs,
  const double* prior_continuations,
  double* continuations,
  uint16_t* policy,
  float* means,
  float* second_moments,
  float* modal_exposures,
  float* entropies,
  float* weights,
  float* opportunities,
  float* probabilities,
  bool finite_horizon,
  bool has_prior_continuation,
  bool collect_statistics,
  bool terminal_closeout
) {
  const int exposure_index = threadIdx.x;
  const int first_time = finite_horizon
    ? score_start + blockIdx.x
    : price_count - 1 - blockIdx.x;
  const int last_time = finite_horizon ? first_time : score_start;
  extern __shared__ unsigned char chain_storage[];
  double* forced_values = reinterpret_cast<double*>(chain_storage);
  double* scan_values_a = forced_values + blockDim.x;
  double* scan_values_b = scan_values_a + blockDim.x;
  uint16_t* scan_targets_a = reinterpret_cast<uint16_t*>(scan_values_b + blockDim.x);
  uint16_t* scan_targets_b = scan_targets_a + blockDim.x;

  for (int time = first_time; time >= last_time; time -= holding_period_steps) {
    const size_t row = static_cast<size_t>(time - score_start) * grid_size;
    if (exposure_index < grid_size) {
      if (time == price_count - 1) {
        const double exposure = oracle_grid_exposure(
          exposure_index,
          grid_size,
          minimum_exposure,
          maximum_exposure
        );
        const double closeout = oracle_rebalance_equity_factor(exposure, 0.0, friction);
        forced_values[exposure_index] = terminal_closeout && closeout > 0.0
          ? log(closeout)
          : 0.0;
      } else {
        const int endpoint_time = min(price_count - 1, time + holding_period_steps);
        const size_t index = row + exposure_index;
        if (!isfinite(holding_values[index])) {
          forced_values[exposure_index] = -INFINITY;
        } else if (finite_horizon && !has_prior_continuation) {
          forced_values[exposure_index] = holding_values[index];
        } else {
          const double* endpoint_continuation = (finite_horizon
              ? prior_continuations
              : continuations)
            + static_cast<size_t>(endpoint_time - score_start) * grid_size;
          forced_values[exposure_index] = holding_values[index] + oracle_interpolate(
            endpoint_continuation,
            grid_size,
            minimum_exposure,
            maximum_exposure,
            endpoint_exposures[index]
          );
        }
      }
    }
    synchronize_oracle_grid();

    if (collect_statistics && exposure_index == 0) {
      double maximum = -INFINITY;
      double minimum = INFINITY;
      int maximum_index = 0;
      bool invalid = false;
      for (int index = 0; index < grid_size; ++index) {
        const double value = forced_values[index];
        if (!isfinite(value)) {
          invalid = true;
          continue;
        }
        if (value > maximum) {
          maximum = value;
          maximum_index = index;
        }
        minimum = fmin(minimum, value);
      }
      if (!isfinite(maximum)) {
        means[time] = 0.0f;
        second_moments[time] = 0.0f;
        modal_exposures[time] = 0.0f;
        entropies[time] = 0.0f;
        opportunities[time] = 0.0f;
        weights[time] = static_cast<float>(opportunity_epsilon);
      } else {
        double total = 0.0;
        double weighted = 0.0;
        double weighted_squared = 0.0;
        double weighted_log_weight = 0.0;
        for (int index = 0; index < grid_size; ++index) {
          const float scaled = isfinite(forced_values[index])
            ? static_cast<float>((forced_values[index] - maximum) / temperature)
            : -50.0f;
          const float probability_weight = expf(fmaxf(-50.0f, scaled));
          if (probabilities) probabilities[static_cast<size_t>(time) * grid_size + index]
            = probability_weight;
          const double exposure = oracle_grid_exposure(
            index,
            grid_size,
            minimum_exposure,
            maximum_exposure
          );
          total += probability_weight;
          weighted += probability_weight * exposure;
          weighted_squared += probability_weight * exposure * exposure;
          weighted_log_weight += probability_weight * logf(probability_weight);
        }
        if (probabilities) {
          for (int index = 0; index < grid_size; ++index) {
            probabilities[static_cast<size_t>(time) * grid_size + index] /= total;
          }
        }
        const double opportunity = fmax(
          maximum - minimum,
          invalid ? temperature * 50.0 : 0.0
        );
        means[time] = static_cast<float>(weighted / total);
        second_moments[time] = static_cast<float>(weighted_squared / total);
        modal_exposures[time] = static_cast<float>(oracle_grid_exposure(
          maximum_index,
          grid_size,
          minimum_exposure,
          maximum_exposure
        ));
        entropies[time] = static_cast<float>(log(total) - weighted_log_weight / total);
        opportunities[time] = static_cast<float>(opportunity);
        weights[time] = static_cast<float>(opportunity + opportunity_epsilon);
      }
    }
    synchronize_oracle_grid();

    if (time == price_count - 1) {
      if (exposure_index < grid_size) {
        continuations[row + exposure_index] = forced_values[exposure_index];
        const double zero_position = -minimum_exposure
          / (maximum_exposure - minimum_exposure) * (grid_size - 1);
        policy[row + exposure_index] = terminal_closeout
          ? static_cast<uint16_t>(llround(zero_position))
          : exposure_index;
      }
      synchronize_oracle_grid();
      continue;
    }

    if (separable_rebalance_costs) {
      scan_values_a[exposure_index] = exposure_index < grid_size
        && isfinite(forced_values[exposure_index])
        ? forced_values[exposure_index] - sell_logs[exposure_index]
        : -INFINITY;
      scan_targets_a[exposure_index] = exposure_index < grid_size
        ? exposure_index
        : UINT16_MAX;
      synchronize_oracle_grid();
      bool source_is_a = true;
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        const double* source_values = source_is_a ? scan_values_a : scan_values_b;
        const uint16_t* source_targets = source_is_a ? scan_targets_a : scan_targets_b;
        double* destination_values = source_is_a ? scan_values_b : scan_values_a;
        uint16_t* destination_targets = source_is_a ? scan_targets_b : scan_targets_a;
        double best = source_values[exposure_index];
        uint16_t best_target = source_targets[exposure_index];
        if (exposure_index >= offset) {
          const double other = source_values[exposure_index - offset];
          const uint16_t other_target = source_targets[exposure_index - offset];
          if (other > best || (other == best && other_target < best_target)) {
            best = other;
            best_target = other_target;
          }
        }
        destination_values[exposure_index] = best;
        destination_targets[exposure_index] = best_target;
        synchronize_oracle_grid();
        source_is_a = !source_is_a;
      }
      const double* prefix_scan_values = source_is_a ? scan_values_a : scan_values_b;
      const uint16_t* prefix_scan_targets = source_is_a ? scan_targets_a : scan_targets_b;
      if (exposure_index < grid_size) {
        continuations[row + exposure_index] = prefix_scan_values[exposure_index];
        policy[row + exposure_index] = prefix_scan_targets[exposure_index];
      }
      synchronize_oracle_grid();

      const int reverse_target = grid_size - 1 - exposure_index;
      scan_values_a[exposure_index] = reverse_target >= 0
        && isfinite(forced_values[reverse_target])
        ? forced_values[reverse_target] - buy_logs[reverse_target]
        : -INFINITY;
      scan_targets_a[exposure_index] = reverse_target >= 0
        ? reverse_target
        : UINT16_MAX;
      synchronize_oracle_grid();
      source_is_a = true;
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        const double* source_values = source_is_a ? scan_values_a : scan_values_b;
        const uint16_t* source_targets = source_is_a ? scan_targets_a : scan_targets_b;
        double* destination_values = source_is_a ? scan_values_b : scan_values_a;
        uint16_t* destination_targets = source_is_a ? scan_targets_b : scan_targets_a;
        double best = source_values[exposure_index];
        uint16_t best_target = source_targets[exposure_index];
        if (exposure_index >= offset) {
          const double other = source_values[exposure_index - offset];
          const uint16_t other_target = source_targets[exposure_index - offset];
          if (other > best || (other == best && other_target < best_target)) {
            best = other;
            best_target = other_target;
          }
        }
        destination_values[exposure_index] = best;
        destination_targets[exposure_index] = best_target;
        synchronize_oracle_grid();
        source_is_a = !source_is_a;
      }
      const double* suffix_scan_values = source_is_a ? scan_values_a : scan_values_b;
      const uint16_t* suffix_scan_targets = source_is_a ? scan_targets_a : scan_targets_b;
      if (exposure_index < grid_size) {
        const int reverse_index = grid_size - 1 - exposure_index;
        const double sell = sell_logs[exposure_index] + continuations[row + exposure_index];
        const double buy = buy_logs[exposure_index] + suffix_scan_values[reverse_index];
        const int sell_target = policy[row + exposure_index];
        const int buy_target = suffix_scan_targets[reverse_index];
        const bool choose_buy = buy > sell || (buy == sell && buy_target < sell_target);
        const double best = choose_buy ? buy : sell;
        continuations[row + exposure_index] = best;
        policy[row + exposure_index] = isfinite(best)
          ? (choose_buy ? buy_target : sell_target)
          : exposure_index;
      }
    } else if (exposure_index < grid_size) {
      double best = -INFINITY;
      int best_target = exposure_index;
      for (int target = 0; target < grid_size; ++target) {
        const double rebalance_log = rebalance_logs[exposure_index * grid_size + target];
        const double forced = forced_values[target];
        if (!isfinite(rebalance_log) || !isfinite(forced)) continue;
        const double value = rebalance_log + forced;
        if (value > best) {
          best = value;
          best_target = target;
        }
      }
      continuations[row + exposure_index] = best;
      policy[row + exposure_index] = best_target;
    }
    synchronize_oracle_grid();
  }
}

__global__ void initialize_value_oracle_terminal_kernel(
  int terminal_time,
  int score_start,
  int grid_size,
  double minimum_exposure,
  double maximum_exposure,
  double friction,
  double* continuations,
  uint16_t* policy
) {
  const int exposure_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (exposure_index >= grid_size) return;
  const double exposure = oracle_grid_exposure(
    exposure_index,
    grid_size,
    minimum_exposure,
    maximum_exposure
  );
  const double closeout = oracle_rebalance_equity_factor(exposure, 0.0, friction);
  const size_t row = static_cast<size_t>(terminal_time - score_start) * grid_size;
  continuations[row + exposure_index] = closeout > 0.0 ? log(closeout) : -INFINITY;
  const double zero_position = -minimum_exposure
    / (maximum_exposure - minimum_exposure) * (grid_size - 1);
  policy[row + exposure_index] = static_cast<uint16_t>(llround(zero_position));
}

__global__ void reconstruct_value_oracle_policy_kernel(
  const double* prices,
  int price_count,
  int score_start,
  int holding_period_steps,
  int grid_size,
  double minimum_exposure,
  double maximum_exposure,
  double maximum_effective_exposure,
  double friction,
  double quote_lend_rate,
  double quote_borrow_rate,
  double asset_borrow_rate,
  double initial_exposure,
  const double* holding_values,
  const double* endpoint_exposures,
  const double* continuations,
  float* path_exposures,
  double* path_equities,
  double* path_metrics
) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double current_exposure = initial_exposure;
  double equity = 1.0;
  double peak_equity = 1.0;
  double maximum_drawdown = 0.0;
  double turnover = 0.0;
  double rebalance_count = 0.0;
  double liquidation_count = 0.0;
  int remaining_hold_steps = 0;
  path_equities[score_start] = equity;
  for (int time = score_start; time + 1 < price_count; ++time) {
    if (remaining_hold_steps == 0) {
      const size_t row = static_cast<size_t>(time - score_start) * grid_size;
      const int endpoint_time = min(price_count - 1, time + holding_period_steps);
      const double* endpoint_continuation = continuations
        + static_cast<size_t>(endpoint_time - score_start) * grid_size;
      double no_trade_log_return = 0.0;
      double no_trade_exposure = current_exposure;
      for (int move = time; move < endpoint_time; ++move) {
        const OracleHoldingOutcome outcome = oracle_holding_outcome(
          no_trade_exposure,
          prices[move],
          prices[move + 1],
          friction,
          minimum_exposure,
          maximum_exposure,
          maximum_effective_exposure,
          quote_lend_rate,
          quote_borrow_rate,
          asset_borrow_rate
        );
        if (!(outcome.equity_factor > 0.0)) {
          no_trade_log_return = -INFINITY;
          no_trade_exposure = 0.0;
          break;
        }
        no_trade_log_return += log(outcome.equity_factor);
        no_trade_exposure = outcome.exposure;
      }
      double best = isfinite(no_trade_log_return)
        ? no_trade_log_return + oracle_interpolate(
            endpoint_continuation,
            grid_size,
            minimum_exposure,
            maximum_exposure,
            no_trade_exposure
          )
        : -INFINITY;
      double target = current_exposure;
      for (int target_index = 0; target_index < grid_size; ++target_index) {
        const size_t index = row + target_index;
        if (!isfinite(holding_values[index])) continue;
        const double candidate_target = oracle_grid_exposure(
          target_index,
          grid_size,
          minimum_exposure,
          maximum_exposure
        );
        const double rebalance = oracle_rebalance_equity_factor(
          current_exposure,
          candidate_target,
          friction
        );
        if (!(rebalance > 0.0)) continue;
        const double forced = holding_values[index] + oracle_interpolate(
          endpoint_continuation,
          grid_size,
          minimum_exposure,
          maximum_exposure,
          endpoint_exposures[index]
        );
        const double value = log(rebalance) + forced;
        if (value > best) {
          best = value;
          target = candidate_target;
        }
      }
      const double exposure_change = fabs(target - current_exposure);
      if (exposure_change > 2.220446049250313e-16) {
        const double rebalance = oracle_rebalance_equity_factor(
          current_exposure,
          target,
          friction
        );
        if (!(rebalance > 0.0)) {
          equity = 0.0;
          current_exposure = 0.0;
          maximum_drawdown = 1.0;
          path_equities[time + 1] = equity;
          remaining_hold_steps = min(holding_period_steps, price_count - 1 - time) - 1;
          continue;
        }
        equity *= rebalance;
        current_exposure = target;
        turnover += exposure_change;
        rebalance_count += 1.0;
        peak_equity = fmax(peak_equity, equity);
        maximum_drawdown = fmax(maximum_drawdown, 1.0 - equity / peak_equity);
      }
      remaining_hold_steps = min(holding_period_steps, price_count - 1 - time);
    }
    path_exposures[time] = static_cast<float>(current_exposure);
    const double held_exposure = current_exposure;
    const OracleHoldingOutcome outcome = oracle_holding_outcome(
      held_exposure,
      prices[time],
      prices[time + 1],
      friction,
      minimum_exposure,
      maximum_exposure,
      maximum_effective_exposure,
      quote_lend_rate,
      quote_borrow_rate,
      asset_borrow_rate
    );
    equity *= outcome.equity_factor;
    current_exposure = outcome.exposure;
    if (held_exposure != 0.0 && outcome.exposure == 0.0) liquidation_count += 1.0;
    peak_equity = fmax(peak_equity, equity);
    maximum_drawdown = fmax(
      maximum_drawdown,
      peak_equity > 0.0 ? 1.0 - equity / peak_equity : 1.0
    );
    path_equities[time + 1] = equity;
    --remaining_hold_steps;
  }
  const int terminal = price_count - 1;
  if (equity > 0.0 && fabs(current_exposure) > 2.220446049250313e-16) {
    const double closeout = oracle_rebalance_equity_factor(current_exposure, 0.0, friction);
    turnover += fabs(current_exposure);
    rebalance_count += 1.0;
    equity = closeout > 0.0 ? equity * closeout : 0.0;
    peak_equity = fmax(peak_equity, equity);
    maximum_drawdown = fmax(
      maximum_drawdown,
      peak_equity > 0.0 ? 1.0 - equity / peak_equity : 1.0
    );
  }
  path_exposures[terminal] = 0.0f;
  path_equities[terminal] = equity;
  path_metrics[0] = equity > 0.0 ? log(equity) : -INFINITY;
  path_metrics[1] = maximum_drawdown;
  path_metrics[2] = turnover;
  path_metrics[3] = rebalance_count;
  path_metrics[4] = liquidation_count;
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
  const float* value_optimal_exposures,
  const float* value_weights,
  const float* strategy_temperatures,
  const float* strategy_quadratic_volatilities,
  int candle_count,
  int score_start,
  float interval_ms,
  int value_holding_period_steps,
  float oracle_friction,
  float quote_lend_rate,
  float quote_borrow_rate,
  float asset_borrow_rate,
  int value_grid_size,
  float value_grid_minimum,
  float value_grid_maximum,
  float maximum_effective_exposure,
  float strategy_quadratic_scale,
  float entropy_gap_lambda,
  float state_mutual_information_lambda,
  float oracle_mutual_information_lambda,
  int oracle_mutual_information_mode,
  const VwKamaParams* parameters,
  int candidate_count,
  float* changes,
  const uint64_t* change_offsets,
  DeviceResult* results,
  const int32_t* transition_offsets,
  uint32_t* transitions,
  int transition_capacity,
  float* precise_features,
  bool fitness_only,
  bool write_transitions
) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  const VwKamaParams p = parameters[candidate];
  float* change_ring = changes + change_offsets[candidate];
  float* mean_change_ring = change_ring + p.efficiency_period;

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
  double distillation_weight = 0.0;
  double distillation_weighted_strategy_entropy = 0.0;
  double distillation_weighted_entropy_gap = 0.0;
  double weighted_strategy_mean = 0.0;
  double weighted_strategy_second_moment = 0.0;
  double weighted_strategy_variance = 0.0;
  double weighted_oracle_mean = 0.0;
  double weighted_oracle_second_moment = 0.0;
  double weighted_oracle_strategy_mean_product = 0.0;
  double strategy_equity = 1.0;
  double oracle_equity = 1.0;
  float strategy_exposure = 0.0f;
  float oracle_exposure = 0.0f;
  double strategy_peak = 1.0;
  double oracle_peak = 1.0;
  double strategy_max_drawdown = 0.0;
  double oracle_max_drawdown = 0.0;
  double strategy_turnover = 0.0;
  double oracle_turnover = 0.0;
  const bool distillation_enabled = value_grid_size >= 3
    && strategy_temperatures
    && strategy_quadratic_volatilities
    && value_means
    && value_weights;
  const bool loss_enabled = distillation_enabled && (
    entropy_gap_lambda > 0.0f
    || state_mutual_information_lambda > 0.0f
    || oracle_mutual_information_lambda > 0.0f
  );
  const int quadratic_capacity = max(1, p.strategy_quadratic_volatility_samples);
  int quadratic_count = 0;
  double quadratic_sum = 0.0;
  double quadratic_sum_squares = 0.0;
  if (distillation_enabled) {
    const int first_return = max(1, feed_start - quadratic_capacity + 1);
    for (int return_index = first_return; return_index <= feed_start; ++return_index) {
      const double log_return = log(
        static_cast<double>(close[return_index]) / close[return_index - 1]
      );
      quadratic_sum += log_return;
      quadratic_sum_squares += log_return * log_return;
      ++quadratic_count;
    }
  }

  for (int index = feed_start; index < candle_count; ++index) {
    const float price = close[index];
    const float candle_volume = volume[index];
    if (distillation_enabled && index > feed_start) {
      if (quadratic_count == quadratic_capacity) {
        const int removed_index = index - quadratic_capacity;
        const double removed = log(
          static_cast<double>(close[removed_index]) / close[removed_index - 1]
        );
        quadratic_sum -= removed;
        quadratic_sum_squares -= removed * removed;
      } else {
        ++quadratic_count;
      }
      const double added = log(static_cast<double>(price) / close[index - 1]);
      quadratic_sum += added;
      quadratic_sum_squares += added * added;
    }

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
        if (transitions && (write_transitions || signal_count < transition_capacity)) {
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
      if (!fitness_only) {
        state_credit += agreement_credit(
          p.agreement_mode,
          current,
          exposure,
          state_from_code(oracle_codes[index])
        );
      }
      if (distillation_enabled) {
        const float expected_log_return = rate / 10000.0f
          * interval_ms * value_holding_period_steps / 3600000.0f;
        const float effective_temperature = p.strategy_temperature * strategy_temperatures[index];
        const float slope = expected_log_return / fmaxf(1e-20f, effective_temperature);
        const double quadratic_mean = quadratic_count > 0
          ? quadratic_sum / quadratic_count
          : 0.0;
        const float volatility = quadratic_count > 0
          ? static_cast<float>(sqrt(fmax(
              0.0,
              quadratic_sum_squares / quadratic_count - quadratic_mean * quadratic_mean
            )))
          : 0.0f;
        const float strategy_quadratic_coefficient =
          p.strategy_quadratic_scale == 0.0f || volatility == 0.0f
            ? 0.0f
            : -p.strategy_quadratic_scale * volatility * volatility;
        const QuadraticExponentialStatistics statistics = loss_enabled
          ? quadratic_exponential_statistics(
              value_grid_size,
              value_grid_minimum,
              value_grid_maximum,
              slope,
              strategy_quadratic_coefficient
            )
          : QuadraticExponentialStatistics {
              quadratic_exponential_log_normalizer(
                value_grid_size,
                value_grid_minimum,
                value_grid_maximum,
                slope,
                strategy_quadratic_coefficient
              ),
              0.0f,
              0.0f,
              0.0f,
            };
        const float log_normalizer = statistics.log_normalizer;
        const float cross_entropy = fmaxf(
          0.0f,
          log_normalizer - slope * value_means[index]
            - strategy_quadratic_coefficient * value_second_moments[index]
        );
        const float weight = value_weights[index];
        distillation_weighted_cross_entropy += static_cast<double>(weight * cross_entropy);
        distillation_weight += weight;
        if (loss_enabled) {
          const float oracle_entropy = value_entropies[index];
          const double normalized_excess_entropy = fmax(
            0.0,
            static_cast<double>(statistics.entropy - oracle_entropy)
              / log(static_cast<double>(value_grid_size))
          );
          distillation_weighted_strategy_entropy += weight * statistics.entropy;
          distillation_weighted_entropy_gap += weight
            * normalized_excess_entropy * normalized_excess_entropy;
          weighted_strategy_mean += weight * statistics.mean;
          weighted_strategy_second_moment += weight * statistics.second_moment;
          weighted_strategy_variance += weight * fmax(
            0.0,
            static_cast<double>(statistics.second_moment)
              - static_cast<double>(statistics.mean) * statistics.mean
          );
          weighted_oracle_mean += weight * value_means[index];
          weighted_oracle_second_moment += weight * value_second_moments[index];
          weighted_oracle_strategy_mean_product += weight
            * value_means[index] * statistics.mean;
          if (precise_features && oracle_mutual_information_mode == 1) {
            const size_t feature = (
              static_cast<size_t>(candidate) * (candle_count - score_start)
              + index - score_start
            ) * 3;
            precise_features[feature] = slope;
            precise_features[feature + 1] = strategy_quadratic_coefficient;
            precise_features[feature + 2] = log_normalizer;
          }
        }
        if (!fitness_only && index + 1 < candle_count) {
          advance_return(
            exposure,
            price,
            close[index + 1],
            oracle_friction,
            value_grid_minimum,
            value_grid_maximum,
            maximum_effective_exposure,
            quote_lend_rate,
            quote_borrow_rate,
            asset_borrow_rate,
            strategy_equity,
            strategy_exposure,
            strategy_peak,
            strategy_max_drawdown,
            strategy_turnover
          );
          if (candidate == 0) {
            advance_return(
              value_optimal_exposures[index],
              price,
              close[index + 1],
              oracle_friction,
              value_grid_minimum,
              value_grid_maximum,
              maximum_effective_exposure,
              quote_lend_rate,
              quote_borrow_rate,
              asset_borrow_rate,
              oracle_equity,
              oracle_exposure,
              oracle_peak,
              oracle_max_drawdown,
              oracle_turnover
            );
          }
        }
      }
    }
  }

  if (!write_transitions) {
    const double inverse_weight = distillation_weight > 0.0 ? 1.0 / distillation_weight : 0.0;
    const double strategy_mean = weighted_strategy_mean * inverse_weight;
    const double strategy_second_moment = weighted_strategy_second_moment * inverse_weight;
    const double state_mutual_information = state_mutual_information_lambda > 0.0f
      ? normalized_gaussian_variance_information(
          fmax(0.0, strategy_second_moment - strategy_mean * strategy_mean),
          weighted_strategy_variance * inverse_weight,
          value_grid_size
        )
      : 0.0;
    const double oracle_mutual_information = oracle_mutual_information_lambda > 0.0f
      && oracle_mutual_information_mode == 0
      ? normalized_gaussian_correlation_information(
          weighted_oracle_mean * inverse_weight,
          weighted_oracle_second_moment * inverse_weight,
          strategy_mean,
          strategy_second_moment,
          weighted_oracle_strategy_mean_product * inverse_weight,
          value_grid_size
        )
      : 0.0;
    const double mixed_loss = distillation_weight > 0.0
      ? distillation_weighted_cross_entropy * inverse_weight
        + entropy_gap_lambda * distillation_weighted_entropy_gap * inverse_weight
        - state_mutual_information_lambda * state_mutual_information
        - oracle_mutual_information_lambda * oracle_mutual_information
      : 0.0;
    results[candidate].state_credit = state_credit;
    results[candidate].signal_count = signal_count;
    results[candidate].distillation_weighted_cross_entropy = distillation_weighted_cross_entropy;
    results[candidate].distillation_weighted_oracle_entropy = 0.0;
    results[candidate].distillation_weight = distillation_weight;
    results[candidate].distillation_opportunity = 0.0;
    results[candidate].distillation_weighted_strategy_entropy =
      distillation_weighted_strategy_entropy;
    results[candidate].distillation_weighted_entropy_gap = distillation_weighted_entropy_gap;
    results[candidate].distillation_state_mutual_information = state_mutual_information;
    results[candidate].distillation_oracle_mutual_information = oracle_mutual_information;
    results[candidate].distillation_mixed_loss = mixed_loss;
    results[candidate].strategy_final_equity = strategy_equity;
    results[candidate].oracle_final_equity = oracle_equity;
    results[candidate].strategy_max_drawdown = strategy_max_drawdown;
    results[candidate].oracle_max_drawdown = oracle_max_drawdown;
    results[candidate].strategy_turnover = strategy_turnover;
    results[candidate].oracle_turnover = oracle_turnover;
  }
}

constexpr int MAX_MUTUAL_INFORMATION_BINS = 32;

__global__ void precise_oracle_mutual_information_joint_kernel(
  const float* oracle_bin_probabilities,
  const float* value_weights,
  const float* precise_features,
  int candle_count,
  int score_start,
  int candidate_count,
  int grid_size,
  float minimum_exposure,
  float maximum_exposure,
  int bins,
  double* precise_joint
) {
  const int lane = blockIdx.x * blockDim.x + threadIdx.x;
  const int candidate = lane / bins;
  const int strategy_bin = lane % bins;
  if (candidate >= candidate_count) return;
  double joint[MAX_MUTUAL_INFORMATION_BINS] = {};
  const int scored_count = candle_count - score_start;
  const int grid_start = (strategy_bin * grid_size + bins - 1) / bins;
  const int grid_end = ((strategy_bin + 1) * grid_size + bins - 1) / bins;
  const float spacing = (maximum_exposure - minimum_exposure) / (grid_size - 1);
  for (int scored_index = 0; scored_index < scored_count; ++scored_index) {
    const size_t feature = (
      static_cast<size_t>(candidate) * scored_count + scored_index
    ) * 3;
    const float linear_coefficient = precise_features[feature];
    const float quadratic_coefficient = precise_features[feature + 1];
    const float log_normalizer = precise_features[feature + 2];
    double strategy_probability = 0.0;
    for (int grid_index = grid_start; grid_index < grid_end; ++grid_index) {
      const float exposure = minimum_exposure + grid_index * spacing;
      strategy_probability += exp(
        static_cast<double>(linear_coefficient) * exposure
        + static_cast<double>(quadratic_coefficient) * exposure * exposure
        - log_normalizer
      );
    }
    const int candle_index = score_start + scored_index;
    const double weighted_strategy_probability = value_weights[candle_index]
      * strategy_probability;
    const size_t oracle_offset = static_cast<size_t>(candle_index) * bins;
    for (int oracle_bin = 0; oracle_bin < bins; ++oracle_bin) {
      joint[oracle_bin] += weighted_strategy_probability
        * oracle_bin_probabilities[oracle_offset + oracle_bin];
    }
  }
  const size_t candidate_offset = static_cast<size_t>(candidate) * bins * bins;
  for (int oracle_bin = 0; oracle_bin < bins; ++oracle_bin) {
    precise_joint[candidate_offset + oracle_bin * bins + strategy_bin] = joint[oracle_bin];
  }
}

__global__ void finalize_precise_oracle_mutual_information_kernel(
  int candidate_count,
  int bins,
  float entropy_gap_lambda,
  float state_mutual_information_lambda,
  float oracle_mutual_information_lambda,
  const double* precise_joint,
  DeviceResult* results
) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) return;
  double oracle_marginal[MAX_MUTUAL_INFORMATION_BINS] = {};
  double strategy_marginal[MAX_MUTUAL_INFORMATION_BINS] = {};
  const size_t offset = static_cast<size_t>(candidate) * bins * bins;
  double total = 0.0;
  for (int oracle_bin = 0; oracle_bin < bins; ++oracle_bin) {
    for (int strategy_bin = 0; strategy_bin < bins; ++strategy_bin) {
      const double value = precise_joint[offset + oracle_bin * bins + strategy_bin];
      total += value;
      oracle_marginal[oracle_bin] += value;
      strategy_marginal[strategy_bin] += value;
    }
  }
  double mutual_information = 0.0;
  double oracle_entropy = 0.0;
  double strategy_entropy = 0.0;
  if (total > 0.0) {
    for (int bin = 0; bin < bins; ++bin) {
      oracle_marginal[bin] /= total;
      strategy_marginal[bin] /= total;
      if (oracle_marginal[bin] > 0.0) {
        oracle_entropy -= oracle_marginal[bin] * log(oracle_marginal[bin]);
      }
      if (strategy_marginal[bin] > 0.0) {
        strategy_entropy -= strategy_marginal[bin] * log(strategy_marginal[bin]);
      }
    }
    for (int oracle_bin = 0; oracle_bin < bins; ++oracle_bin) {
      for (int strategy_bin = 0; strategy_bin < bins; ++strategy_bin) {
        const double probability = precise_joint[
          offset + oracle_bin * bins + strategy_bin
        ] / total;
        if (probability > 0.0) {
          mutual_information += probability * log(
            probability / (oracle_marginal[oracle_bin] * strategy_marginal[strategy_bin])
          );
        }
      }
    }
  }
  const double normalized = oracle_entropy > 2.220446049250313e-16
      && strategy_entropy > 2.220446049250313e-16
    ? fmax(0.0, fmin(1.0, mutual_information / sqrt(oracle_entropy * strategy_entropy)))
    : 0.0;
  DeviceResult& result = results[candidate];
  result.distillation_oracle_mutual_information = normalized;
  const double inverse_weight = result.distillation_weight > 0.0
    ? 1.0 / result.distillation_weight
    : 0.0;
  result.distillation_mixed_loss = result.distillation_weighted_cross_entropy * inverse_weight
    + entropy_gap_lambda * result.distillation_weighted_entropy_gap * inverse_weight
    - state_mutual_information_lambda * result.distillation_state_mutual_information
    - oracle_mutual_information_lambda * normalized;
}

void prepare_precise_fitness_storage(CudaFitnessCase* test_case, int candidate_count) {
  if (test_case->oracle_mutual_information_lambda <= 0.0f
    || test_case->oracle_mutual_information_mode != 1) return;
  const size_t scored_count = test_case->candle_count - test_case->score_start;
  test_case->precise_features.allocate(
    static_cast<size_t>(candidate_count) * scored_count * 3
  );
  test_case->precise_joint.allocate(
    static_cast<size_t>(candidate_count)
      * test_case->mutual_information_bins * test_case->mutual_information_bins
  );
}

void launch_precise_fitness_kernels(
  CudaFitnessCase* test_case,
  int candidate_count,
  cudaStream_t stream = nullptr
) {
  if (test_case->oracle_mutual_information_lambda <= 0.0f
    || test_case->oracle_mutual_information_mode != 1) return;
  const int joint_threads = 128;
  const int joint_lanes = candidate_count * test_case->mutual_information_bins;
  precise_oracle_mutual_information_joint_kernel<<<
    (joint_lanes + joint_threads - 1) / joint_threads,
    joint_threads,
    0,
    stream
  >>>(
    test_case->oracle_bin_probabilities.get(),
    test_case->value_weights.get(),
    test_case->precise_features.get(),
    test_case->candle_count,
    test_case->score_start,
    candidate_count,
    test_case->value_grid_size,
    test_case->value_grid_minimum,
    test_case->value_grid_maximum,
    test_case->mutual_information_bins,
    test_case->precise_joint.get()
  );
  const int finalize_threads = 128;
  finalize_precise_oracle_mutual_information_kernel<<<
    (candidate_count + finalize_threads - 1) / finalize_threads,
    finalize_threads,
    0,
    stream
  >>>(
    candidate_count,
    test_case->mutual_information_bins,
    test_case->entropy_gap_lambda,
    test_case->state_mutual_information_lambda,
    test_case->oracle_mutual_information_lambda,
    test_case->precise_joint.get(),
    test_case->results.get()
  );
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

extern "C" int vw_kama_cuda_prepare_value_oracle(
  const double* prices,
  int price_count,
  int score_start,
  int holding_period_steps,
  int value_horizon_steps,
  int grid_size,
  double minimum_exposure,
  double maximum_exposure,
  double maximum_effective_exposure,
  double temperature,
  double friction,
  double opportunity_epsilon,
  double quote_lend_rate,
  double quote_borrow_rate,
  double asset_borrow_rate,
  double initial_exposure,
  int terminal_index,
  float* means,
  float* second_moments,
  float* modal_exposures,
  float* entropies,
  float* weights,
  float* opportunities,
  float* probabilities,
  float* path_exposures,
  double* path_equities,
  double* path_metrics,
  double* elapsed_ms
) {
  try {
    if (!prices || !means || !second_moments || !modal_exposures
      || !entropies || !weights || !opportunities || !path_exposures
      || !path_equities || !path_metrics
      || !elapsed_ms) {
      throw std::runtime_error("Exposure-value CUDA oracle received a null pointer");
    }
    if (price_count < 2 || score_start < 0 || score_start >= price_count
      || holding_period_steps < 1 || value_horizon_steps < holding_period_steps
      || grid_size < 3 || grid_size > 1024 || terminal_index <= score_start
      || terminal_index >= price_count) {
      throw std::runtime_error("Exposure-value CUDA oracle received invalid dimensions");
    }
    if (!(minimum_exposure < 0.0) || !(maximum_exposure > 0.0)
      || maximum_effective_exposure < fmax(fabs(minimum_exposure), fabs(maximum_exposure))
      || !(temperature > 0.0) || friction < 0.0 || friction >= 1.0
      || opportunity_epsilon < 0.0 || quote_lend_rate < 0.0
      || quote_borrow_rate < 0.0 || asset_borrow_rate < 0.0
      || !std::isfinite(initial_exposure)
      || std::abs(initial_exposure) > maximum_effective_exposure) {
      throw std::runtime_error("Exposure-value CUDA oracle received invalid options");
    }

    DeviceBuffer<double> device_prices(price_count);
    holding_period_steps = std::min(holding_period_steps, price_count - 1);
    value_horizon_steps = std::min(value_horizon_steps, price_count - 1);
    const int scored_count = price_count - score_start;
    const size_t oracle_cells = static_cast<size_t>(scored_count) * grid_size;
    size_t free_device_bytes = 0;
    size_t total_device_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_device_bytes, &total_device_bytes), "query oracle CUDA memory");
    const size_t required_cell_bytes = oracle_cells
      * (4 * sizeof(double) + sizeof(uint16_t));
    if (required_cell_bytes > free_device_bytes * 4 / 5) {
      throw std::runtime_error("Exposure-value CUDA oracle grid exceeds the device-memory admission limit");
    }
    DeviceBuffer<double> continuations(oracle_cells);
    DeviceBuffer<double> alternate_continuations(oracle_cells);
    DeviceBuffer<double> holding_values(oracle_cells);
    DeviceBuffer<double> endpoint_exposures(oracle_cells);
    const bool separable_rebalance_costs = 1.0 - friction * maximum_exposure > 0.0
      && 1.0 - friction + friction * minimum_exposure > 0.0;
    const bool needs_rebalance_matrix = !separable_rebalance_costs;
    DeviceBuffer<double> rebalance_logs(
      needs_rebalance_matrix ? static_cast<size_t>(grid_size) * grid_size : 0
    );
    DeviceBuffer<double> sell_logs(separable_rebalance_costs ? grid_size : 0);
    DeviceBuffer<double> buy_logs(separable_rebalance_costs ? grid_size : 0);
    DeviceBuffer<uint16_t> policy(oracle_cells);
    DeviceBuffer<float> device_means(price_count);
    DeviceBuffer<float> device_second_moments(price_count);
    DeviceBuffer<float> device_modal_exposures(price_count);
    DeviceBuffer<float> device_entropies(price_count);
    DeviceBuffer<float> device_weights(price_count);
    DeviceBuffer<float> device_opportunities(price_count);
    DeviceBuffer<float> device_probabilities(
      probabilities ? static_cast<size_t>(price_count) * grid_size : 0
    );
    DeviceBuffer<float> device_path_exposures(price_count);
    DeviceBuffer<double> device_path_equities(price_count);
    DeviceBuffer<double> device_path_metrics(5);

    cuda_check(cudaMemcpy(
      device_prices.get(),
      prices,
      price_count * sizeof(double),
      cudaMemcpyHostToDevice
    ), "copy exposure-value oracle prices");
    if (separable_rebalance_costs) {
      std::vector<double> host_sell_logs(grid_size);
      std::vector<double> host_buy_logs(grid_size);
      for (int index = 0; index < grid_size; ++index) {
        const double exposure = minimum_exposure
          + static_cast<double>(index) / (grid_size - 1)
            * (maximum_exposure - minimum_exposure);
        host_sell_logs[index] = std::log(1.0 - friction * exposure);
        host_buy_logs[index] = std::log(1.0 - friction + friction * exposure);
      }
      cuda_check(cudaMemcpy(
        sell_logs.get(),
        host_sell_logs.data(),
        host_sell_logs.size() * sizeof(double),
        cudaMemcpyHostToDevice
      ), "copy exposure-value sell fee factors");
      cuda_check(cudaMemcpy(
        buy_logs.get(),
        host_buy_logs.data(),
        host_buy_logs.size() * sizeof(double),
        cudaMemcpyHostToDevice
      ), "copy exposure-value buy fee factors");
    } else if (needs_rebalance_matrix) {
      std::vector<double> host_rebalance_logs(static_cast<size_t>(grid_size) * grid_size);
      for (int current_index = 0; current_index < grid_size; ++current_index) {
        const double current = minimum_exposure
          + static_cast<double>(current_index) / (grid_size - 1)
            * (maximum_exposure - minimum_exposure);
        for (int target_index = 0; target_index < grid_size; ++target_index) {
          const double target = minimum_exposure
            + static_cast<double>(target_index) / (grid_size - 1)
              * (maximum_exposure - minimum_exposure);
          const double factor = host_oracle_rebalance_equity_factor(current, target, friction);
          host_rebalance_logs[static_cast<size_t>(current_index) * grid_size + target_index]
            = factor > 0.0 ? std::log(factor) : -std::numeric_limits<double>::infinity();
        }
      }
      cuda_check(cudaMemcpy(
        rebalance_logs.get(),
        host_rebalance_logs.data(),
        host_rebalance_logs.size() * sizeof(double),
        cudaMemcpyHostToDevice
      ), "copy exposure-value oracle rebalance values");
    }
    cuda_check(cudaMemset(
      continuations.get(),
      0,
      oracle_cells * sizeof(double)
    ), "clear oracle continuation values");
    cuda_check(cudaMemset(device_means.get(), 0, price_count * sizeof(float)), "clear oracle means");
    cuda_check(cudaMemset(device_second_moments.get(), 0, price_count * sizeof(float)), "clear oracle second moments");
    cuda_check(cudaMemset(device_modal_exposures.get(), 0, price_count * sizeof(float)), "clear oracle modes");
    cuda_check(cudaMemset(device_entropies.get(), 0, price_count * sizeof(float)), "clear oracle entropies");
    cuda_check(cudaMemset(device_weights.get(), 0, price_count * sizeof(float)), "clear oracle weights");
    cuda_check(cudaMemset(device_opportunities.get(), 0, price_count * sizeof(float)), "clear oracle opportunities");
    if (probabilities) {
      cuda_check(cudaMemset(
        device_probabilities.get(),
        0,
        static_cast<size_t>(price_count) * grid_size * sizeof(float)
      ), "clear oracle probabilities");
    }
    cuda_check(cudaMemset(
      device_path_exposures.get(),
      0,
      price_count * sizeof(float)
    ), "clear exposure-value path exposures");
    cuda_check(cudaMemset(
      device_path_equities.get(),
      0,
      price_count * sizeof(double)
    ), "clear exposure-value path equities");

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cuda_check(cudaEventCreate(&start_event), "create oracle CUDA start event");
    cuda_check(cudaEventCreate(&stop_event), "create oracle CUDA stop event");
    cuda_check(cudaEventRecord(start_event), "record oracle CUDA start event");
    int threads = 32;
    while (threads < grid_size) threads *= 2;
    const int hold_threads = 256;
    const int hold_blocks = static_cast<int>((oracle_cells + hold_threads - 1) / hold_threads);
    const int terminal_threads = 128;
    const int terminal_blocks = (grid_size + terminal_threads - 1) / terminal_threads;
    auto prepare_holds = [&](int active_price_count, int duration_steps) {
      prepare_value_oracle_holds_kernel<<<hold_blocks, hold_threads>>>(
        device_prices.get(),
        active_price_count,
        score_start,
        duration_steps,
        grid_size,
        minimum_exposure,
        maximum_exposure,
        maximum_effective_exposure,
        friction,
        quote_lend_rate,
        quote_borrow_rate,
        asset_borrow_rate,
        holding_values.get(),
        endpoint_exposures.get()
      );
      cuda_check(cudaGetLastError(), "launch exposure-value hold precomputation kernel");
    };
    const size_t chain_storage_bytes = static_cast<size_t>(threads)
      * (3 * sizeof(double) + 2 * sizeof(uint16_t));
    if (value_horizon_steps >= price_count - 1 - score_start) {
      prepare_holds(price_count, holding_period_steps);
      const int chain_count = std::min(holding_period_steps, scored_count);
      prepare_value_oracle_chains_kernel<<<chain_count, threads, chain_storage_bytes>>>(
        price_count, score_start, holding_period_steps, grid_size,
        minimum_exposure, maximum_exposure, temperature, friction, opportunity_epsilon,
        holding_values.get(), endpoint_exposures.get(), rebalance_logs.get(),
        separable_rebalance_costs, sell_logs.get(), buy_logs.get(), nullptr,
        continuations.get(), policy.get(), device_means.get(), device_second_moments.get(),
        device_modal_exposures.get(), device_entropies.get(), device_weights.get(),
        device_opportunities.get(), probabilities ? device_probabilities.get() : nullptr,
        false, false, true, false
      );
    } else {
      std::vector<int> block_durations;
      for (int remaining = value_horizon_steps; remaining > 0;) {
        const int duration = std::min(holding_period_steps, remaining);
        block_durations.push_back(duration);
        remaining -= duration;
      }
      double* prior = nullptr;
      double* current = continuations.get();
      for (int level = static_cast<int>(block_durations.size()) - 1; level >= 0; --level) {
        const int duration = block_durations[level];
        prepare_holds(price_count, duration);
        prepare_value_oracle_chains_kernel<<<scored_count, threads, chain_storage_bytes>>>(
          price_count, score_start, duration, grid_size,
          minimum_exposure, maximum_exposure, temperature, friction, opportunity_epsilon,
          holding_values.get(), endpoint_exposures.get(), rebalance_logs.get(),
          separable_rebalance_costs, sell_logs.get(), buy_logs.get(), prior,
          current, policy.get(), device_means.get(), device_second_moments.get(),
          device_modal_exposures.get(), device_entropies.get(), device_weights.get(),
          device_opportunities.get(), probabilities ? device_probabilities.get() : nullptr,
          true, prior != nullptr, level == 0, false
        );
        prior = current;
        current = current == continuations.get()
          ? alternate_continuations.get()
          : continuations.get();
      }
    }
    cuda_check(cudaGetLastError(), "launch exposure-value oracle chain kernels");
    const int path_price_count = terminal_index + 1;
    const int path_scored_count = path_price_count - score_start;
    prepare_holds(path_price_count, holding_period_steps);
    initialize_value_oracle_terminal_kernel<<<terminal_blocks, terminal_threads>>>(
      terminal_index,
      score_start,
      grid_size,
      minimum_exposure,
      maximum_exposure,
      friction,
      continuations.get(),
      policy.get()
    );
    cuda_check(cudaGetLastError(), "initialize full-window oracle terminal row");
    const int path_chain_count = std::min(holding_period_steps, path_scored_count);
    prepare_value_oracle_chains_kernel<<<path_chain_count, threads, chain_storage_bytes>>>(
      path_price_count, score_start, holding_period_steps, grid_size,
      minimum_exposure, maximum_exposure, temperature, friction, opportunity_epsilon,
      holding_values.get(), endpoint_exposures.get(), rebalance_logs.get(),
      separable_rebalance_costs, sell_logs.get(), buy_logs.get(), nullptr,
      continuations.get(), policy.get(), device_means.get(), device_second_moments.get(),
      device_modal_exposures.get(), device_entropies.get(), device_weights.get(),
      device_opportunities.get(), nullptr,
      false, false, false, true
    );
    cuda_check(cudaGetLastError(), "launch full-window exposure-value Bellman kernel");
    reconstruct_value_oracle_policy_kernel<<<1, 1>>>(
      device_prices.get(),
      path_price_count,
      score_start,
      holding_period_steps,
      grid_size,
      minimum_exposure,
      maximum_exposure,
      maximum_effective_exposure,
      friction,
      quote_lend_rate,
      quote_borrow_rate,
      asset_borrow_rate,
      initial_exposure,
      holding_values.get(),
      endpoint_exposures.get(),
      continuations.get(),
      device_path_exposures.get(),
      device_path_equities.get(),
      device_path_metrics.get()
    );
    cuda_check(cudaGetLastError(), "launch exposure-value policy kernel");
    cuda_check(cudaEventRecord(stop_event), "record oracle CUDA stop event");
    cuda_check(cudaEventSynchronize(stop_event), "synchronize exposure-value oracle kernels");
    float kernel_elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(
      &kernel_elapsed_ms,
      start_event,
      stop_event
    ), "measure exposure-value oracle kernels");
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    *elapsed_ms = kernel_elapsed_ms;

    const auto copy_output = [price_count](float* host, const DeviceBuffer<float>& device, const char* label) {
      cuda_check(cudaMemcpy(
        host,
        device.get(),
        price_count * sizeof(float),
        cudaMemcpyDeviceToHost
      ), label);
    };
    copy_output(means, device_means, "copy oracle means");
    copy_output(second_moments, device_second_moments, "copy oracle second moments");
    copy_output(modal_exposures, device_modal_exposures, "copy oracle modes");
    copy_output(entropies, device_entropies, "copy oracle entropies");
    copy_output(weights, device_weights, "copy oracle weights");
    copy_output(opportunities, device_opportunities, "copy oracle opportunities");
    copy_output(path_exposures, device_path_exposures, "copy full-window oracle exposures");
    cuda_check(cudaMemcpy(
      path_equities,
      device_path_equities.get(),
      price_count * sizeof(double),
      cudaMemcpyDeviceToHost
    ), "copy full-window oracle equities");
    cuda_check(cudaMemcpy(
      path_metrics,
      device_path_metrics.get(),
      5 * sizeof(double),
      cudaMemcpyDeviceToHost
    ), "copy full-window oracle metrics");
    if (probabilities) {
      cuda_check(cudaMemcpy(
        probabilities,
        device_probabilities.get(),
        static_cast<size_t>(price_count) * grid_size * sizeof(float),
        cudaMemcpyDeviceToHost
      ), "copy oracle probabilities");
    }
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

extern "C" uint64_t vw_kama_cuda_create_fitness_case(
  const double* high,
  const double* low,
  const double* close,
  const double* volume,
  const float* value_means,
  const float* value_second_moments,
  const float* value_entropies,
  const float* value_weights,
  const float* strategy_temperatures,
  const float* strategy_quadratic_volatilities,
  const float* oracle_bin_probabilities,
  int candle_count,
  int score_start,
  double interval_ms,
  int value_holding_period_steps,
  double oracle_friction,
  double quote_lend_rate,
  double quote_borrow_rate,
  double asset_borrow_rate,
  int value_grid_size,
  double value_grid_minimum,
  double value_grid_maximum,
  double maximum_effective_exposure,
  double strategy_quadratic_scale,
  double entropy_gap_lambda,
  double state_mutual_information_lambda,
  double oracle_mutual_information_lambda,
  int oracle_mutual_information_mode,
  int mutual_information_bins,
  int include_high_low,
  double* resident_bytes
) {
  try {
    if (!close || !volume || !value_means || !value_second_moments || !value_entropies
      || !value_weights || !strategy_temperatures || !strategy_quadratic_volatilities
      || !resident_bytes || (include_high_low && (!high || !low))) {
      throw std::runtime_error("CUDA fitness case received a null input pointer");
    }
    if (candle_count <= 0 || score_start < 0 || score_start >= candle_count
      || interval_ms <= 0.0 || value_holding_period_steps < 1 || value_grid_size < 3
      || value_grid_minimum >= 0.0 || value_grid_maximum <= 0.0
      || maximum_effective_exposure < fmax(fabs(value_grid_minimum), fabs(value_grid_maximum))
      || !std::isfinite(strategy_quadratic_scale) || strategy_quadratic_scale < 0.0
      || !std::isfinite(entropy_gap_lambda) || entropy_gap_lambda < 0.0
      || !std::isfinite(state_mutual_information_lambda) || state_mutual_information_lambda < 0.0
      || !std::isfinite(oracle_mutual_information_lambda) || oracle_mutual_information_lambda < 0.0
      || (oracle_mutual_information_mode != 0 && oracle_mutual_information_mode != 1)
      || mutual_information_bins < 2
      || mutual_information_bins > std::min(MAX_MUTUAL_INFORMATION_BINS, value_grid_size)
      || (oracle_mutual_information_lambda > 0.0 && oracle_mutual_information_mode == 1
        && !oracle_bin_probabilities)
      || oracle_friction < 0.0 || quote_lend_rate < 0.0
      || quote_borrow_rate < 0.0 || asset_borrow_rate < 0.0) {
      throw std::runtime_error("CUDA fitness case received invalid dimensions or options");
    }
    cuda_check(cudaSetDevice(0), "select CUDA fitness case device");
    auto result = std::make_unique<CudaFitnessCase>();
    result->candle_count = candle_count;
    result->score_start = score_start;
    result->interval_ms = static_cast<float>(interval_ms);
    result->value_holding_period_steps = value_holding_period_steps;
    result->oracle_friction = static_cast<float>(oracle_friction);
    result->quote_lend_rate = static_cast<float>(quote_lend_rate);
    result->quote_borrow_rate = static_cast<float>(quote_borrow_rate);
    result->asset_borrow_rate = static_cast<float>(asset_borrow_rate);
    result->value_grid_size = value_grid_size;
    result->value_grid_minimum = static_cast<float>(value_grid_minimum);
    result->value_grid_maximum = static_cast<float>(value_grid_maximum);
    result->maximum_effective_exposure = static_cast<float>(maximum_effective_exposure);
    result->strategy_quadratic_scale = static_cast<float>(strategy_quadratic_scale);
    result->entropy_gap_lambda = static_cast<float>(entropy_gap_lambda);
    result->state_mutual_information_lambda = static_cast<float>(state_mutual_information_lambda);
    result->oracle_mutual_information_lambda = static_cast<float>(oracle_mutual_information_lambda);
    result->oracle_mutual_information_mode = oracle_mutual_information_mode;
    result->mutual_information_bins = mutual_information_bins;
    result->has_high_low = include_high_low != 0;

    std::vector<float> host_close(candle_count);
    std::vector<float> host_volume(candle_count);
    std::vector<float> host_high(include_high_low ? candle_count : 0);
    std::vector<float> host_low(include_high_low ? candle_count : 0);
    for (int index = 0; index < candle_count; ++index) {
      host_close[index] = static_cast<float>(close[index]);
      host_volume[index] = static_cast<float>(volume[index]);
      if (include_high_low) {
        host_high[index] = static_cast<float>(high[index]);
        host_low[index] = static_cast<float>(low[index]);
      }
      if (index >= score_start) {
        result->distillation_weight += value_weights[index];
        result->distillation_weighted_oracle_entropy += value_weights[index]
          * value_entropies[index];
      }
    }

    result->close.allocate(candle_count);
    result->volume.allocate(candle_count);
    result->value_means.allocate(candle_count);
    const bool loss_enabled = entropy_gap_lambda > 0.0
      || state_mutual_information_lambda > 0.0
      || oracle_mutual_information_lambda > 0.0;
    result->value_second_moments.allocate(candle_count);
    result->value_entropies.allocate(loss_enabled ? candle_count : 0);
    result->value_weights.allocate(candle_count);
    result->strategy_temperatures.allocate(candle_count);
    result->strategy_quadratic_volatilities.allocate(candle_count);
    const bool precise_enabled = oracle_mutual_information_lambda > 0.0
      && oracle_mutual_information_mode == 1;
    result->oracle_bin_probabilities.allocate(
      precise_enabled ? static_cast<size_t>(candle_count) * mutual_information_bins : 0
    );
    if (include_high_low) {
      result->high.allocate(candle_count);
      result->low.allocate(candle_count);
      cuda_check(cudaMemcpy(result->high.get(), host_high.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident high");
      cuda_check(cudaMemcpy(result->low.get(), host_low.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident low");
    }
    cuda_check(cudaMemcpy(result->close.get(), host_close.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident close");
    cuda_check(cudaMemcpy(result->volume.get(), host_volume.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident volume");
    cuda_check(cudaMemcpy(result->value_means.get(), value_means, candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident value means");
    cuda_check(cudaMemcpy(result->value_second_moments.get(), value_second_moments, candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident value second moments");
    if (loss_enabled) {
      cuda_check(cudaMemcpy(result->value_entropies.get(), value_entropies, candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident value entropies");
    }
    cuda_check(cudaMemcpy(result->value_weights.get(), value_weights, candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident value weights");
    cuda_check(cudaMemcpy(result->strategy_temperatures.get(), strategy_temperatures, candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident strategy temperatures");
    cuda_check(cudaMemcpy(result->strategy_quadratic_volatilities.get(), strategy_quadratic_volatilities, candle_count * sizeof(float), cudaMemcpyHostToDevice), "upload resident strategy quadratic volatilities");
    if (precise_enabled) {
      cuda_check(cudaMemcpy(
        result->oracle_bin_probabilities.get(),
        oracle_bin_probabilities,
        static_cast<size_t>(candle_count) * mutual_information_bins * sizeof(float),
        cudaMemcpyHostToDevice
      ), "upload resident oracle MI bins");
    }
    result->resident_bytes = static_cast<size_t>(candle_count)
      * ((include_high_low ? 9 : 7) + (loss_enabled ? 1 : 0)) * sizeof(float)
      + (precise_enabled
        ? static_cast<size_t>(candle_count) * mutual_information_bins * sizeof(float)
        : 0);
    *resident_bytes = static_cast<double>(result->resident_bytes);
    last_error.clear();
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result.release()));
  } catch (const std::exception& error) {
    last_error = error.what();
    return 0;
  }
}

extern "C" int vw_kama_cuda_evaluate_fitness_case(
  uint64_t handle,
  const void* parameter_data,
  int candidate_count,
  void* output_data
) {
  try {
    auto* test_case = reinterpret_cast<CudaFitnessCase*>(static_cast<uintptr_t>(handle));
    if (!test_case || test_case->magic != CudaFitnessCase::MAGIC
      || !parameter_data || !output_data || candidate_count <= 0) {
      throw std::runtime_error("CUDA fitness evaluation received an invalid case or batch");
    }
    const auto* parameters = static_cast<const VwKamaParams*>(parameter_data);
    auto* output = static_cast<VwKamaResult*>(output_data);
    std::vector<uint64_t> host_change_offsets(candidate_count + 1, 0);
    for (int index = 0; index < candidate_count; ++index) {
      const bool dmi_enabled = parameters[index].confirmation_mix > 0.0f
        && parameters[index].confirmation_dmi_weight > 0.0f;
      if (dmi_enabled && !test_case->has_high_low) {
        throw std::runtime_error("CUDA fitness case omitted high/low columns required by DMI");
      }
      const bool mean_reversion_enabled = parameters[index].mean_reversion_reversal_threshold > 0.0f;
      host_change_offsets[index + 1] = host_change_offsets[index]
        + parameters[index].efficiency_period
        + (mean_reversion_enabled ? parameters[index].mean_reversion_efficiency_period : 0);
    }
    test_case->parameters.allocate(candidate_count);
    test_case->change_offsets.allocate(candidate_count + 1);
    test_case->changes.allocate(host_change_offsets.back());
    test_case->results.allocate(candidate_count);
    prepare_precise_fitness_storage(test_case, candidate_count);
    cuda_check(cudaMemcpy(
      test_case->parameters.get(),
      parameters,
      candidate_count * sizeof(VwKamaParams),
      cudaMemcpyHostToDevice
    ), "upload resident-case parameters");
    cuda_check(cudaMemcpy(
      test_case->change_offsets.get(),
      host_change_offsets.data(),
      host_change_offsets.size() * sizeof(uint64_t),
      cudaMemcpyHostToDevice
    ), "upload resident-case ring offsets");

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cuda_check(cudaEventCreate(&start_event), "create resident fitness start event");
    cuda_check(cudaEventCreate(&stop_event), "create resident fitness stop event");
    cuda_check(cudaEventRecord(start_event), "record resident fitness start event");
    const int threads = 32;
    const int blocks = (candidate_count + threads - 1) / threads;
    evaluate_kernel<<<blocks, threads>>>(
      test_case->high.get(),
      test_case->low.get(),
      test_case->close.get(),
      test_case->volume.get(),
      nullptr,
      test_case->value_means.get(),
      test_case->value_second_moments.get(),
      test_case->value_entropies.get(),
      nullptr,
      test_case->value_weights.get(),
      test_case->strategy_temperatures.get(),
      test_case->strategy_quadratic_volatilities.get(),
      test_case->candle_count,
      test_case->score_start,
      test_case->interval_ms,
      test_case->value_holding_period_steps,
      test_case->oracle_friction,
      test_case->quote_lend_rate,
      test_case->quote_borrow_rate,
      test_case->asset_borrow_rate,
      test_case->value_grid_size,
      test_case->value_grid_minimum,
      test_case->value_grid_maximum,
      test_case->maximum_effective_exposure,
      test_case->strategy_quadratic_scale,
      test_case->entropy_gap_lambda,
      test_case->state_mutual_information_lambda,
      test_case->oracle_mutual_information_lambda,
      test_case->oracle_mutual_information_mode,
      test_case->parameters.get(),
      candidate_count,
      test_case->changes.get(),
      test_case->change_offsets.get(),
      test_case->results.get(),
      nullptr,
      nullptr,
      0,
      test_case->precise_features.get(),
      true,
      false
    );
    cuda_check(cudaGetLastError(), "launch resident VW-KAMA fitness kernel");
    launch_precise_fitness_kernels(test_case, candidate_count);
    cuda_check(cudaGetLastError(), "launch resident precise oracle MI kernels");
    cuda_check(cudaEventRecord(stop_event), "record resident fitness stop event");
    cuda_check(cudaEventSynchronize(stop_event), "synchronize resident fitness kernel");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "measure resident fitness kernel");
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    std::vector<DeviceResult> host_results(candidate_count);
    cuda_check(cudaMemcpy(
      host_results.data(),
      test_case->results.get(),
      candidate_count * sizeof(DeviceResult),
      cudaMemcpyDeviceToHost
    ), "download resident fitness results");
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
      output[candidate] = {
        0.0,
        0.0,
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        static_cast<double>(elapsed_ms),
        host_results[candidate].distillation_weighted_cross_entropy,
        test_case->distillation_weighted_oracle_entropy,
        host_results[candidate].distillation_weight,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        test_case->candle_count - test_case->score_start,
        0,
        0,
        0,
        host_results[candidate].distillation_weighted_strategy_entropy,
        host_results[candidate].distillation_weighted_entropy_gap,
        host_results[candidate].distillation_state_mutual_information,
        host_results[candidate].distillation_oracle_mutual_information,
        host_results[candidate].distillation_mixed_loss,
      };
    }
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

extern "C" int vw_kama_cuda_evaluate_fitness_cases(
  const uint64_t* handles,
  int case_count,
  const void* parameter_data,
  int candidate_count,
  void* output_data
) {
  try {
    if (!handles || case_count <= 0 || !parameter_data || candidate_count <= 0 || !output_data) {
      throw std::runtime_error("CUDA fitness scheduler received an invalid batch");
    }
    const auto* all_parameters = static_cast<const VwKamaParams*>(parameter_data);
    auto* output = static_cast<VwKamaResult*>(output_data);
    std::vector<CudaFitnessCase*> cases(case_count);
    std::vector<std::vector<uint64_t>> host_offsets(case_count);
    std::vector<cudaStream_t> streams(case_count);
    std::vector<cudaEvent_t> starts(case_count);
    std::vector<cudaEvent_t> stops(case_count);

    for (int case_index = 0; case_index < case_count; ++case_index) {
      auto* test_case = reinterpret_cast<CudaFitnessCase*>(
        static_cast<uintptr_t>(handles[case_index])
      );
      if (!test_case || test_case->magic != CudaFitnessCase::MAGIC) {
        throw std::runtime_error("CUDA fitness scheduler received an invalid case handle");
      }
      cases[case_index] = test_case;
      const VwKamaParams* parameters = all_parameters
        + static_cast<size_t>(case_index) * candidate_count;
      auto& offsets = host_offsets[case_index];
      offsets.assign(candidate_count + 1, 0);
      for (int candidate = 0; candidate < candidate_count; ++candidate) {
        const bool dmi_enabled = parameters[candidate].confirmation_mix > 0.0f
          && parameters[candidate].confirmation_dmi_weight > 0.0f;
        if (dmi_enabled && !test_case->has_high_low) {
          throw std::runtime_error("CUDA fitness scheduler case omitted required DMI columns");
        }
        const bool mean_reversion_enabled = parameters[candidate].mean_reversion_reversal_threshold > 0.0f;
        offsets[candidate + 1] = offsets[candidate]
          + parameters[candidate].efficiency_period
          + (mean_reversion_enabled ? parameters[candidate].mean_reversion_efficiency_period : 0);
      }
      test_case->parameters.allocate(candidate_count);
      test_case->change_offsets.allocate(candidate_count + 1);
      test_case->changes.allocate(offsets.back());
      test_case->results.allocate(candidate_count);
      prepare_precise_fitness_storage(test_case, candidate_count);
      cuda_check(cudaMemcpy(
        test_case->parameters.get(),
        parameters,
        candidate_count * sizeof(VwKamaParams),
        cudaMemcpyHostToDevice
      ), "upload scheduled resident-case parameters");
      cuda_check(cudaMemcpy(
        test_case->change_offsets.get(),
        offsets.data(),
        offsets.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice
      ), "upload scheduled resident-case ring offsets");
      cuda_check(cudaStreamCreateWithFlags(&streams[case_index], cudaStreamNonBlocking), "create fitness scheduler stream");
      cuda_check(cudaEventCreate(&starts[case_index]), "create fitness scheduler start event");
      cuda_check(cudaEventCreate(&stops[case_index]), "create fitness scheduler stop event");
    }

    const int threads = 32;
    const int blocks = (candidate_count + threads - 1) / threads;
    for (int case_index = 0; case_index < case_count; ++case_index) {
      CudaFitnessCase* test_case = cases[case_index];
      cuda_check(cudaEventRecord(starts[case_index], streams[case_index]), "record scheduled fitness start");
      evaluate_kernel<<<blocks, threads, 0, streams[case_index]>>>(
        test_case->high.get(),
        test_case->low.get(),
        test_case->close.get(),
        test_case->volume.get(),
        nullptr,
        test_case->value_means.get(),
        test_case->value_second_moments.get(),
        test_case->value_entropies.get(),
        nullptr,
        test_case->value_weights.get(),
        test_case->strategy_temperatures.get(),
        test_case->strategy_quadratic_volatilities.get(),
        test_case->candle_count,
        test_case->score_start,
        test_case->interval_ms,
        test_case->value_holding_period_steps,
        test_case->oracle_friction,
        test_case->quote_lend_rate,
        test_case->quote_borrow_rate,
        test_case->asset_borrow_rate,
        test_case->value_grid_size,
        test_case->value_grid_minimum,
        test_case->value_grid_maximum,
        test_case->maximum_effective_exposure,
        test_case->strategy_quadratic_scale,
        test_case->entropy_gap_lambda,
        test_case->state_mutual_information_lambda,
        test_case->oracle_mutual_information_lambda,
        test_case->oracle_mutual_information_mode,
        test_case->parameters.get(),
        candidate_count,
        test_case->changes.get(),
        test_case->change_offsets.get(),
        test_case->results.get(),
        nullptr,
        nullptr,
        0,
        test_case->precise_features.get(),
        true,
        false
      );
      cuda_check(cudaGetLastError(), "launch scheduled resident VW-KAMA fitness kernel");
      launch_precise_fitness_kernels(test_case, candidate_count, streams[case_index]);
      cuda_check(cudaGetLastError(), "launch scheduled precise oracle MI kernels");
      cuda_check(cudaEventRecord(stops[case_index], streams[case_index]), "record scheduled fitness stop");
    }

    for (int case_index = 0; case_index < case_count; ++case_index) {
      cuda_check(cudaEventSynchronize(stops[case_index]), "synchronize scheduled fitness kernel");
      float elapsed_ms = 0.0f;
      cuda_check(cudaEventElapsedTime(
        &elapsed_ms,
        starts[case_index],
        stops[case_index]
      ), "measure scheduled fitness kernel");
      CudaFitnessCase* test_case = cases[case_index];
      std::vector<DeviceResult> host_results(candidate_count);
      cuda_check(cudaMemcpy(
        host_results.data(),
        test_case->results.get(),
        candidate_count * sizeof(DeviceResult),
        cudaMemcpyDeviceToHost
      ), "download scheduled resident fitness results");
      for (int candidate = 0; candidate < candidate_count; ++candidate) {
        output[static_cast<size_t>(case_index) * candidate_count + candidate] = {
          0.0,
          0.0,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          static_cast<double>(elapsed_ms),
          host_results[candidate].distillation_weighted_cross_entropy,
          test_case->distillation_weighted_oracle_entropy,
          host_results[candidate].distillation_weight,
          0.0,
          1.0,
          1.0,
          0.0,
          0.0,
          0.0,
          0.0,
          test_case->candle_count - test_case->score_start,
          0,
          0,
          0,
          host_results[candidate].distillation_weighted_strategy_entropy,
          host_results[candidate].distillation_weighted_entropy_gap,
          host_results[candidate].distillation_state_mutual_information,
          host_results[candidate].distillation_oracle_mutual_information,
          host_results[candidate].distillation_mixed_loss,
        };
      }
      cudaEventDestroy(starts[case_index]);
      cudaEventDestroy(stops[case_index]);
      cudaStreamDestroy(streams[case_index]);
    }
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}

extern "C" double vw_kama_cuda_fitness_case_device_bytes(uint64_t handle) {
  auto* test_case = reinterpret_cast<CudaFitnessCase*>(static_cast<uintptr_t>(handle));
  if (!test_case || test_case->magic != CudaFitnessCase::MAGIC) return 0.0;
  return static_cast<double>(test_case->resident_bytes
    + test_case->parameters.capacity() * sizeof(VwKamaParams)
    + test_case->change_offsets.capacity() * sizeof(uint64_t)
    + test_case->changes.capacity() * sizeof(float)
    + test_case->results.capacity() * sizeof(DeviceResult)
    + test_case->precise_features.capacity() * sizeof(float)
    + test_case->precise_joint.capacity() * sizeof(double));
}

extern "C" int vw_kama_cuda_destroy_fitness_case(uint64_t handle) {
  try {
    auto* test_case = reinterpret_cast<CudaFitnessCase*>(static_cast<uintptr_t>(handle));
    if (!test_case || test_case->magic != CudaFitnessCase::MAGIC) {
      throw std::runtime_error("CUDA fitness case handle is invalid");
    }
    test_case->magic = 0;
    delete test_case;
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
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
  const float* value_optimal_exposures,
  const float* value_entropies,
  const float* value_weights,
  const float* value_opportunities,
  const float* oracle_bin_probabilities,
  const float* strategy_temperatures,
  const float* strategy_quadratic_volatilities,
  int candle_count,
  int score_start,
  double interval_ms,
  int value_holding_period_steps,
  double oracle_friction,
  double quote_lend_rate,
  double quote_borrow_rate,
  double asset_borrow_rate,
  double match_window_ms,
  double timing_half_life_ms,
  int value_grid_size,
  double value_grid_minimum,
  double value_grid_maximum,
  double maximum_effective_exposure,
  double strategy_quadratic_scale,
  double entropy_gap_lambda,
  double state_mutual_information_lambda,
  double oracle_mutual_information_lambda,
  int oracle_mutual_information_mode,
  int mutual_information_bins,
  const void* parameter_data,
  int candidate_count,
  int fitness_only,
  void* output_data
) {
  try {
    if (!close_times || !high || !low || !close || !volume || !oracle_codes || !parameter_data || !output_data) {
      throw std::runtime_error("VW-KAMA CUDA received a null input pointer");
    }
    if (candle_count <= 0 || candidate_count <= 0 || score_start < 0 || score_start >= candle_count) {
      throw std::runtime_error("VW-KAMA CUDA received invalid dimensions");
    }
    if (interval_ms <= 0 || value_holding_period_steps < 1
      || match_window_ms < 0 || timing_half_life_ms <= 0
      || quote_lend_rate < 0 || quote_borrow_rate < 0 || asset_borrow_rate < 0) {
      throw std::runtime_error("VW-KAMA CUDA received invalid timing options");
    }
    const bool distillation_enabled = value_grid_size >= 3;
    const bool loss_enabled = entropy_gap_lambda > 0.0
      || state_mutual_information_lambda > 0.0
      || oracle_mutual_information_lambda > 0.0;
    const bool precise_enabled = oracle_mutual_information_lambda > 0.0
      && oracle_mutual_information_mode == 1;
    if (distillation_enabled && (
      !value_means
      || !value_second_moments
      || !value_weights
      || !strategy_temperatures
      || !strategy_quadratic_volatilities
      || (loss_enabled && !value_entropies)
      || (precise_enabled && !oracle_bin_probabilities)
      || (!fitness_only && (!value_optimal_exposures || !value_entropies || !value_opportunities))
      || value_grid_minimum >= 0
      || value_grid_maximum <= 0
      || maximum_effective_exposure < fmax(fabs(value_grid_minimum), fabs(value_grid_maximum))
      || !std::isfinite(strategy_quadratic_scale) || strategy_quadratic_scale < 0.0
      || !std::isfinite(entropy_gap_lambda) || entropy_gap_lambda < 0.0
      || !std::isfinite(state_mutual_information_lambda) || state_mutual_information_lambda < 0.0
      || !std::isfinite(oracle_mutual_information_lambda) || oracle_mutual_information_lambda < 0.0
      || (oracle_mutual_information_mode != 0 && oracle_mutual_information_mode != 1)
      || mutual_information_bins < 2
      || mutual_information_bins > std::min(MAX_MUTUAL_INFORMATION_BINS, value_grid_size)
    )) {
      throw std::runtime_error("VW-KAMA CUDA received invalid value-distillation inputs");
    }
    const auto* parameters = static_cast<const VwKamaParams*>(parameter_data);
    auto* output = static_cast<VwKamaResult*>(output_data);

    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) throw std::runtime_error("no CUDA device is available");
    cuda_check(cudaSetDevice(0), "cudaSetDevice");

    bool any_dmi_enabled = false;
    for (int index = 0; index < candidate_count; ++index) {
      any_dmi_enabled = any_dmi_enabled || (
        parameters[index].confirmation_mix > 0.0f
        && parameters[index].confirmation_dmi_weight > 0.0f
      );
    }
    std::vector<float> host_high(any_dmi_enabled ? candle_count : 0);
    std::vector<float> host_low(any_dmi_enabled ? candle_count : 0);
    std::vector<float> host_close(candle_count);
    std::vector<float> host_volume(candle_count);
    for (int index = 0; index < candle_count; ++index) {
      if (any_dmi_enabled) {
        host_high[index] = static_cast<float>(high[index]);
        host_low[index] = static_cast<float>(low[index]);
      }
      host_close[index] = static_cast<float>(close[index]);
      host_volume[index] = static_cast<float>(volume[index]);
    }

    std::vector<uint64_t> host_change_offsets(candidate_count + 1, 0);
    for (int index = 0; index < candidate_count; ++index) {
      const bool mean_reversion_enabled = parameters[index].mean_reversion_reversal_threshold > 0.0f;
      host_change_offsets[index + 1] = host_change_offsets[index]
        + parameters[index].efficiency_period
        + (mean_reversion_enabled ? parameters[index].mean_reversion_efficiency_period : 0);
    }

    DeviceBuffer<float> device_high(any_dmi_enabled ? candle_count : 0);
    DeviceBuffer<float> device_low(any_dmi_enabled ? candle_count : 0);
    DeviceBuffer<float> device_close(candle_count);
    DeviceBuffer<float> device_volume(candle_count);
    DeviceBuffer<uint8_t> device_oracle(fitness_only ? 0 : candle_count);
    DeviceBuffer<float> device_value_means(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_value_second_moments(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_value_entropies(
      distillation_enabled && loss_enabled ? candle_count : 0
    );
    DeviceBuffer<float> device_value_optimal_exposures(
      distillation_enabled && !fitness_only ? candle_count : 0
    );
    DeviceBuffer<float> device_value_weights(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_strategy_temperatures(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_strategy_quadratic_volatilities(distillation_enabled ? candle_count : 0);
    DeviceBuffer<float> device_oracle_bin_probabilities(
      precise_enabled ? static_cast<size_t>(candle_count) * mutual_information_bins : 0
    );
    DeviceBuffer<VwKamaParams> device_parameters(candidate_count);
    DeviceBuffer<uint64_t> device_change_offsets(candidate_count + 1);
    DeviceBuffer<float> device_changes(host_change_offsets.back());
    DeviceBuffer<DeviceResult> device_results(candidate_count);
    DeviceBuffer<float> device_precise_features(
      precise_enabled
        ? static_cast<size_t>(candidate_count) * (candle_count - score_start) * 3
        : 0
    );
    DeviceBuffer<double> device_precise_joint(
      precise_enabled
        ? static_cast<size_t>(candidate_count) * mutual_information_bins * mutual_information_bins
        : 0
    );
    if (any_dmi_enabled) {
      cuda_check(cudaMemcpy(device_high.get(), host_high.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy high");
      cuda_check(cudaMemcpy(device_low.get(), host_low.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy low");
    }
    cuda_check(cudaMemcpy(device_close.get(), host_close.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy close");
    cuda_check(cudaMemcpy(device_volume.get(), host_volume.data(), candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy volume");
    if (!fitness_only) {
      cuda_check(cudaMemcpy(device_oracle.get(), oracle_codes, candle_count, cudaMemcpyHostToDevice), "copy oracle");
    }
    double weighted_oracle_entropy = 0.0;
    double distillation_weight = 0.0;
    double distillation_opportunity = 0.0;
    if (distillation_enabled) {
      cuda_check(cudaMemcpy(device_value_means.get(), value_means, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value means");
      cuda_check(cudaMemcpy(device_value_second_moments.get(), value_second_moments, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value second moments");
      if (loss_enabled) {
        cuda_check(cudaMemcpy(device_value_entropies.get(), value_entropies, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value entropies");
      }
      cuda_check(cudaMemcpy(device_value_weights.get(), value_weights, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value weights");
      cuda_check(cudaMemcpy(device_strategy_temperatures.get(), strategy_temperatures, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy strategy temperatures");
      cuda_check(cudaMemcpy(device_strategy_quadratic_volatilities.get(), strategy_quadratic_volatilities, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy strategy quadratic volatilities");
      if (precise_enabled) {
        cuda_check(cudaMemcpy(
          device_oracle_bin_probabilities.get(),
          oracle_bin_probabilities,
          static_cast<size_t>(candle_count) * mutual_information_bins * sizeof(float),
          cudaMemcpyHostToDevice
        ), "copy oracle MI bins");
      }
      if (!fitness_only) {
        cuda_check(cudaMemcpy(device_value_optimal_exposures.get(), value_optimal_exposures, candle_count * sizeof(float), cudaMemcpyHostToDevice), "copy value optimal exposures");
      }
      for (int index = score_start; index < candle_count; ++index) {
        const double weight = value_weights[index];
        distillation_weight += weight;
        if (!fitness_only) {
          weighted_oracle_entropy += weight * value_entropies[index];
          distillation_opportunity += value_opportunities[index];
        }
      }
    }
    cuda_check(cudaMemcpy(
      device_parameters.get(),
      parameters,
      candidate_count * sizeof(VwKamaParams),
      cudaMemcpyHostToDevice
    ), "copy parameters");
    cuda_check(cudaMemcpy(
      device_change_offsets.get(),
      host_change_offsets.data(),
      host_change_offsets.size() * sizeof(uint64_t),
      cudaMemcpyHostToDevice
    ), "copy candidate ring offsets");

    const int capture_capacity = fitness_only
      ? 0
      : std::min(4096, candle_count - score_start);
    std::vector<int32_t> transition_offsets(candidate_count + 1, 0);
    if (!fitness_only) {
      for (int candidate = 0; candidate <= candidate_count; ++candidate) {
        transition_offsets[candidate] = candidate * capture_capacity;
      }
    }
    DeviceBuffer<int32_t> device_transition_offsets(fitness_only ? 0 : candidate_count);
    DeviceBuffer<uint32_t> device_transitions(
      fitness_only ? 0 : static_cast<size_t>(candidate_count) * capture_capacity
    );
    if (!fitness_only) {
      cuda_check(cudaMemcpy(
        device_transition_offsets.get(),
        transition_offsets.data(),
        candidate_count * sizeof(int32_t),
        cudaMemcpyHostToDevice
      ), "copy VW-KAMA capture offsets");
    }

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    cuda_check(cudaEventCreate(&start_event), "create CUDA start event");
    cuda_check(cudaEventCreate(&stop_event), "create CUDA stop event");
    cuda_check(cudaEventRecord(start_event), "record CUDA start event");

    const int threads = 32;
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
      device_value_optimal_exposures.get(),
      device_value_weights.get(),
      device_strategy_temperatures.get(),
      device_strategy_quadratic_volatilities.get(),
      candle_count,
      score_start,
      static_cast<float>(interval_ms),
      value_holding_period_steps,
      static_cast<float>(oracle_friction),
      static_cast<float>(quote_lend_rate),
      static_cast<float>(quote_borrow_rate),
      static_cast<float>(asset_borrow_rate),
      value_grid_size,
      static_cast<float>(value_grid_minimum),
      static_cast<float>(value_grid_maximum),
      static_cast<float>(maximum_effective_exposure),
      static_cast<float>(strategy_quadratic_scale),
      static_cast<float>(entropy_gap_lambda),
      static_cast<float>(state_mutual_information_lambda),
      static_cast<float>(oracle_mutual_information_lambda),
      oracle_mutual_information_mode,
      device_parameters.get(),
      candidate_count,
      device_changes.get(),
      device_change_offsets.get(),
      device_results.get(),
      device_transition_offsets.get(),
      device_transitions.get(),
      capture_capacity,
      device_precise_features.get(),
      fitness_only != 0,
      false
    );
    cuda_check(cudaGetLastError(), "launch VW-KAMA score kernel");

    if (precise_enabled) {
      const int joint_threads = 128;
      const int joint_lanes = candidate_count * mutual_information_bins;
      precise_oracle_mutual_information_joint_kernel<<<
        (joint_lanes + joint_threads - 1) / joint_threads,
        joint_threads
      >>>(
        device_oracle_bin_probabilities.get(),
        device_value_weights.get(),
        device_precise_features.get(),
        candle_count,
        score_start,
        candidate_count,
        value_grid_size,
        static_cast<float>(value_grid_minimum),
        static_cast<float>(value_grid_maximum),
        mutual_information_bins,
        device_precise_joint.get()
      );
      const int finalize_threads = 128;
      finalize_precise_oracle_mutual_information_kernel<<<
        (candidate_count + finalize_threads - 1) / finalize_threads,
        finalize_threads
      >>>(
        candidate_count,
        mutual_information_bins,
        static_cast<float>(entropy_gap_lambda),
        static_cast<float>(state_mutual_information_lambda),
        static_cast<float>(oracle_mutual_information_lambda),
        device_precise_joint.get(),
        device_results.get()
      );
      cuda_check(cudaGetLastError(), "launch precise oracle MI kernels");
    }

    std::vector<DeviceResult> host_results(candidate_count);
    cuda_check(cudaMemcpy(
      host_results.data(),
      device_results.get(),
      candidate_count * sizeof(DeviceResult),
      cudaMemcpyDeviceToHost
    ), "copy VW-KAMA score results");

    if (fitness_only) {
      cuda_check(cudaEventRecord(stop_event), "record fitness-only CUDA stop event");
      cuda_check(cudaEventSynchronize(stop_event), "synchronize fitness-only VW-KAMA CUDA kernel");
      float elapsed_ms = 0;
      cuda_check(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event), "measure fitness-only CUDA kernel");
      cudaEventDestroy(start_event);
      cudaEventDestroy(stop_event);
      for (int candidate = 0; candidate < candidate_count; ++candidate) {
        output[candidate] = {
          0.0,
          0.0,
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          std::numeric_limits<double>::quiet_NaN(),
          static_cast<double>(elapsed_ms),
          host_results[candidate].distillation_weighted_cross_entropy,
          0.0,
          host_results[candidate].distillation_weight,
          0.0,
          1.0,
          1.0,
          0.0,
          0.0,
          0.0,
          0.0,
          candle_count - score_start,
          0,
          0,
          0,
          host_results[candidate].distillation_weighted_strategy_entropy,
          host_results[candidate].distillation_weighted_entropy_gap,
          host_results[candidate].distillation_state_mutual_information,
          host_results[candidate].distillation_oracle_mutual_information,
          host_results[candidate].distillation_mixed_loss,
        };
      }
      last_error.clear();
      return 0;
    }

    const int actual_transition_count = std::accumulate(
      host_results.begin(),
      host_results.end(),
      0,
      [](int sum, const DeviceResult& result) { return sum + result.signal_count; }
    );
    const bool capture_overflow = std::any_of(
      host_results.begin(),
      host_results.end(),
      [capture_capacity](const DeviceResult& result) {
        return result.signal_count > capture_capacity;
      }
    );
    if (capture_overflow) {
      transition_offsets.assign(candidate_count + 1, 0);
      for (int index = 0; index < candidate_count; ++index) {
        transition_offsets[index + 1] = transition_offsets[index] + host_results[index].signal_count;
      }
      device_transitions.allocate(transition_offsets.back());
      cuda_check(cudaMemcpy(
        device_transition_offsets.get(),
        transition_offsets.data(),
        candidate_count * sizeof(int32_t),
        cudaMemcpyHostToDevice
      ), "copy compact VW-KAMA transition offsets");
    }
    if (capture_overflow && actual_transition_count > 0) {
      evaluate_kernel<<<blocks, threads>>>(
        device_high.get(),
        device_low.get(),
        device_close.get(),
        device_volume.get(),
        device_oracle.get(),
        device_value_means.get(),
        device_value_second_moments.get(),
        device_value_entropies.get(),
        device_value_optimal_exposures.get(),
        device_value_weights.get(),
        device_strategy_temperatures.get(),
        device_strategy_quadratic_volatilities.get(),
        candle_count,
        score_start,
        static_cast<float>(interval_ms),
        value_holding_period_steps,
        static_cast<float>(oracle_friction),
        static_cast<float>(quote_lend_rate),
        static_cast<float>(quote_borrow_rate),
        static_cast<float>(asset_borrow_rate),
        value_grid_size,
        static_cast<float>(value_grid_minimum),
        static_cast<float>(value_grid_maximum),
        static_cast<float>(maximum_effective_exposure),
        static_cast<float>(strategy_quadratic_scale),
        static_cast<float>(entropy_gap_lambda),
        static_cast<float>(state_mutual_information_lambda),
        static_cast<float>(oracle_mutual_information_lambda),
        oracle_mutual_information_mode,
        device_parameters.get(),
        candidate_count,
        device_changes.get(),
        device_change_offsets.get(),
        device_results.get(),
        device_transition_offsets.get(),
        device_transitions.get(),
        candle_count - score_start,
        nullptr,
        false,
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

    const int transition_storage_count = actual_transition_count == 0
      ? 0
      : capture_overflow
        ? transition_offsets.back()
        : candidate_count * capture_capacity;
    std::vector<uint32_t> host_transitions(transition_storage_count);
    if (transition_storage_count > 0) {
      cuda_check(cudaMemcpy(
        host_transitions.data(),
        device_transitions.get(),
        transition_storage_count * sizeof(uint32_t),
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
      const int transition_end = transition_offsets[candidate]
        + host_results[candidate].signal_count;
      for (int index = transition_offsets[candidate]; index < transition_end; ++index) {
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
        weighted_oracle_entropy,
        distillation_weight,
        distillation_opportunity,
        host_results[candidate].strategy_final_equity,
        host_results[0].oracle_final_equity,
        host_results[candidate].strategy_max_drawdown,
        host_results[0].oracle_max_drawdown,
        host_results[candidate].strategy_turnover,
        host_results[0].oracle_turnover,
        candle_count - score_start,
        host_results[candidate].signal_count,
        static_cast<int32_t>(oracle_transitions.size()),
        alignment.matched_count,
        host_results[candidate].distillation_weighted_strategy_entropy,
        host_results[candidate].distillation_weighted_entropy_gap,
        host_results[candidate].distillation_state_mutual_information,
        host_results[candidate].distillation_oracle_mutual_information,
        host_results[candidate].distillation_mixed_loss,
      };
    }
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    return -1;
  }
}
