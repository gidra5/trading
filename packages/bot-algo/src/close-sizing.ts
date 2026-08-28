const DEFAULT_QUANTITY_EPSILON = 0.00000001;

export interface MinimumNotionalCloseSizingInput {
  requestedQuantity: number;
  availableQuantity: number;
  executionPrice: number;
  minNotional: number;
  quantityEpsilon?: number;
}

export function closeQuantityWithoutMinimumNotionalRemainder(
  input: MinimumNotionalCloseSizingInput,
): number {
  const availableQuantity = positiveFinite(input.availableQuantity);
  const requestedQuantity = Math.min(
    availableQuantity,
    positiveFinite(input.requestedQuantity),
  );
  if (requestedQuantity <= 0) {
    return 0;
  }

  const executionPrice = positiveFinite(input.executionPrice);
  const minNotional = positiveFinite(input.minNotional);
  if (executionPrice <= 0 || minNotional <= 0) {
    return requestedQuantity;
  }

  const quantityEpsilon =
    positiveFinite(input.quantityEpsilon) || DEFAULT_QUANTITY_EPSILON;
  const remainingQuantity = Math.max(
    0,
    availableQuantity - requestedQuantity,
  );
  if (
    remainingQuantity > quantityEpsilon &&
    remainingQuantity * executionPrice < minNotional
  ) {
    return availableQuantity;
  }

  return requestedQuantity;
}

function positiveFinite(value: number | undefined): number {
  return value !== undefined && Number.isFinite(value) && value > 0
    ? value
    : 0;
}
