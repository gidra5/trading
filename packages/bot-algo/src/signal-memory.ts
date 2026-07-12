export function signalBeyondFriction(
  price: number,
  lastSignalPrice: number | null,
  friction: number,
): boolean {
  return lastSignalPrice === null
    || friction <= 0
    || Math.abs(price - lastSignalPrice) > lastSignalPrice * friction;
}
