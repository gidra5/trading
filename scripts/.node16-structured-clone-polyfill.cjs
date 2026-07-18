globalThis.structuredClone ??= (value) => JSON.parse(JSON.stringify(value));
