import { defineConfig, type ProxyOptions } from "vite";
import solid from "vite-plugin-solid";
import UnoCSS from "unocss/vite";

const env =
  (globalThis as typeof globalThis & {
    process?: { env?: Record<string, string | undefined> };
  }).process?.env ?? {};

const backendTarget =
  env.TRADING_BACKEND_URL ??
  env.TRADING_API_URL ??
  env.API_URL ??
  env.BACKEND_URL ??
  "http://localhost:3001";

const backendProxy: ProxyOptions = {
  target: backendTarget,
  changeOrigin: true,
  ws: true,
  rewrite: (path) => path.replace(/^\/backend(?=\/|$)/, "") || "/",
};

const proxy = {
  "/backend": backendProxy,
};

function parsePort(value: string | undefined): number | undefined {
  if (!value) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

const webPort = parsePort(env.TRADING_WEB_PORT ?? env.WEB_PORT ?? env.VITE_PORT);

export default defineConfig({
  plugins: [solid(), UnoCSS()],
  build: {
    target: "esnext",
  },
  server: {
    host: "0.0.0.0",
    port: webPort ?? 5173,
    proxy,
  },
  preview: {
    host: "0.0.0.0",
    port: webPort ?? 4173,
    proxy,
  },
});
