import "@unocss/reset/tailwind.css";
import "virtual:uno.css";
import "./styles.css";
import { createSignal, onCleanup, onMount } from "solid-js";
import { render } from "solid-js/web";
import { App } from "./App";
import { KamaInspectorPage } from "./KamaInspectorPage";
import { MlpTrainingPage } from "./MlpTrainingPage";
import { PortfolioIndexPage } from "./PortfolioIndexPage";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found.");
}

function Root() {
  const [hash, setHash] = createSignal(window.location.hash);
  const update = () => setHash(window.location.hash);
  onMount(() => window.addEventListener("hashchange", update));
  onCleanup(() => window.removeEventListener("hashchange", update));
  if (hash().startsWith("#/mlp-training")) return <MlpTrainingPage />;
  if (hash().startsWith("#/kama-inspector")) return <KamaInspectorPage />;
  if (hash().startsWith("#/portfolio-index")) return <PortfolioIndexPage />;
  return <App />;
}

render(() => <Root />, root);
