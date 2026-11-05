import "@unocss/reset/tailwind.css";
import "virtual:uno.css";
import "./styles.css";
import { Match, Suspense, Switch, createSignal, lazy, onCleanup, onMount } from "solid-js";
import { render } from "solid-js/web";
const App = lazy(() => import("./App").then(({ App }) => ({ default: App })));
const KamaInspectorPage = lazy(() => import("./KamaInspectorPage").then(({ KamaInspectorPage }) => ({ default: KamaInspectorPage })));
const MlpTrainingPage = lazy(() => import("./MlpTrainingPage").then(({ MlpTrainingPage }) => ({ default: MlpTrainingPage })));
const PortfolioIndexPage = lazy(() => import("./PortfolioIndexPage").then(({ PortfolioIndexPage }) => ({ default: PortfolioIndexPage })));

const root = document.getElementById("root");

if (!root) {
  throw new Error("Root element not found.");
}

function Root() {
  const [hash, setHash] = createSignal(window.location.hash);
  const update = () => setHash(window.location.hash);
  onMount(() => window.addEventListener("hashchange", update));
  onCleanup(() => window.removeEventListener("hashchange", update));
  return (
    <Suspense fallback={<div class="p-6">Loading…</div>}>
      <Switch fallback={<App />}>
        <Match when={hash().startsWith("#/mlp-training")}><MlpTrainingPage /></Match>
        <Match when={hash().startsWith("#/kama-inspector")}><KamaInspectorPage /></Match>
        <Match when={hash().startsWith("#/portfolio-index")}><PortfolioIndexPage /></Match>
      </Switch>
    </Suspense>
  );
}

render(() => <Root />, root);
