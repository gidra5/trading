import assert from "node:assert/strict";
import test from "node:test";
import { OracleCrraBot, observableTree, solveCrra, finiteActionMenu, expectedUtility, crra2,
  terminalUtility, type OracleRules, type OracleState, type SolverConfig } from "../src/oracle-crra-bot.js";
import { ONE, ZERO, rat, serializeRat, eq, fraction, toNumber } from "../src/oracle-rational.js";
import { advance, terminalClose } from "../src/oracle-execution.js";

const rules: OracleRules = {
  makerFee: rat("1/10000"), takerFee: rat("1/1000"), reserveLeverage: rat(2),
  entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
  quoteMinimum: rat(1), quoteMaximum: rat(1000), priceStep: rat("1/100"), maxOrders: 1,
};
const initial: OracleState = { quote: rat(100), asset: ZERO, price: rat(100), previousPrice: rat(100), orders: [], time: 0 };
const config: SolverConfig = { riskAversion: 2, rules, funding: () => ONE, menu: finiteActionMenu({
  exposureTargets: [-1, 0, 1].map(rat), limitNotionalFractions: [], limitOffsetsInTicks: [], maximumNewOrders: 0,
}) };

test("Bellman closes early when holding would leave an uncloseable terminal position", () => {
  const limited = { ...rules, quoteMaximum: rat(100) };
  const holding = { ...initial, quote: rat(1000), asset: rat(2) };
  const menu: SolverConfig["menu"] = () => [
    { id: "hold", action: { marketQuote: ZERO, cancelIds: [], place: [] } },
    { id: "close-early", action: { marketQuote: rat(100), cancelIds: [], place: [] } },
  ];
  const tree = observableTree({ children: [{ probability: 1, price: 100 }] }, limited.priceStep);
  const solved = solveCrra(holding, tree, { ...config, rules: limited, menu }, true);
  assert.equal(solved.id, "close-early");
  assert.equal(solved.feasibleCandidates, 1);
  const terminal = terminalClose(advance(solved.state, rat(100), ONE, limited), limited);
  assert(eq(terminal.state.asset, ZERO));
  assert(eq(terminal.state.quote, rat("1199.8")));
  assert.throws(() => solveCrra(holding, tree, { ...config, rules: limited,
    menu: () => [{ id: "hold", action: { marketQuote: ZERO, cancelIds: [], place: [] } }] }), /No feasible/);
  // A supported branch that cannot close invalidates the complete action.
  const risk = observableTree({ children: [{ probability: "0.99", price: 100 },
    { probability: "0.01", price: 101 }] }, limited.priceStep);
  assert.throws(() => solveCrra(holding, risk, { ...config, rules: limited, menu }), /No feasible/);
});

test("expected CRRA maximizes before observation; it does not aggregate clairvoyant actions", () => {
  const tree = observableTree({ children: [{ probability: "1/2", price: 150 }, { probability: "1/2", price: 50 }] }, rules.priceStep);
  const result = solveCrra(initial, tree, config, true);
  assert(eq(result.action.marketQuote, ZERO));
  assert.equal(serializeRat(result.value!), "-1/100");
  assert.equal(result.certificate!.candidates.length, 3);
  const knownUp = solveCrra(initial, observableTree({ children: [{ probability: 1, price: 150 }] }, rules.priceStep), config);
  assert(knownUp.action.marketQuote.n < 0n);
});

test("unobserved scenario labels cannot select different continuation actions", () => {
  const tree = observableTree({ children: [
    { probability: "1/2", price: "100.001", children: [{ probability: 1, price: 150 }] },
    { probability: "1/2", price: "99.999", children: [{ probability: 1, price: 50 }] },
  ] }, rules.priceStep);
  assert.equal(tree.children.length, 1);
  assert.equal(tree.children[0].children.length, 2);
  const result = solveCrra(initial, tree, config, true);
  assert(eq(result.action.marketQuote, ZERO));
  const selected = result.certificate!.candidates.find((c) => c.id === result.id)!;
  assert.equal(selected.branches.length, 1);
  const child = selected.branches[0].child!;
  assert.equal(JSON.parse(child.chosenId)[0], "0");
});

test("positive-probability ruin is bottom; zero-mass ruin contributes nothing", () => {
  assert.equal(expectedUtility([{ probability: rat("1/2"), utility: crra2(rat(200)) },
    { probability: rat("1/2"), utility: crra2(ZERO) }]), null);
  assert.equal(serializeRat(expectedUtility([{ probability: ONE, utility: crra2(rat(100)) },
    { probability: ZERO, utility: null }])!), "-1/100");
  assert.throws(() => expectedUtility([{ probability: rat(2), utility: crra2(rat(100)) }]));
});

test("a passive next-tick fill is available to the solver and uses maker fees", () => {
  const makerConfig: SolverConfig = { ...config, menu: finiteActionMenu({ exposureTargets: [ZERO],
    limitNotionalFractions: [ONE], limitOffsetsInTicks: [100], maximumNewOrders: 1 }) };
  const tree = observableTree({ children: [{ probability: 1, price: 99,
    children: [{ probability: 1, price: 110 }] }] }, rules.priceStep);
  const result = solveCrra(initial, tree, makerConfig);
  assert.equal(result.action.place.length, 1);
  assert.equal(result.action.place[0].side, "buy");
  assert(eq(result.action.marketQuote, ZERO));
});

test("bot executes exact selected account state and enforces tick sequencing", () => {
  const bot = new OracleCrraBot(initial, config);
  const decision = bot.decide(rat(100), { children: [{ probability: 1, price: 150 }] });
  assert.strictEqual(bot.state, decision.state);
  assert.throws(() => bot.decide(rat(100), { children: [] }));
  bot.observe(rat(150));
  assert.equal(bot.state.time, 1);
  assert.throws(() => bot.observe(rat(151)));
  assert(bot.close().wealth.n > 0n);
  assert.throws(() => bot.close());
});

test("normalization is exact and input histories are immutable", () => {
  const tree = Object.freeze({ children: Object.freeze([
    Object.freeze({ probability: 0.1, price: 99 }), Object.freeze({ probability: 0.2, price: 101 }),
  ]) });
  const canonical = observableTree(tree, rules.priceStep);
  assert.equal(serializeRat(canonical.children[0].probability), "1/3");
  assert.throws(() => observableTree({ children: [{ probability: -1, price: 99 }, { probability: 2, price: 101 }] }, rules.priceStep));
});

test("public solver rejects hidden duplicate observations and menu rejects duplicate placements", () => {
  assert.throws(() => solveCrra(initial, { children: [
    { probability: rat("1/2"), price: rat(100), children: [] },
    { probability: rat("1/2"), price: rat(100), children: [] },
  ] }, config), /Indistinguishable/);
  assert.throws(() => finiteActionMenu({ exposureTargets: [], limitNotionalFractions: [ONE],
    limitOffsetsInTicks: [1, 1], maximumNewOrders: 1 }), /Duplicate/);
});

test("an unreachable fee-aware target does not remove a feasible hold action", () => {
  const highRules = { ...rules, takerFee: rat("0.1"), entryLower: rat(-30), entryUpper: rat(30),
    maintenanceLower: rat(-40), maintenanceUpper: rat(40) };
  const drifted = { ...initial, quote: rat(-2400), asset: rat(25) };
  const menu = finiteActionMenu({ exposureTargets: [rat(20)], limitNotionalFractions: [],
    limitOffsetsInTicks: [], maximumNewOrders: 0 });
  assert.doesNotThrow(() => menu({ state: drifted, consumedQuote: ZERO, fills: [] }, { children: [] }, highRules));
});

test("display conversion retains finite ratios when exact account integers exceed float range", () => {
  const large = 10n ** 310n;
  assert(Math.abs(toNumber(fraction(large + 1n, large * 2n + 1n)) - 0.5) < 1e-14);
  assert(Math.abs(toNumber(fraction(10n ** 308n + 1n, 10n ** 310n + 1n)) - 0.01) < 1e-14);
});

test("gamma zero accepts a profitable risky trade that gamma two rejects", () => {
  const tree = observableTree({ children: [{ probability: "1/2", price: 200 },
    { probability: "1/2", price: 50 }] }, rules.priceStep);
  const cautious = solveCrra(initial, tree, config);
  const neutral = solveCrra(initial, tree, { ...config, riskAversion: 0 }, true);
  assert(eq(cautious.action.marketQuote, ZERO));
  assert(neutral.action.marketQuote.n < 0n);
  assert(neutral.value && neutral.value.n > 0n);
  assert.equal(neutral.certificate!.riskAversion, 0);
});

test("gamma zero values ruin finitely and maximizes expected terminal wealth", () => {
  assert.equal(serializeRat(terminalUtility(ZERO, 0)!), "-1");
  assert.equal(terminalUtility(rat(-1), 0), null);
  const risky = expectedUtility([{ probability: rat("1/2"), utility: terminalUtility(rat(300), 0) },
    { probability: rat("1/2"), utility: terminalUtility(ZERO, 0) }]);
  assert.equal(serializeRat(risky!), "149");
  assert.equal(terminalUtility(ZERO, 2), null);
});

test("Bellman continuation carries inventory when separate one-tick roundtrips cannot cover fees", () => {
  const carryRules = { ...rules, takerFee: rat("0.0005"), entryLower: ZERO, entryUpper: ONE,
    maintenanceLower: ZERO, maintenanceUpper: ONE, maxOrders: 0 };
  const carryConfig: SolverConfig = { ...config, riskAversion: 0, rules: carryRules,
    menu: finiteActionMenu({ exposureTargets: [ZERO, ONE], limitNotionalFractions: [],
      limitOffsetsInTicks: [], maximumNewOrders: 0 }) };
  const future = { children: [{ probability: "3/4", price: "100.18" },
    { probability: "1/4", price: "99.99" }] };
  const full = { children: [{ probability: 1, price: "100.06", children: future.children }] };
  const oneTick = solveCrra(initial, observableTree({ children: [{ probability: 1, price: "100.06" }] },
    carryRules.priceStep), carryConfig);
  assert(eq(oneTick.action.marketQuote, ZERO));
  const freshAtChild = { ...initial, price: rat("100.06"), previousPrice: initial.price, time: 1 };
  assert(eq(solveCrra(freshAtChild, observableTree(future, carryRules.priceStep), carryConfig).action.marketQuote, ZERO));
  const bot = new OracleCrraBot(initial, carryConfig);
  const first = bot.decide(initial.price, full, true);
  assert(eq(first.action.marketQuote, rat(-100)));
  assert(first.value && toNumber(first.value) > 99);
  bot.observe(rat("100.06"));
  const inventory = bot.state.asset;
  const second = bot.decide(bot.state.price, future, true);
  assert(eq(second.action.marketQuote, ZERO));
  assert(eq(bot.state.asset, inventory));
  bot.observe(rat("100.18"));
  assert(toNumber(bot.close().wealth) > 100);
});

test("observable feature branches may differ at equal prices; identical observations must merge", () => {
  const forecasts=(keys:readonly string[])=>({children:[
    {probability:'1/2',price:100,observationKey:keys[0],children:[{probability:1,price:150}]},
    {probability:'1/2',price:100,observationKey:keys[1],children:[{probability:1,price:50}]},
  ]});
  const visible=observableTree(forecasts(['observed-feature-up','observed-feature-down']),rules.priceStep);
  const hidden=observableTree(forecasts(['same-observation','same-observation']),rules.priceStep);
  assert.equal(visible.children.length,2);assert.equal(hidden.children.length,1);
  const onlyRootHold:SolverConfig={...config,riskAversion:0,menu:(arrival,tree,r)=>arrival.state.time===0
    ?[{id:'root-hold',action:{marketQuote:ZERO,cancelIds:[],place:[]}}]:config.menu(arrival,tree,r)};
  const informed=solveCrra(initial,visible,onlyRootHold);
  const uninformed=solveCrra(initial,hidden,onlyRootHold);
  assert(toNumber(informed.value!)>toNumber(uninformed.value!)+20);
});
