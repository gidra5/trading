This is a hand made trading interface built around binance api.

The main objective of the project - find and implement a trading strategy that will yield the best results, get profitable regardless of the market conditions.

There are three main pieces in this project:
1. The trading interface available as a web app
2. The trading agents/bots running on the backend in real time
3. The server backend that stores persistent state, bot controls/settings, backtesting, paper-trading, track various metrics and market histories.

The interface is built with solidjs and unocss, with a simple hand made ui kit and design system on top of it.
The stack includes machine learning tools like neural networks and reinforcement learning, decision trees, etc.

we also do experiments to search for optimal strategies and algorithms, record different observations into a documentation. 

Everything should be easy to setup and deploy with single command on a cheap rented server, potentially in a docker container, and must be able to run indefinitely unattended. Both clean setup and restart should be possible.

Expect that I can inspect code at any time and make changes.

Never keep around legacy code, prefer updating the whole codebase instead of complicating it with backwards compatibility.

Treat configs, candles, histories, and other data contracts as immutable by default. Use readonly types and share references. Copy only to construct a changed value or a private mutable builder, rather than cloning inputs and outputs defensively.

We can split code into top-level and bottom-level layers. I will write most of the top-level code, that will describe rough interface for the bottom-level code. You write the bottom-level code, and adjust the top-level code to actually work. Treat is as a kind of pair programming - you can ask for changes at my end, while i implement them and request new changes from you.
