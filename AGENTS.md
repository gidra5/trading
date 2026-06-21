This is a hand made trading interface built around binance api.

The main objective of the project - find and implement a trading strategy that will yield the best results, approaching the "perfect margin trader" as much as possible and get profitable regardless of the market conditions.

perfect margin trader - assume that any price movement yields proportional profit, scaled by max leverage. Models the idea that the perfect trader will go full short with max leverage on any downward move, and full long with max leverage on upward move.

There are three main pieces in this project:
1. The trading interface available as a web app
2. The trading agents/bots running on the backend in real time
3. The server backend that stores persistent state, bot controls/settings, backtesting, paper-trading, track various metrics and market histories.

Use a common js stack for all pieces.
The interface is built with solidjs and unocss, with a simple hand made ui kit and design system on top of it.
The stack might include machine learning tools like neural networks and reinforcement learning at some point.

we also an do experimentation to search for optimal strategies and algorithms, record different observations into a documentation. 

Everything should be easy to setup and deploy with single command on a cheap rented server, potentially in a docker container, and must be able to run indefinitely unattended. Both clean setup and restart should be possible.

Expect that I can inspect code at any time and make changes.

Never keep around legacy code, prefer updating the whole codebase instead of complicating it with backwards compatibility.