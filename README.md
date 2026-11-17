# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/saifulsidik801-ship-it/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/saifulsidik801-ship-it/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour: given hour 1's print, predict the 5%, 25%, 50%, 75% and 95% quantiles of each of hours 2–24 of 2026-11-19 (UTC).

The model is fitted at run time from the case's own `comparables` — past big launches' release days. For each hour it regresses the log ratio ln(v_h / v_1) on ln(v_1) with a Huber-robust quadratic, because a launch day's shape depends strongly on how big hour 1 already is (huge first hour: flat or decaying; tiny first hour: several-fold growth as the world unlocks). The spread is size-dependent too: ln|residual| regressed on ln(v_1), blended with the plain residual spread. Leave-one-out pinball loss on the kit's 37 comparable launches: **0.381** overall, **0.201** on launches with hour 1 above 50k viewers — against 0.930 / 0.251 for the house baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network, deterministic, well under a second per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=59gqp20q&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
