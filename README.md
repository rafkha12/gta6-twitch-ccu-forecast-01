# GTA VI Twitch CCU forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rafkha12/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rafkha12/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on make.fun.

## What it does

`main.ts` (Node 22, standard library only) reads one case file `/input/<id>.in` and writes `/output/<id>.out`: for each of hours 2–24 of the release day, five quantiles of that hour's Twitch concurrent viewers in the game's category.

The model is a **size-conditioned comparable launch curve**. Every past launch in the case's comparables contributes its release-day curve as log-ratios to its own hour-1 print. For the case at hand, each comparable is weighted by how close its hour-1 print is to the case's (Gaussian kernel, bandwidth 0.8 on log viewers), and each hour is forecast from the weighted mean and spread of those ratios. A launch that opens huge follows a much flatter curve than one that opens nearly empty and explodes later — pooling them all fits neither.

Validated by leave-one-out over the comparables, running this exact program on one pseudo-case per launch: **0.254 mean pinball loss on big and mid-size launches** (persistence baseline 0.321), **0.212 on big launches alone** (baseline 0.261). The baseline's stated practice loss is 0.387.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` file in the input directory, writes one `.out` per case. Deterministic: two runs write identical bytes. No network, well under the 8 s per-case cap.
