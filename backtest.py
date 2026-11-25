"""Honest test: predict each of the last N matches using ONLY data before it.
Usage: python backtest.py E0 150"""
import sys, pandas as pd
import numpy as np
import data, model

div = sys.argv[1] if len(sys.argv) > 1 else "E0"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
df = data.history(div)
test = df.tail(N)
hit = ll = hit_m = n = 0
for k in range(0, N, 10):                       # refit every 10 matches (fast)
    chunk = test.iloc[k:k + 10]
    train = df[df.Date < chunk.Date.min()]
    if len(train) < 200: continue
    m = model.fit(train, chunk.Date.min())
    for r in chunk.itertuples():
        if r.HomeTeam not in m["idx"] or r.AwayTeam not in m["idx"]: continue
        s = model.summarize(model.score_matrix(m, r.HomeTeam, r.AwayTeam)[0])
        mk = data.market_probs(df.loc[r.Index])
        p, _ = model.blend(s["p"], mk)
        real = 0 if r.FTHG > r.FTAG else 1 if r.FTHG == r.FTAG else 2
        n += 1; hit += int(np.argmax(p) == real); ll += -np.log(p[real])
        if mk: hit_m += int(np.argmax(mk) == real)
print(f"matches {n} | accuracy {100*hit/n:.1f}% | bookmakers alone {100*hit_m/n:.1f}% | log-loss {ll/n:.3f}")
print("Reference: always picking the favourite is ~50-54%; log-loss below 1.0 is good.")
