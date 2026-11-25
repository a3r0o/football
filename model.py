"""Prediction engine. Exact maths, no random simulation:
 1. Dixon-Coles Poisson with time decay (attack/defence per team, home edge, low-score correction)
 2. Elo rating (strength display + opponent-quality in form)
 3. Bookmaker probabilities (margin removed) blended with the model
"""
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

HALF_LIFE_DAYS = 200
MAXG = 10


def fit(df, now, lam=2.0):
    teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df.HomeTeam.map(idx).values
    ai = df.AwayTeam.map(idx).values
    yh = df.FTHG.values.astype(float)
    ya = df.FTAG.values.astype(float)
    age = (now - df.Date).dt.days.clip(lower=0).values
    w = 0.5 ** (age / HALF_LIFE_DAYS)

    def unpack(p):
        return p[0], p[1], p[2:2 + n], p[2 + n:]

    def f(p):
        c, h, att, dfn = unpack(p)
        mh = np.exp(c + h + att[hi] + dfn[ai])
        ma = np.exp(c + att[ai] + dfn[hi])
        ll = (w * (yh * np.log(mh) - mh + ya * np.log(ma) - ma)).sum()
        rh, ra = w * (yh - mh), w * (ya - ma)
        g = np.zeros_like(p)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g[2:2 + n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n)
        g[2 + n:] = np.bincount(ai, rh, n) + np.bincount(hi, ra, n)
        reg = lam * (p[2:] ** 2).sum()
        greg = np.zeros_like(p)
        greg[2:] = 2 * lam * p[2:]
        return -ll + reg, -g + greg

    p0 = np.zeros(2 + 2 * n)
    p0[0] = math.log(max(yh.mean(), 0.5))
    res = minimize(f, p0, jac=True, method="L-BFGS-B")
    c, h, att, dfn = unpack(res.x)
    # low-score correlation parameter by grid search
    mh = np.exp(c + h + att[hi] + dfn[ai])
    ma = np.exp(c + att[ai] + dfn[hi])
    best, rho = -1e18, 0.0
    for r in np.linspace(-0.25, 0.10, 36):
        t = np.array([_tau(x, y, a, b, r) for x, y, a, b in zip(yh, ya, mh, ma)])
        if (t <= 0).any():
            continue
        s = (w * np.log(t)).sum()
        if s > best:
            best, rho = s, r
    return {"idx": idx, "c": c, "h": h, "att": att, "dfn": dfn, "rho": rho}


def _tau(x, y, mh, ma, rho):
    if x == 0 and y == 0: return 1 - mh * ma * rho
    if x == 0 and y == 1: return 1 + mh * rho
    if x == 1 and y == 0: return 1 + ma * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def score_matrix(m, home, away):
    i, j = m["idx"][home], m["idx"][away]
    mh = math.exp(m["c"] + m["h"] + m["att"][i] + m["dfn"][j])
    ma = math.exp(m["c"] + m["att"][j] + m["dfn"][i])
    g = np.arange(MAXG + 1)
    mat = np.outer(poisson.pmf(g, mh), poisson.pmf(g, ma))
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= _tau(x, y, mh, ma, m["rho"])
    return mat / mat.sum(), mh, ma


def summarize(mat):
    p1 = float(np.tril(mat, -1).sum())
    px = float(np.trace(mat))
    p2 = float(np.triu(mat, 1).sum())
    ii, jj = np.indices(mat.shape)
    over = float(mat[(ii + jj) >= 3].sum())
    btts = float(mat[1:, 1:].sum())
    top = sorted(((mat[a, b], a, b) for a in range(6) for b in range(6)), reverse=True)[:5]
    return {"p": [p1, px, p2], "over25": over, "btts": btts,
            "scores": [{"score": f"{a}-{b}", "prob": float(p)} for p, a, b in top]}


def elo_table(df, k=20, home_adv=60):
    r = {}
    for h, a, gh, ga in zip(df.HomeTeam, df.AwayTeam, df.FTHG, df.FTAG):
        rh, ra = r.get(h, 1500.0), r.get(a, 1500.0)
        e = 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400))
        s = 1.0 if gh > ga else 0.5 if gh == ga else 0.0
        mult = math.log(abs(gh - ga) + 1) + 1
        d = k * mult * (s - e)
        r[h], r[a] = rh + d, ra - d
    return r


def blend(model_p, market_p, w_model=0.45):
    if not market_p:
        return list(model_p), 1.0
    p = [w_model * a + (1 - w_model) * b for a, b in zip(model_p, market_p)]
    s = sum(p)
    return [x / s for x in p], w_model


def confidence(final, model_p, market_p):
    top = max(final)
    gap = sum(abs(a - b) for a, b in zip(model_p, market_p)) / 2 if market_p else 0.15
    score = top - 0.5 * gap
    return "High" if score >= 0.55 else "Medium" if score >= 0.42 else "Low"


def team_profile(df, team, elo, now=None):
    t = df[(df.HomeTeam == team) | (df.AwayTeam == team)]
    rows = []
    for r in t.itertuples():
        home = r.HomeTeam == team
        gf, ga = (r.FTHG, r.FTAG) if home else (r.FTAG, r.FTHG)
        opp = r.AwayTeam if home else r.HomeTeam
        res = "W" if gf > ga else "D" if gf == ga else "L"
        rows.append({"date": r.Date, "home": home, "gf": gf, "ga": ga, "res": res,
                     "opp": opp, "opp_elo": elo.get(opp, 1500)})
    def agg(rs):
        if not rs: return None
        pts = sum(3 if x["res"] == "W" else 1 if x["res"] == "D" else 0 for x in rs)
        return {"n": len(rs), "form": "".join(x["res"] for x in rs[-10:]),
                "ppg": round(pts / len(rs), 2),
                "gf": round(sum(x["gf"] for x in rs) / len(rs), 2),
                "ga": round(sum(x["ga"] for x in rs) / len(rs), 2),
                "clean_sheets": sum(x["ga"] == 0 for x in rs),
                "avg_opp_elo": round(sum(x["opp_elo"] for x in rs) / len(rs))}
    last20 = rows[-20:]
    after_loss = [last20[i + 1] for i in range(len(last20) - 1) if last20[i]["res"] == "L"]
    streak, kind = 0, rows[-1]["res"] if rows else None
    for x in reversed(rows):
        if x["res"] == kind: streak += 1
        else: break
    return {
        "last10": agg(rows[-10:]),
        "home_last": agg([x for x in last20 if x["home"]]),
        "away_last": agg([x for x in last20 if not x["home"]]),
        "after_loss": {"n": len(after_loss), "unbeaten": sum(x["res"] != "L" for x in after_loss)},
        "streak": f"{streak}{kind}" if kind else "-",
        "days_since_last": (now - rows[-1]["date"]).days if rows and now is not None else None,
        "last_date": rows[-1]["date"].strftime("%Y-%m-%d") if rows else None,
        "elo": round(elo.get(team, 1500)),
    }


def h2h(df, home, away, n=6):
    t = df[((df.HomeTeam == home) & (df.AwayTeam == away)) | ((df.HomeTeam == away) & (df.AwayTeam == home))]
    return [{"date": r.Date.strftime("%Y-%m-%d"), "home": r.HomeTeam, "away": r.AwayTeam,
             "score": f"{r.FTHG}-{r.FTAG}"} for r in t.tail(n).itertuples()]
