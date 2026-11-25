"""Builds the static data files used by docs/index.html. Run by GitHub Actions every hour."""
import os, json, datetime as dt
import pandas as pd
import data, model

OUT = os.path.join(os.path.dirname(__file__), "docs", "data")
os.makedirs(OUT, exist_ok=True)
conv = lambda o: o.item() if hasattr(o, "item") else str(o)


def build_league(div, force=True):
    hist = data.history(div, force)
    if hist.empty or len(hist) < 150:
        return None
    now = pd.Timestamp(dt.date.today())
    m = model.fit(hist, now)
    elo = model.elo_table(hist)
    recent = hist[hist.Date > hist.Date.max() - pd.Timedelta(days=400)]
    teams = sorted(set(recent.HomeTeam) | set(recent.AwayTeam))
    names = sorted(m["idx"])
    fx = data.fixtures(div, force)
    fixtures = []
    if len(fx):
        for _, r in fx[fx.Date >= now - pd.Timedelta(days=1)].iterrows():
            fixtures.append({"home": r.HomeTeam, "away": r.AwayTeam, "date": r.Date.strftime("%Y-%m-%d"),
                             "time": str(r.get("Time", "") if pd.notna(r.get("Time", "")) else ""),
                             "market": data.market_probs(r)})
    out = {
        "updated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "data_until": hist.Date.max().strftime("%Y-%m-%d"),
        "matches_used": int(len(hist)),
        "teams": teams,
        "params": {"c": m["c"], "h": m["h"], "rho": m["rho"],
                   "t": {n: [float(m["att"][m["idx"][n]]), float(m["dfn"][m["idx"][n]])] for n in names}},
        "profiles": {t: model.team_profile(hist, t, elo, now) for t in teams},
        "names": names,
        "matches": [[r.Date.strftime("%Y-%m-%d"), names.index(r.HomeTeam), names.index(r.AwayTeam), r.FTHG, r.FTAG]
                    for r in hist.itertuples()],
        "fixtures": fixtures,
    }
    json.dump(out, open(os.path.join(OUT, f"{div}.json"), "w"), default=conv, separators=(",", ":"))
    return out


if __name__ == "__main__":
    done = []
    for div, (country, name) in data.LEAGUES.items():
        try:
            if build_league(div):
                done.append({"code": div, "country": country, "name": name})
                print("built", div)
        except Exception as e:
            print("skip", div, e)
    json.dump({"updated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "leagues": done},
              open(os.path.join(OUT, "leagues.json"), "w"))
