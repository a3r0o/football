"""Builds the static data files used by docs/index.html. Run by GitHub Actions every hour."""
import os, json, math, datetime as dt
import pandas as pd
import data, model

OUT = os.path.join(os.path.dirname(__file__), "docs", "data")
os.makedirs(OUT, exist_ok=True)


def clean(o):
    """Make everything valid JSON: numpy -> python, NaN/inf -> None."""
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def write_if_changed(path, obj):
    """Only rewrite when real content changed (ignores the 'updated' stamp), so GitHub is not spammed with commits."""
    obj = clean(obj)
    try:
        old = json.load(open(path))
        a = {k: v for k, v in old.items() if k != "updated"}
        b = {k: v for k, v in obj.items() if k != "updated"}
        if a == b:
            return False
    except Exception:
        pass
    json.dump(obj, open(path, "w"), separators=(",", ":"), allow_nan=False)
    return True


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
    write_if_changed(os.path.join(OUT, f"{div}.json"), out)
    return out


if __name__ == "__main__":
    done = []
    for div, (country, name) in data.LEAGUES.items():
        try:
            if build_league(div):
                print("built", div)
        except Exception as e:
            print("skip", div, e)
        if os.path.exists(os.path.join(OUT, f"{div}.json")):   # keep a league even if one download failed
            done.append({"code": div, "country": country, "name": name})
    if done:  # never wipe a good league list if every download failed
        write_if_changed(os.path.join(OUT, "leagues.json"),
                         {"updated": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "leagues": done})
