import os, json, datetime as dt, urllib.parse
import pandas as pd, requests
from flask import Flask, jsonify, request, send_from_directory
import data, model

app = Flask(__name__, static_folder="static")
LOG = os.path.join(os.path.dirname(__file__), "predictions.json")
_cache = {}


def trained(div, force=False):
    hist = data.history(div, force)
    if hist.empty:
        return None
    key = (div, len(hist), str(hist.Date.max()))
    if key not in _cache:
        now = pd.Timestamp(dt.date.today())
        _cache.clear()
        _cache[key] = (hist, model.fit(hist, now), model.elo_table(hist))
    return _cache[key]


@app.get("/")
def home():
    return send_from_directory("static", "index.html")


@app.get("/api/leagues")
def leagues():
    return jsonify([{"code": k, "country": v[0], "name": v[1]} for k, v in data.LEAGUES.items()])


@app.get("/api/teams")
def teams():
    t = trained(request.args["div"], request.args.get("refresh") == "1")
    if not t: return jsonify({"error": "No data downloaded. Check your internet connection."}), 503
    hist = t[0]
    last = hist.Date.max()
    recent = hist[hist.Date > last - pd.Timedelta(days=400)]
    return jsonify(sorted(set(recent.HomeTeam) | set(recent.AwayTeam)))


@app.get("/api/fixtures")
def fixtures():
    fx = data.fixtures(request.args["div"], request.args.get("refresh") == "1")
    d = request.args.get("date")
    if len(fx) and d:
        fx = fx[fx.Date == pd.Timestamp(d)]
    out = [{"home": r.HomeTeam, "away": r.AwayTeam, "date": r.Date.strftime("%Y-%m-%d"),
            "time": getattr(r, "Time", "")} for r in fx.itertuples()] if len(fx) else []
    return jsonify(out)


def palmares(team):
    """Best-effort history text from TheSportsDB free API."""
    try:
        r = requests.get("https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=" + urllib.parse.quote(team), timeout=8).json()
        t = (r.get("teams") or [None])[0]
        if t:
            return {"founded": t.get("intFormedYear"), "stadium": t.get("strStadium"),
                    "text": (t.get("strDescriptionEN") or "")[:600], "badge": t.get("strBadge")}
    except Exception:
        pass
    return None


@app.get("/api/predict")
def predict():
    div, home_t, away_t = request.args["div"], request.args["home"], request.args["away"]
    force = request.args.get("refresh") == "1"
    t = trained(div, force)
    if not t: return jsonify({"error": "No data available."}), 503
    hist, m, elo = t
    for n in (home_t, away_t):
        if n not in m["idx"]:
            return jsonify({"error": f"No history for {n} in this league."}), 400
    now = pd.Timestamp(dt.date.today())
    mat, xh, xa = model.score_matrix(m, home_t, away_t)
    s = model.summarize(mat)
    market = None
    fx = data.fixtures(div, force)
    if len(fx):
        row = fx[(fx.HomeTeam == home_t) & (fx.AwayTeam == away_t)]
        if len(row): market = data.market_probs(row.iloc[0])
    final, w = model.blend(s["p"], market)
    labels = ["Home win", "Draw", "Away win"]
    pick = labels[max(range(3), key=lambda i: final[i])]
    res = {
        "home": home_t, "away": away_t, "pick": pick, "final": final,
        "model": s["p"], "market": market, "model_weight": w,
        "confidence": model.confidence(final, s["p"], market),
        "xg": [round(xh, 2), round(xa, 2)], "over25": s["over25"], "btts": s["btts"],
        "scores": s["scores"],
        "home_profile": model.team_profile(hist, home_t, elo, now),
        "away_profile": model.team_profile(hist, away_t, elo, now),
        "h2h": model.h2h(hist, home_t, away_t),
        "data_until": hist.Date.max().strftime("%Y-%m-%d"),
        "matches_used": int(len(hist)),
        "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "palmares": {"home": palmares(home_t), "away": palmares(away_t)},
        "note": None if market else "No bookmaker odds found for this match, so this is the statistical model alone.",
    }
    log(div, res)
    return jsonify(res)


def log(div, res):
    try:
        L = json.load(open(LOG)) if os.path.exists(LOG) else []
        L.append({"div": div, "home": res["home"], "away": res["away"],
                  "p": res["final"], "at": res["updated"]})
        json.dump(L[-500:], open(LOG, "w"))
    except Exception:
        pass


@app.get("/api/accuracy")
def accuracy():
    """Compare saved predictions with real results that have since been played."""
    if not os.path.exists(LOG): return jsonify({"n": 0})
    L, done, hit, seen = json.load(open(LOG)), 0, 0, set()
    for e in L:
        key = (e["div"], e["home"], e["away"], e["at"][:10])
        if key in seen: continue
        seen.add(key)
        hist = data.history(e["div"])
        d = pd.Timestamp(e["at"][:10])
        r = hist[(hist.HomeTeam == e["home"]) & (hist.AwayTeam == e["away"]) & (hist.Date >= d)]
        if len(r):
            r = r.iloc[0]
            real = 0 if r.FTHG > r.FTAG else 1 if r.FTHG == r.FTAG else 2
            done += 1
            hit += int(max(range(3), key=lambda i: e["p"][i]) == real)
    return jsonify({"n": done, "correct": hit, "pct": round(100 * hit / done, 1) if done else None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
