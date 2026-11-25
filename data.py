"""Free data layer: football-data.co.uk CSVs (results + bookmaker odds + upcoming fixtures).
No API key needed. Files are cached on disk; refresh re-downloads (max once per 5 min per file)."""
import io, os, time, datetime as dt
import pandas as pd, requests

BASE = "https://www.football-data.co.uk"
CACHE = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE, exist_ok=True)
TTL_MIN = int(os.environ.get("CACHE_MINUTES", "60"))
MIN_FORCE_SEC = 300  # be polite to the free host

LEAGUES = {
 "E0": ("England", "Premier League"), "E1": ("England", "Championship"),
 "E2": ("England", "League One"), "E3": ("England", "League Two"),
 "SC0": ("Scotland", "Premiership"), "D1": ("Germany", "Bundesliga"),
 "D2": ("Germany", "2. Bundesliga"), "I1": ("Italy", "Serie A"),
 "I2": ("Italy", "Serie B"), "SP1": ("Spain", "La Liga"),
 "SP2": ("Spain", "Segunda Division"), "F1": ("France", "Ligue 1"),
 "F2": ("France", "Ligue 2"), "N1": ("Netherlands", "Eredivisie"),
 "B1": ("Belgium", "Pro League"), "P1": ("Portugal", "Primeira Liga"),
 "T1": ("Turkey", "Super Lig"), "G1": ("Greece", "Super League"),
}
BOOKS = ["B365", "PS", "WH", "BW", "VC", "Avg", "Max"]


def _fetch(url, force=False):
    name = url.replace("https://", "").replace("/", "_")
    path = os.path.join(CACHE, name)
    age = time.time() - os.path.getmtime(path) if os.path.exists(path) else 1e12
    limit = MIN_FORCE_SEC if force else TTL_MIN * 60
    if age > limit:
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 footpredict"})
            r.raise_for_status()
            open(path, "wb").write(r.content)
        except Exception:
            if not os.path.exists(path):
                return None
    return open(path, "rb").read() if os.path.exists(path) else None


def _read(raw):
    if not raw:
        return pd.DataFrame()
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"])


def season_codes(n=3):
    t = dt.date.today()
    y = t.year if t.month >= 7 else t.year - 1
    return [f"{(y - k) % 100:02d}{(y - k + 1) % 100:02d}" for k in range(n)]


def market_probs(row):
    """Average de-margined bookmaker probabilities across all books present."""
    ps = []
    for b in BOOKS:
        try:
            o = [float(row[b + s]) for s in "HDA"]
        except Exception:
            continue
        if all(x > 1 for x in o):
            inv = [1 / x for x in o]
            tot = sum(inv)
            ps.append([x / tot for x in inv])
    if not ps:
        return None
    return [sum(p[i] for p in ps) / len(ps) for i in range(3)]


def history(div, force=False, seasons=3):
    frames = []
    for code in season_codes(seasons):
        df = _read(_fetch(f"{BASE}/mmz4281/{code}/{div}.csv", force))
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["FTHG", "FTAG"]).sort_values("Date").reset_index(drop=True)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


def fixtures(div=None, force=False):
    df = _read(_fetch(f"{BASE}/fixtures.csv", force))
    if len(df) and div:
        df = df[df["Div"] == div]
    return df.sort_values("Date").reset_index(drop=True) if len(df) else df
