"""
College football player props: build pipeline.

Downloads cfbfastR / ESPN data from the sportsdataverse releases, trains every market
on one season and tests it blind on the next, prices the upcoming week (lines from ESPN's
public scoreboard feed), and writes a self-contained site/index.html.

Usage:  python cfb_build.py                 # auto-detect season and week
        python cfb_build.py --season 2026 --week 4
"""
import argparse, datetime as dt, itertools, json, math, os, urllib.request
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
REL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard?groups=80&limit=400&seasontype=2&week={week}"
os.makedirs(DATA, exist_ok=True)

POSMAP = {"8": "QB", "9": "RB", "1": "WR", "7": "TE"}
SKILL = {"QB", "RB", "WR", "TE"}
PBP_COLS = ["game_id", "id_play", "week", "season", "season_type", "pos_team", "def_pos_team", "home_team", "away_team",
            "home_team_id", "away_team_id", "home_team_division", "away_team_division", "home_team_conference", "away_team_conference",
            "play_type", "yards_to_goal", "yards_gained", "rush", "pass", "rush_td", "pass_td",
            "rush_player_id", "reception_player_id", "target_player_id", "completion_player_id", "incompletion_player_id", "play_text"]


def fetch(url, name, refresh):
    path = os.path.join(DATA, name)
    if os.path.exists(path) and not refresh:
        return path
    try:
        urllib.request.urlretrieve(url, path); return path
    except Exception as e:
        print(f"  ! {name}: {e}"); return path if os.path.exists(path) else None


def pq(tag, name, refresh=False, columns=None):
    p = fetch(f"{REL}{tag}/{name}.parquet", f"{tag}_{name}.parquet", refresh)
    if not p: return None
    try: return pd.read_parquet(p, columns=columns)
    except Exception: return pd.read_parquet(p)


ap = argparse.ArgumentParser()
ap.add_argument("--season", type=int); ap.add_argument("--week", type=int)
ap.add_argument("--out", default=os.path.join(ROOT, "site", "index.html"))
args = ap.parse_args()

today = dt.date.today()
S = args.season or (today.year if today.month >= 3 else today.year - 1)
print("Season", S)
SEASONS = [S - 3, S - 2, S - 1, S]
TRAIN, TEST = S - 2, S - 1

raw, sched, bet, rost = {}, {}, {}, {}
for y in SEASONS:
    fresh = y == S
    raw[y] = pq("cfbfastR_cfb_pbp", f"play_by_play_{y}", fresh, PBP_COLS)
    sched[y] = pq("espn_cfb_schedules", f"cfb_schedule_{y}", fresh)
    bet[y] = pq("espn_cfb_betting", f"betting_{y}", fresh)
    rost[y] = pq("espn_cfb_game_rosters", f"game_rosters_{y}", fresh,
                 ["athlete_id", "full_name", "jersey", "position_href", "team_id", "team_abbreviation", "team_short_display_name",
                  "team_color", "team_alternate_color", "did_not_play", "game_id", "week"])

# ----------------------------------------------------------------------------- players and teams
POS, NAME, JERSEY = {}, {}, {}
for y in SEASONS:
    r = rost[y]
    if r is None or not len(r): continue
    r = r.copy(); r["pc"] = r.position_href.astype(str).str.extract(r"positions/(\d+)")[0].map(POSMAP)
    r = r.sort_values("week")
    for a, pc, nm, j in zip(r.athlete_id.astype(str), r.pc, r.full_name, r.jersey):
        if isinstance(pc, str): POS[a] = pc
        if isinstance(nm, str): NAME[a] = nm
        if isinstance(j, str) and j.isdigit(): JERSEY[a] = int(j)

TEAMS = {}  # id -> dict(abbr, short, c1, c2)
for y in SEASONS:
    r = rost[y]
    if r is None or not len(r): continue
    for t in r.drop_duplicates("team_id", keep="last").itertuples():
        TEAMS[str(t.team_id)] = dict(abbr=t.team_abbreviation, short=t.team_short_display_name,
                                     c1="#" + str(t.team_color or "555555"), c2="#" + str(t.team_alternate_color or "999999"))


import re, unicodedata
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?")


def normname(s):
    s = unicodedata.normalize("NFD", str(s)).encode("ascii", "ignore").decode().lower()
    s = SUFFIX.sub("", s.replace(".", " ").replace("'", "").replace("-", " "))
    return " ".join(s.split())


NAMEIDX = {}  # (season, team_id) -> {normalized key: athlete_id}
JERSEYIDX = {}  # (season, team_id) -> {jersey: [(athlete_id, normalized name)]}
for y in SEASONS:
    rr = rost[y]
    if rr is None or not len(rr): continue
    d = rr[["team_id", "athlete_id", "full_name", "jersey"]].dropna(subset=["team_id", "athlete_id", "full_name"]).drop_duplicates(["team_id", "athlete_id"])
    for team, j, a, nm in zip(d.team_id.astype(str), d.jersey, d.athlete_id.astype(str), d.full_name):
        if isinstance(j, str) and j.isdigit(): JERSEYIDX.setdefault((y, team), {}).setdefault(int(j), []).append((a, normname(nm)))
    for team, g in d.groupby("team_id"):
        full, init = {}, {}
        for a, nm in zip(g.athlete_id.astype(str), g.full_name):
            n = normname(nm); parts = n.split()
            if not parts: continue
            full[n] = a
            if len(parts) >= 2:
                k = parts[0][0] + " " + " ".join(parts[1:]); init.setdefault(k, set()).add(a)
        idx = dict(full); idx.update({k: next(iter(v)) for k, v in init.items() if len(v) == 1})
        NAMEIDX[(y, str(team))] = idx

JPASS = re.compile(r"#(\d+)\s+([^#(]+?)\s+pass\b", re.I)
PASS_RX = [re.compile(r"^(?:\(\d+:\d+\)\s*)?(?:[\w-]+\s+)?([A-Z][^#(]+?) pass (?:complete|to)", re.I), re.compile(r"pass from ([^(]+?)(?: \(|$)", re.I)]
JRUSH = re.compile(r"#(\d+)\s+([^#(]+?)\s+(?:rush|run)\b", re.I)
JREC = re.compile(r"pass(?: complete| incomplete)?[^#]{0,40}?\bto\s+#(\d+)\s+([^#(,]+?)\s+(?:for|caught|at|out)\b", re.I)
RUSH_RX = [re.compile(r"^(.+?) (?:run for|rush for|\d+ yd run|\d+ yard run)", re.I)]
REC_RX = [re.compile(r"pass (?:complete )?to (.+?) for", re.I), re.compile(r"^(.+?) \d+ y(?:d|ard)s? pass from", re.I),
          re.compile(r"(?:^|\s)to ([A-Z][^,#(]+?) for -?\d+ y", re.I)]


def by_jersey(y, team, jersey, name):
    cands = (JERSEYIDX.get((y, team)) or {}).get(int(jersey), [])
    if len(cands) == 1: return cands[0][0]
    last = normname(name).split()[-1] if isinstance(name, str) and normname(name) else ""
    hit = [a for a, n in cands if n.split() and n.split()[-1] == last]
    return hit[0] if len(hit) == 1 else None


def lookup(y, team, name, other=None):
    a = _lookup(y, team, name)
    return a if a or other is None else _lookup(y, other, name)


def _lookup(y, team, name):
    if not isinstance(name, str): return None
    idx = NAMEIDX.get((y, team)) or {}
    n = normname(name)
    if n in idx: return idx[n]
    parts = n.split()
    if len(parts) >= 2 and len(parts[0]) == 1: return idx.get(parts[0] + " " + " ".join(parts[1:]))
    if len(parts) >= 2: return idx.get(parts[0][0] + " " + " ".join(parts[1:]))
    return None


def recover(p, y):
    """Fill missing scorer IDs on touchdown plays by reading the name from the play text."""
    fixed = 0
    for i in p.index[(p.rush_td == 1) & p.rush_player_id.isna() & p.play_text.notna()]:
        m = JRUSH.search(p.at[i, "play_text"])
        if m and (a := by_jersey(y, p.at[i, "posteam"], m.group(1), m.group(2).replace(".", ". "))): p.at[i, "rush_player_id"] = a; fixed += 1; continue
        for rx in RUSH_RX:
            m = rx.search(p.at[i, "play_text"])
            if m and (a := lookup(y, p.at[i, "posteam"], m.group(1))): p.at[i, "rush_player_id"] = a; fixed += 1; break
    for i in p.index[(p.pass_td == 1) & p.reception_player_id.isna() & p.play_text.notna()]:
        m = JREC.search(p.at[i, "play_text"])
        if m and (a := by_jersey(y, p.at[i, "posteam"], m.group(1), m.group(2).replace(".", ". "))): p.at[i, "reception_player_id"] = a; fixed += 1; continue
        for rx in REC_RX:
            m = rx.search(p.at[i, "play_text"])
            if m and (a := lookup(y, p.at[i, "posteam"], m.group(1), p.at[i, "defteam"])): p.at[i, "reception_player_id"] = a; fixed += 1; break
    for i in p.index[(p.pass_td == 1) & p.completion_player_id.isna() & p.play_text.notna()]:
        m = JPASS.search(p.at[i, "play_text"])
        if m and (a := by_jersey(y, p.at[i, "posteam"], m.group(1), m.group(2).replace(".", ". "))): p.at[i, "completion_player_id"] = a; fixed += 1; continue
        for rx in PASS_RX:
            m = rx.search(p.at[i, "play_text"])
            if m and (a := lookup(y, p.at[i, "posteam"], m.group(1), p.at[i, "defteam"])): p.at[i, "completion_player_id"] = a; fixed += 1; break
    return fixed


def prep(p, y):
    p = p[(p.season_type == "regular")].drop_duplicates("id_play").copy()
    name2id = {}
    for side in ("home", "away"):
        for n, i in zip(p[f"{side}_team"], p[f"{side}_team_id"]):
            if pd.notna(i): name2id[n] = str(int(i))
    p["posteam"] = p.pos_team.map(name2id); p["defteam"] = p.def_pos_team.map(name2id)
    p = p[p.posteam.notna() & p.defteam.notna()]
    for c in ["rush_player_id", "reception_player_id", "target_player_id", "completion_player_id", "incompletion_player_id"]:
        p[c] = p[c].map(lambda v: str(int(v)) if pd.notna(v) else None)
    n_fix = recover(p, y)
    if n_fix: print(f"  {y}: recovered {n_fix} touchdown scorers from play text")
    notsack = ~p.play_type.fillna("").str.contains("Sack")
    p["recv"] = p.reception_player_id.fillna(p.target_player_id)
    p["passer"] = p.completion_player_id.fillna(p.incompletion_player_id)
    p["is_rush"] = (p.rush == 1) & p.rush_player_id.notna() & notsack
    p["is_tgt"] = (p["pass"] == 1) & p.recv.notna() & notsack
    p["is_pa"] = (p["pass"] == 1) & p.passer.notna() & notsack
    p["rz"] = p.yards_to_goal <= 20; p["i10"] = p.yards_to_goal <= 10
    p["ez"] = p.is_tgt & p.i10                     # college: inside-10 targets stand in for end zone targets
    p["cyd"] = np.where(p.reception_player_id.notna(), p.yards_gained.fillna(0), 0.0)
    p["rtd"] = ((p.rush_td == 1) & p.is_rush).astype(int)
    p["ptd"] = ((p.pass_td == 1) & p.is_tgt).astype(int)
    p["td_player_id"] = np.where(p.rtd == 1, p.rush_player_id, np.where(p.ptd == 1, p.recv, None))
    return p


pbp = {y: prep(raw[y], y) if raw[y] is not None and len(raw[y]) else None for y in SEASONS}
FBS = set()
for y in SEASONS:
    p = raw[y]
    if p is None: continue
    for side in ("home", "away"):
        FBS |= set(p.loc[p[f"{side}_team_division"] == "fbs", f"{side}_team_id"].dropna().astype(int).astype(str))
CONF = {}
for y in SEASONS:
    p = raw[y]
    if p is None: continue
    for side in ("home", "away"):
        for i, c in zip(p[f"{side}_team_id"], p[f"{side}_team_conference"]):
            if pd.notna(i) and isinstance(c, str): CONF[str(int(i))] = c
empty = pbp[S - 1].iloc[0:0]
for y in SEASONS:
    if pbp[y] is None: pbp[y] = empty
print("FBS teams:", len(FBS))

# ----------------------------------------------------------------------------- shared helpers (same logic as the NFL model)
def team_game_totals(p):
    q = p.assign(rzr=p.is_rush & p.rz, i10r=p.is_rush & p.i10, rzt=p.is_tgt & p.rz)
    return q.groupby(["game_id", "posteam"]).agg(rush=("is_rush", "sum"), rz_rush=("rzr", "sum"), i10_rush=("i10r", "sum"),
                                                 tgt=("is_tgt", "sum"), rz_tgt=("rzt", "sum"), ez_tgt=("ez", "sum")).reset_index()


def events(p):
    r = p[p.is_rush][["game_id", "week", "posteam", "rush_player_id", "rz", "i10", "rtd", "yards_gained"]].rename(columns={"rush_player_id": "pid"})
    t = p[p.is_tgt][["game_id", "week", "posteam", "recv", "rz", "ez", "ptd", "cyd"]].rename(columns={"recv": "pid"})
    return r, t


TG = {y: team_game_totals(pbp[y]) for y in SEASONS}
EV = {y: events(pbp[y]) for y in SEASONS}
METRICS = ["s_rush", "s_rz_rush", "s_i10_rush", "s_tgt", "s_rz_tgt", "s_ez_tgt"]


def shares(y, max_week=None):
    r, t = EV[y]; tg = TG[y]
    if max_week is not None:
        r, t = r[r.week < max_week], t[t.week < max_week]
        tg = tg[tg.game_id.isin(set(pbp[y][pbp[y].week < max_week].game_id))]
    pr = r.groupby(["pid", "posteam"]).agg(rush=("rz", "size"), rz_rush=("rz", "sum"), i10_rush=("i10", "sum"), rush_td=("rtd", "sum"), ryd=("yards_gained", "sum"))
    pt = t.groupby(["pid", "posteam"]).agg(tgt=("rz", "size"), rz_tgt=("rz", "sum"), ez_tgt=("ez", "sum"), rec_td=("ptd", "sum"), cyd=("cyd", "sum"))
    df = pr.join(pt, how="outer").fillna(0).reset_index()
    app = pd.concat([r[["pid", "posteam", "game_id"]], t[["pid", "posteam", "game_id"]]]).drop_duplicates()
    tot = app.merge(tg, on=["game_id", "posteam"]).groupby(["pid", "posteam"]).agg(
        g=("game_id", "nunique"), T_rush=("rush", "sum"), T_rz_rush=("rz_rush", "sum"), T_i10_rush=("i10_rush", "sum"),
        T_tgt=("tgt", "sum"), T_rz_tgt=("rz_tgt", "sum"), T_ez_tgt=("ez_tgt", "sum")).reset_index()
    df = df.merge(tot, on=["pid", "posteam"])
    for m in ["rush", "rz_rush", "i10_rush", "tgt", "rz_tgt", "ez_tgt"]:
        df["s_" + m] = np.where(df["T_" + m] > 0, df[m] / df["T_" + m].replace(0, 1), 0.0)
    return df.rename(columns={"posteam": "team"})


def team_td_split(y, max_week=None):
    p = pbp[y] if max_week is None else pbp[y][pbp[y].week < max_week]
    o = p.groupby("posteam").agg(rtd=("rtd", "sum"), ptd=("ptd", "sum"))
    d = p.groupby("defteam").agg(rtd=("rtd", "sum"), ptd=("ptd", "sum"))
    return o, d


def lines(y):
    """game_id -> (home, away, home_margin, total) from ESPN schedules + betting files."""
    s, b = sched[y], bet[y]
    if s is None or b is None: return {}
    m = s.merge(b[["game_id", "home_team_spread", "over_under"]], on="game_id", how="left")
    out = {}
    for g in m.itertuples():
        if pd.isna(g.home_team_spread) or pd.isna(g.over_under): continue
        out[str(g.game_id)] = (str(g.home_id), str(g.away_id), -float(g.home_team_spread), float(g.over_under), int(g.week))
    return out


LINES = {y: {g: v for g, v in lines(y).items() if v[0] in FBS and v[1] in FBS} for y in SEASONS}   # FBS vs FBS only


def tinfo_for(T, wk):
    t = {}
    for gid, (h, a, mg, tot, w) in LINES[T].items():
        if w != wk: continue
        t[h] = (tot / 2 + mg / 2, a, mg, tot); t[a] = (tot / 2 - mg / 2, h, -mg, tot)
    return t


def td_per_pt(years):
    num = den = 0.0
    for y in years:
        tds = pbp[y].groupby(["game_id", "posteam"]).apply(lambda d: d.rtd.sum() + d.ptd.sum(), include_groups=False)
        for gid, (h, a, mg, tot, w) in LINES[y].items():
            for team, imp in ((h, tot / 2 + mg / 2), (a, tot / 2 - mg / 2)):
                if team in FBS and (int(gid), team) in tds.index:
                    num += tds[(int(gid), team)]; den += imp
    return num / den


A = td_per_pt([S - 2, S - 1])
lp = pd.concat([pbp[S - 2], pbp[S - 1]])
LG_R = float(lp.rtd.sum() / (lp.rtd.sum() + lp.ptd.sum()))
print(f"TDs per implied point {A:.4f}, league rush TD share {LG_R:.3f}")


def team_r(team, opp, oc, dc, op, dp, beta, m=10):
    def frac(a, b, t):
        r = a.rtd.get(t, 0) * 2 + b.rtd.get(t, 0); tot = r + a.ptd.get(t, 0) * 2 + b.ptd.get(t, 0)
        return (r + m * LG_R) / (tot + m)
    off, de = frac(oc, op, team), frac(dc, dp, opp)
    return float(np.clip(off + beta * (de - LG_R), 0.15, 0.75)), off, de


def assemble(cur, prior, team_of):
    cg = cur.set_index(["pid", "team"]); pg = {k: v for k, v in prior.groupby("pid")}
    rows = []
    for pid, team in team_of.items():
        row = {"pid": pid, "team": team}
        if (pid, team) in cg.index:
            c = cg.loc[(pid, team)]
            row.update(n=int(c.g), td=int(c.rush_td + c.rec_td), c_car=float(c.rush), c_ryd=float(c.ryd), c_tg=float(c.tgt), c_cyd=float(c.cyd),
                       **{"c_" + m: float(c[m]) for m in METRICS})
        else:
            row.update(n=0, td=0, c_car=0.0, c_ryd=0.0, c_tg=0.0, c_cyd=0.0, **{"c_" + m: 0.0 for m in METRICS})
        if pid in pg:
            pp = pg[pid]; same = pp[pp.team == team]
            src = same.iloc[0] if len(same) else pp.sort_values("g").iloc[-1]
            row.update(pn=int(src.g), psame=bool(len(same)), ptd=int(src.rush_td + src.rec_td), p_car=float(src.rush), p_ryd=float(src.ryd),
                       p_tg=float(src.tgt), p_cyd=float(src.cyd), **{"p_" + m: float(src[m]) for m in METRICS})
        else:
            row.update(pn=0, psame=False, ptd=0, p_car=0.0, p_ryd=0.0, p_tg=0.0, p_cyd=0.0, **{"p_" + m: 0.0 for m in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def blend(df, K, nt):
    kp = np.where(df.pn > 0, np.where(df.psame, K, K * nt), 0.0); d = df.n + kp
    return {m: np.where(d > 0, (df.n * df["c_" + m] + kp * df["p_" + m]) / np.where(d > 0, d, 1), 0) for m in METRICS}


def raw_lambda(df, P):
    w = blend(df, P["K"], P["newteam"])
    rush = P["w_i10"] * w["s_i10_rush"] + P["w_rz"] * w["s_rz_rush"] + (1 - P["w_i10"] - P["w_rz"]) * w["s_rush"]
    rec = P["w_ez"] * w["s_ez_tgt"] + P["w_rzt"] * w["s_rz_tgt"] + (1 - P["w_ez"] - P["w_rzt"]) * w["s_tgt"]
    return np.maximum(df.teamTD * (df.r * rush + (1 - df.r) * rec), 1e-4)


def design(df, P):
    lr = np.log(raw_lambda(df, P)); fresh = 1.0 / (1.0 + df.n)
    return np.column_stack([np.ones(len(df)), lr, (df.pn == 0).astype(float),
                            np.clip(df.margin, 0, 42) / 10, np.clip(-df.margin, 0, 42) / 10, lr * fresh, fresh,
                            (df.pos == "QB").astype(float), (df.pos == "TE").astype(float), (df.pos == "WR").astype(float)])


def fit_cloglog(X, y, ridge=1.0, iters=40):
    b = np.zeros(X.shape[1]); b[0] = np.log(-np.log(1 - y.mean()))
    R = np.eye(X.shape[1]) * ridge; R[0, 0] = 0
    for _ in range(iters):
        eta = np.clip(X @ b, -8, 3); e = np.exp(eta); mu = np.clip(1 - np.exp(-e), 1e-6, 1 - 1e-6); dmu = e * np.exp(-e)
        w = dmu ** 2 / (mu * (1 - mu)); z = eta + (y - mu) / np.maximum(dmu, 1e-9)
        nb = np.linalg.solve(X.T @ (w[:, None] * X) + R, X.T @ (w * z))
        if np.max(np.abs(nb - b)) < 1e-7: b = nb; break
        b = nb
    return b


def predict_glm(X, b): return 1 - np.exp(-np.exp(np.clip(X @ b, -8, 3)))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def week_frame(T, wk, prior, op, dp, tdev, require_history=True):
    cur, (oc, dc) = shares(T, wk), team_td_split(T, wk)
    r, t = EV[T]
    played = pd.concat([r[r.week == wk][["pid", "posteam"]], t[t.week == wk][["pid", "posteam"]]]).drop_duplicates("pid")
    played = played[played.posteam.isin(FBS)]
    df = assemble(cur, prior, dict(zip(played.pid, played.posteam)))
    if not len(df): return None
    df["pos"] = df.pid.map(POS)
    df = df[df.pos.isin(SKILL)]
    if require_history: df = df[(df.n > 0) | (df.pn > 0)]
    tinfo = tinfo_for(T, wk)
    df = df[df.team.isin(tinfo)].copy()
    df["impl"] = df.team.map(lambda x: tinfo[x][0]); df["teamTD"] = A * df.impl
    df["opp"] = df.team.map(lambda x: tinfo[x][1]); df["margin"] = df.team.map(lambda x: tinfo[x][2]); df["total"] = df.team.map(lambda x: tinfo[x][3])
    for b_ in (0.0, 0.25, 0.5):
        df[f"r_{b_}"] = [team_r(a, o, oc, dc, op, dp, b_)[0] for a, o in zip(df.team, df.opp)]
    df["y"] = df.pid.isin(set(tdev[tdev.week == wk].td_player_id)).astype(int)
    df["ryd_act"] = df.pid.map(r[r.week == wk].groupby("pid").yards_gained.sum()).fillna(0.0)
    df["cyd_act"] = df.pid.map(t[t.week == wk].groupby("pid").cyd.sum()).fillna(0.0)
    df["week"] = wk
    df["naive"] = ((df.td + 0.5 * df.ptd * np.minimum(1, 8 / np.maximum(df.pn, 1))) / (df.n + 0.5 * np.minimum(df.pn, 8)).clip(lower=1)).clip(0.01, 0.95)
    return df


def season_ctx(T):
    prior, (op, dp) = shares(T - 1), team_td_split(T - 1)
    tdev = pbp[T][pbp[T].td_player_id.notna()][["week", "td_player_id"]]
    return prior, op, dp, tdev


def season_frames(T):
    ctx = season_ctx(T)
    out = [f for wk in range(1, int(pbp[T].week.max()) + 1) if (f := week_frame(T, wk, *ctx)) is not None]
    return pd.concat(out, ignore_index=True)


print("Building walk-forward frames…")
tr, te = season_frames(TRAIN), season_frames(TEST)
print(f"  train {TRAIN}: {len(tr)} player-games | test {TEST}: {len(te)} player-games")


def search_structural(d):
    best = None
    for K, nt, wi, wr, we, wt, b in itertools.product([1, 2, 3], [0.35, 0.6], [0.3, 0.45, 0.6], [0.1, 0.2], [0.1, 0.25], [0.1, 0.2], [0.0, 0.25]):
        P = dict(K=K, newteam=nt, w_i10=wi, w_rz=wr, w_ez=we, w_rzt=wt, beta=b)
        x = d.assign(r=d[f"r_{b}"]); X = np.column_stack([np.ones(len(x)), np.log(raw_lambda(x, P))])
        bb = fit_cloglog(X, x.y.values, ridge=0.0, iters=25); s = ll(predict_glm(X, bb), x.y.values)
        if best is None or s < best[0]: best = (s, P)
    return best[1]


print("Fitting on", TRAIN)
P = search_structural(tr); print("  structural", P)
trX, teX = design(tr.assign(r=tr[f"r_{P['beta']}"]), P), design(te.assign(r=te[f"r_{P['beta']}"]), P)
b_full = fit_cloglog(trX, tr.y.values); b_v1 = fit_cloglog(trX[:, :2], tr.y.values, ridge=0.0)
p_full, p_v1 = predict_glm(teX, b_full), predict_glm(teX[:, :2], b_v1)
yt = te.y.values


def calib_bins(p, y):
    d = pd.DataFrame({"p": p, "y": y}); d["bin"] = pd.cut(d.p, [0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, 1])
    c = d.groupby("bin", observed=True).agg(pred=("p", "mean"), act=("y", "mean"), n=("y", "size")).reset_index()
    return [dict(pred=round(r.pred, 4), act=round(r.act, 4), n=int(r.n)) for r in c.itertuples()]


tt = te.assign(p=p_full); top = tt.sort_values("p", ascending=False).groupby("week").head(20)
backtest = dict(train=TRAIN, test=TEST, n=int(len(te)), tds=int(yt.sum()), weeks=f"1-{int(te.week.max())}",
                logloss=ll(p_full, yt), logloss_v1=ll(p_v1, yt), logloss_naive=ll(te.naive.values, yt), logloss_base=ll(np.full(len(te), tr.y.mean()), yt),
                brier=float(np.mean((p_full - yt) ** 2)), brier_naive=float(np.mean((te.naive.values - yt) ** 2)),
                calibration=calib_bins(p_full, yt), top20_pred=float(top.p.mean()), top20_act=float(top.y.mean()),
                by_pos={pos: dict(pred=float(g.p.mean()), act=float(g.y.mean()), n=int(len(g))) for pos, g in tt.groupby("pos") if len(g) > 50})
print(json.dumps({k: v for k, v in backtest.items() if k not in ("calibration", "by_pos")}, indent=1))
b_live = fit_cloglog(np.vstack([trX, teX]), np.concatenate([tr.y.values, yt]))
coef = dict(intercept=round(float(b_live[0]), 5), log_raw=round(float(b_live[1]), 5), snap=0.0, snap_trend=0.0,
            rz_snap_prior=0.0, no_rz_prior=round(float(b_live[2]), 5), fav=round(float(b_live[3]), 5), dog=round(float(b_live[4]), 5),
            lr_fresh=round(float(b_live[5]), 5), fresh=round(float(b_live[6]), 5),
            is_qb=round(float(b_live[7]), 5), is_te=round(float(b_live[8]), 5), is_wr=round(float(b_live[9]), 5))
print("Live coefficients", coef)

# ----------------------------------------------------------------------------- props
def passers(y, max_week=None):
    p = pbp[y] if max_week is None else pbp[y][pbp[y].week < max_week]
    q = p[p.is_pa]
    return q.groupby(["passer", "posteam"]).agg(att=("id_play", "size"), pyd=("cyd", "sum"), ptd=("ptd", "sum"), g=("game_id", "nunique")).reset_index().rename(columns={"passer": "pid", "posteam": "team"})


def pace(y, max_week=None):
    p = pbp[y] if max_week is None else pbp[y][pbp[y].week < max_week]
    o = p.groupby("posteam").agg(ra=("is_rush", "sum"), pa=("is_pa", "sum"), g=("game_id", "nunique"))
    d = p.assign(ry=np.where(p.is_rush, p.yards_gained, 0), py=np.where(p.is_pa, p.cyd, 0)).groupby("defteam").agg(ra=("is_rush", "sum"), ry=("ry", "sum"), pa=("is_pa", "sum"), py=("py", "sum"))
    return o, d


lg = pd.concat([pbp[S - 2], pbp[S - 1]])
LG_YPC = float(lg[lg.is_rush].yards_gained.mean()); LG_YPA = float(lg[lg.is_pa].cyd.mean())
POS_YPC = {k: float(lg[lg.is_rush & (lg.rush_player_id.map(POS) == k)].yards_gained.mean()) for k in ["QB", "RB", "WR", "TE"]}
POS_YPT = {k: float(lg[lg.is_tgt & (lg.recv.map(POS) == k)].cyd.mean()) for k in ["RB", "WR", "TE", "QB"]}
POS_YPC = {k: (v if v == v else LG_YPC) for k, v in POS_YPC.items()}; POS_YPT = {k: (v if v == v else 7.0) for k, v in POS_YPT.items()}
YPC_PRIOR, YPA_PRIOR, YPT_PRIOR = 60.0, 200.0, 40.0
print(f"League YPC {LG_YPC:.2f} YPA {LG_YPA:.2f}")


def blend_pace(T, wk):
    oc, dc = pace(T, wk); op, dp = pace(T - 1); out = {}
    for t in set(op.index) | set(oc.index):
        g = lambda df, c: float(df[c].get(t, 0)) if len(df) else 0.0
        den = 2 * g(oc, "g") + g(op, "g") or 1
        ra, ry = 2 * g(dc, "ra") + g(dp, "ra"), 2 * g(dc, "ry") + g(dp, "ry"); pa_, py = 2 * g(dc, "pa") + g(dp, "pa"), 2 * g(dc, "py") + g(dp, "py")
        out[t] = dict(rush_pg=(2 * g(oc, "ra") + g(op, "ra")) / den, pass_pg=(2 * g(oc, "pa") + g(op, "pa")) / den,
                      def_ypc=(ry + 150 * LG_YPC) / (ra + 150), def_ypa=(py + 250 * LG_YPA) / (pa_ + 250))
    return out


def ypc_blend(cc, cy, pc, py, pos): return (cy + .7 * py + YPC_PRIOR * POS_YPC.get(pos, LG_YPC)) / (cc + .7 * pc + YPC_PRIOR)


def prop_frames(T, posset, extra):
    ctx = season_ctx(T); rows = []
    for wk in range(1, int(pbp[T].week.max()) + 1):
        f = week_frame(T, wk, *ctx)
        if f is None: continue
        pc = blend_pace(T, wk); f = f[f.pos.isin(posset)].copy()
        f["hist_rush"] = f.team.map(lambda t: pc.get(t, {}).get("rush_pg", 38)); f["hist_pass"] = f.team.map(lambda t: pc.get(t, {}).get("pass_pg", 32))
        f["dypc"] = f.opp.map(lambda t: pc.get(t, {}).get("def_ypc", LG_YPC)); f["dypa"] = f.opp.map(lambda t: pc.get(t, {}).get("def_ypa", LG_YPA))
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


def team_frames(T):
    rows = []
    for wk in range(1, int(pbp[T].week.max()) + 1):
        pc = blend_pace(T, wk); ti = tinfo_for(T, wk); cur = pbp[T][pbp[T].week == wk]
        act = cur.groupby("posteam").agg(ra=("is_rush", "sum"), pa=("is_pa", "sum"))
        for t, (imp, opp, mg, tot) in ti.items():
            if t in act.index and t in pc and t in FBS:
                rows.append(dict(team=t, hist_rush=pc[t]["rush_pg"], hist_pass=pc[t]["pass_pg"], margin=mg, total=tot, ra=act.ra[t], pa=act.pa[t]))
    return pd.DataFrame(rows)


def qb_frames(T):
    rows = []
    for wk in range(1, int(pbp[T].week.max()) + 1):
        pc = blend_pace(T, wk); ti = tinfo_for(T, wk); cur, pri = passers(T, wk), passers(T - 1)
        (oc, dc), (op, dp) = team_td_split(T, wk), team_td_split(T - 1)
        g = pbp[T][(pbp[T].week == wk) & pbp[T].is_pa].groupby(["posteam", "passer"]).agg(att=("id_play", "size"), pyd=("cyd", "sum"), ptd=("ptd", "sum")).reset_index()
        starters = g.sort_values("att").groupby("posteam").tail(1)
        for s in starters.itertuples():
            if s.posteam not in ti or s.posteam not in pc or s.posteam not in FBS: continue
            imp, opp, mg, tot = ti[s.posteam]; c = cur[cur.pid == s.passer]; pp = pri[pri.pid == s.passer]
            rows.append(dict(pid=s.passer, team=s.posteam, week=wk, att_c=c.att.sum(), pyd_c=c.pyd.sum(), att_p=pp.att.sum(), pyd_p=pp.pyd.sum(),
                             hist_pass=pc[s.posteam]["pass_pg"], dypa=pc.get(opp, {}).get("def_ypa", LG_YPA), margin=mg, total=tot, impl=imp,
                             r=team_r(s.posteam, opp, oc, dc, op, dp, P["beta"])[0], pyd=s.pyd, ptd=s.ptd))
    return pd.DataFrame(rows)


def ols(X, y): X1 = np.column_stack([np.ones(len(X)), X]); return np.linalg.lstsq(X1, y, rcond=None)[0]


def lad(X, y, iters=60):
    """Median regression (least absolute deviations) by iteratively reweighted least squares."""
    X1 = np.column_stack([np.ones(len(X)), X]); b = np.linalg.lstsq(X1, y, rcond=None)[0]
    for _ in range(iters):
        w = 1 / np.maximum(np.abs(y - X1 @ b), 1.0)
        nb = np.linalg.lstsq(X1 * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        if np.max(np.abs(nb - b)) < 1e-6: b = nb; break
        b = nb
    return b


print("Fitting prop models on", TRAIN)
tf_tr, tf_te = team_frames(TRAIN), team_frames(TEST)
b_ra = ols(tf_tr[["hist_rush", "margin", "total"]].values, tf_tr.ra.values); b_pa = ols(tf_tr[["hist_pass", "margin", "total"]].values, tf_tr.pa.values)
team_ra = lambda h, m, t: b_ra[0] + b_ra[1] * h + b_ra[2] * m + b_ra[3] * t
team_pa = lambda h, m, t: b_pa[0] + b_pa[1] * h + b_pa[2] * m + b_pa[3] * t


def rush_base(f):
    sh = blend(f, P["K"], P["newteam"])["s_rush"]
    ypc = np.array([ypc_blend(a, b_, c, d, pos) for a, b_, c, d, pos in zip(f.c_car, f.c_ryd, f.p_car, f.p_ryd, f.pos)])
    return team_ra(f.hist_rush, f.margin, f.total) * sh * ypc * np.sqrt(f.dypc / LG_YPC)


def rec_base(f):
    sh = blend(f, P["K"], P["newteam"])["s_tgt"]
    ypt = np.array([(cy + .7 * py + YPT_PRIOR * POS_YPT.get(pos, 7.5)) / (ct + .7 * pt + YPT_PRIOR) for ct, cy, pt, py, pos in zip(f.c_tg, f.c_cyd, f.p_tg, f.p_cyd, f.pos)])
    return team_pa(f.hist_pass, f.margin, f.total) * sh * ypt * np.sqrt(f.dypa / LG_YPA)


def pass_base(q):
    ypa = (q.pyd_c + .7 * q.pyd_p + YPA_PRIOR * LG_YPA) / (q.att_c + .7 * q.att_p + YPA_PRIOR)
    return team_pa(q.hist_pass, q.margin, q.total) * ypa * np.sqrt(q.dypa / LG_YPA)


def pg_avg(f, cc, pc):
    return ((f[cc] + 0.5 * f[pc] * np.minimum(1, 8 / np.maximum(f.pn, 1))) / (f.n + 0.5 * np.minimum(f.pn, 8)).clip(lower=1)).values


def fit_sd(pred, y): return ols(pred[:, None], np.abs(y - pred) * np.sqrt(np.pi / 2))
from math import erf
pover = lambda line, mu, sd: 0.5 * (1 - np.vectorize(erf)((line - mu) / (np.maximum(sd, 1) * np.sqrt(2))))


def cal_over(pred, sd, y, med):
    return [dict(off=o, pred=float(pover(pred + med + o, pred + med, sd).mean()), act=float((y > pred + med + o).mean())) for o in (-20, -10, 0, 10, 20)]


rf_tr, rf_te = prop_frames(TRAIN, {"RB", "QB", "WR"}, None), prop_frames(TEST, {"RB", "QB", "WR"}, None)
cf_tr, cf_te = prop_frames(TRAIN, {"WR", "TE", "RB"}, None), prop_frames(TEST, {"WR", "TE", "RB"}, None)
qf_tr, qf_te = qb_frames(TRAIN), qb_frames(TEST)
props_bt = {}
fits = {}
AVG = {"ryd": ("c_ryd", "p_ryd"), "cyd": ("c_cyd", "p_cyd")}


def Xmat(key, f, base):
    bs = base(f).values
    return np.column_stack([bs, pg_avg(f, *AVG[key])]) if key in AVG else bs[:, None]


for key, ftr, fte, base, col in (("ryd", rf_tr, rf_te, rush_base, "ryd_act"), ("cyd", cf_tr, cf_te, rec_base, "cyd_act"), ("pyd", qf_tr, qf_te, pass_base, "pyd")):
    Xtr, Xte = Xmat(key, ftr, base), Xmat(key, fte, base)
    bb = lad(Xtr, ftr[col].values); ptr = np.clip(bb[0] + Xtr @ bb[1:], 0, None); pte = np.clip(bb[0] + Xte @ bb[1:], 0, None)
    sd = fit_sd(ptr, ftr[col].values); med = float(np.median(ftr[col].values - ptr)); sdte = sd[0] + sd[1] * pte
    if key == "pyd":
        naive = ((fte.pyd_c + 4 * np.where(fte.att_p > 0, fte.pyd_p / np.maximum(1, np.round(fte.att_p / 30)), LG_YPA * 30)) /
                 (np.where(fte.att_c > 0, np.maximum(1, np.round(fte.att_c / 30)), 0) + 4)).values
    else:
        naive = pg_avg(fte, *AVG[key])
    yv = fte[col].values
    props_bt[key] = dict(n=int(len(fte)), mae=float(np.mean(np.abs(pte + med - yv))), mae_naive=float(np.mean(np.abs(naive - yv))),
                         mae_flat=float(np.mean(np.abs(ftr[col].mean() - yv))), corr=float(np.corrcoef(pte, yv)[0, 1]),
                         cal=cal_over(pte[m_], sdte[m_], yv[m_], med) if (m_ := (pte + med >= (15 if key != "pyd" else 0))).sum() > 100 else cal_over(pte, sdte, yv, med))
def fit_ptd(q):
    """Poisson fit: TD passes = k * team pass-TD expectation * exp(c * points favored beyond 14 / 10)."""
    base = A * q.impl * (1 - q.r); fav = np.clip(q.margin - 14, 0, 40) / 10; best = None
    for c in np.arange(-0.6, 0.21, 0.02):
        lam0 = base * np.exp(c * fav); k = q.ptd.sum() / lam0.sum(); lam = k * lam0
        dev = float(np.sum(lam - q.ptd * np.log(np.maximum(lam, 1e-9))))
        if best is None or dev < best[0]: best = (dev, k, c)
    return best[1], best[2]


def lam_ptd(q, k, c): return k * A * q.impl * (1 - q.r) * np.exp(c * np.clip(q.margin - 14, 0, 40) / 10)


k_ptd, c_ptd = fit_ptd(qf_tr); lam_te = lam_ptd(qf_te, k_ptd, c_ptd)
big = qf_te.margin >= 21
print(f"  pass TD blowout term {c_ptd:+.2f} per 10 pts beyond 14 | big favorites: pred {lam_te[big].mean():.2f} act {qf_te.ptd[big].mean():.2f} (n={big.sum()})")
pk = lambda lam, k: 1 - sum(np.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
props_bt["ptd"] = dict(n=int(len(qf_te)), ladder=[dict(line=l, pred=float(pk(lam_te, int(l + .5)).mean()), act=float((qf_te.ptd >= l + .5).mean())) for l in (0.5, 1.5, 2.5)],
                       mean_pred=float(lam_te.mean()), mean_act=float(qf_te.ptd.mean()))
# rushing TDs split from the anytime model
w_ = blend(te.assign(r=te[f"r_{P['beta']}"]), P["K"], P["newteam"]); r_ = te[f"r_{P['beta']}"]
ru = r_ * (P["w_i10"] * w_["s_i10_rush"] + P["w_rz"] * w_["s_rz_rush"] + (1 - P["w_i10"] - P["w_rz"]) * w_["s_rush"])
rc = (1 - r_) * (P["w_ez"] * w_["s_ez_tgt"] + P["w_rzt"] * w_["s_rz_tgt"] + (1 - P["w_ez"] - P["w_rzt"]) * w_["s_tgt"])
p_rtd = 1 - np.exp(-(-np.log(1 - np.clip(p_full, 1e-6, 1 - 1e-6))) * np.where(ru + rc > 0, ru / np.maximum(ru + rc, 1e-9), 0))
rset = set(zip(pbp[TEST][pbp[TEST].rtd == 1].week, pbp[TEST][pbp[TEST].rtd == 1].rush_player_id))
y_rtd = np.array([(w, pid) in rset for w, pid in zip(te.week, te.pid)]).astype(int)
props_bt["rtd"] = dict(n=int(len(te)), tds=int(y_rtd.sum()), logloss=ll(p_rtd, y_rtd), logloss_base=ll(np.full(len(te), y_rtd.mean()), y_rtd), calibration=calib_bins(p_rtd, y_rtd))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("cal", "ladder", "calibration")} for k, v in props_bt.items()}, indent=1))

# refit on both seasons for live use
tf_all = pd.concat([tf_tr, tf_te]); b_ra = ols(tf_all[["hist_rush", "margin", "total"]].values, tf_all.ra.values); b_pa = ols(tf_all[["hist_pass", "margin", "total"]].values, tf_all.pa.values)
live_fit = {}
for key, mk, fa, base, col in (("ry", "ryd", pd.concat([rf_tr, rf_te]), rush_base, "ryd_act"), ("cy", "cyd", pd.concat([cf_tr, cf_te]), rec_base, "cyd_act"), ("py", "pyd", pd.concat([qf_tr, qf_te]), pass_base, "pyd")):
    X = Xmat(mk, fa, base); bb = lad(X, fa[col].values); pr = np.clip(bb[0] + X @ bb[1:], 0, None)
    live_fit[key] = (bb, fit_sd(pr, fa[col].values), float(np.median(fa[col].values - pr)))
qa = pd.concat([qf_tr, qf_te]); k_ptd, c_ptd = fit_ptd(qa)
PROPS = dict(b_ra=[round(float(x), 4) for x in b_ra], b_pa=[round(float(x), 4) for x in b_pa],
             b_ry=[round(float(x), 4) for x in live_fit["ry"][0]], sd_ry=[round(float(x), 4) for x in live_fit["ry"][1]], med_ry=round(live_fit["ry"][2], 3),
             b_cy=[round(float(x), 4) for x in live_fit["cy"][0]], sd_cy=[round(float(x), 4) for x in live_fit["cy"][1]], med_cy=round(live_fit["cy"][2], 3),
             b_py=[round(float(x), 4) for x in live_fit["py"][0]], sd_py=[round(float(x), 4) for x in live_fit["py"][1]], med_py=round(live_fit["py"][2], 3),
             k_ptd=round(k_ptd, 4), c_ptd=round(float(c_ptd), 4), LG_YPC=round(LG_YPC, 3), LG_YPA=round(LG_YPA, 3), POS_YPC={k: round(v, 3) for k, v in POS_YPC.items()},
             POS_YPT={k: round(v, 3) for k, v in POS_YPT.items()}, YPC_PRIOR=YPC_PRIOR, YPA_PRIOR=YPA_PRIOR, YPT_PRIOR=YPT_PRIOR, backtest=props_bt)

# ----------------------------------------------------------------------------- the slate: live lines from ESPN, else a demo of the latest completed week
done_weeks = sorted(int(w) for w in pbp[S].week.unique()) if len(pbp[S]) else []
W = args.week or ((max(done_weeks) + 1) if done_weeks else 1)


def espn_slate(week):
    try:
        req = urllib.request.Request(ESPN.format(week=week), headers={"User-Agent": "Mozilla/5.0"})
        js = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        print("  ESPN feed unavailable:", e); return None
    games = []
    for ev in js.get("events", []):
        c = ev["competitions"][0]; comp = {x["homeAway"]: x for x in c["competitors"]}
        if "home" not in comp or "away" not in comp: continue
        h, a = comp["home"]["team"], comp["away"]["team"]
        for tm in (h, a):
            tid = str(tm["id"])
            TEAMS.setdefault(tid, {}); TEAMS[tid].update(abbr=tm.get("abbreviation"), short=tm.get("shortDisplayName") or tm.get("location"),
                                                        c1="#" + (tm.get("color") or "555555"), c2="#" + (tm.get("alternateColor") or "999999"))
        o = (c.get("odds") or [{}])[0]
        spread, total = o.get("spread"), o.get("overUnder")

        def num(x):
            try: return float(str(x).replace("o", "").replace("u", ""))
            except Exception: return None
        ps, tt = o.get("pointSpread", {}), o.get("total", {})
        done = bool(((ev.get("status") or {}).get("type") or {}).get("completed"))
        games.append(dict(id=str(ev["id"]), date=ev.get("date"), home=str(h["id"]), away=str(a["id"]), completed=done,
                          spread=None if spread is None else -float(spread), total=None if total is None else float(total),
                          home_spread_odds=num(ps.get("home", {}).get("close", {}).get("odds")), away_spread_odds=num(ps.get("away", {}).get("close", {}).get("odds")),
                          over_odds=num(tt.get("over", {}).get("close", {}).get("odds")), under_odds=num(tt.get("under", {}).get("close", {}).get("odds")),
                          neutral=bool(c.get("neutralSite"))))
    return games


demo = False
if args.week:
    slate_raw = espn_slate(W)
else:
    # The latest week with data may still have games left (Thursday/Friday games finish first)
    last = max(done_weeks) if done_weeks else 1
    cur_games = espn_slate(last)
    if cur_games and any(not g["completed"] for g in cur_games):
        W, slate_raw = last, cur_games
    else:
        W, slate_raw = last + 1, espn_slate(last + 1)
if slate_raw:
    slate_raw = [g for g in slate_raw if not g.get("completed")]   # price only games not yet played
if not slate_raw:
    demo = True; W = max(done_weeks) if done_weeks else 1
    print(f"Demo mode: pricing completed Week {W} with its closing lines")
    slate_raw = []
    s = sched[S][sched[S].week == W]
    for g in s.itertuples():
        L = LINES[S].get(str(g.game_id))
        if not L: continue
        slate_raw.append(dict(id=str(g.game_id), date=g.game_date, home=L[0], away=L[1], spread=L[2], total=L[3],
                              home_spread_odds=None, away_spread_odds=None, over_odds=None, under_odds=None, neutral=bool(g.neutral_site)))
print(f"Pricing Week {W}: {len(slate_raw)} games")


def no_vig(a, b):
    if a is None or b is None: return None
    pa = 100 / (a + 100) if a > 0 else -a / (-a + 100); pb = 100 / (b + 100) if b > 0 else -b / (-b + 100); return pa / (pa + pb)


SD, PDF0 = 16.0, 1 / (16.0 * math.sqrt(2 * math.pi))       # college margins and totals spread wider than the NFL's
slate, tinfo = [], {}
for g in slate_raw:
    if g["spread"] is None or g["total"] is None: continue
    if g["home"] not in FBS or g["away"] not in FBS: continue   # FBS vs FBS only
    po, ph = no_vig(g["over_odds"], g["under_odds"]), no_vig(g["home_spread_odds"], g["away_spread_odds"])
    d = dt.datetime.fromisoformat(str(g["date"]).replace("Z", "+00:00")).astimezone(dt.timezone(dt.timedelta(hours=-4)))
    slate.append(dict(id=g["id"], home=g["home"], away=g["away"], spread=g["spread"], total=g["total"],
                      spread_adj=round(g["spread"] + ((ph - .5) / PDF0 if ph else 0), 2), total_adj=round(g["total"] + ((po - .5) / PDF0 if po else 0), 2),
                      over_odds=g["over_odds"], under_odds=g["under_odds"], weekday=d.strftime("%A"), time=d.strftime("%H:%M"),
                      neutral=g["neutral"], roof=None, wind=None, noline=False))
    tinfo[g["home"]] = g["away"]; tinfo[g["away"]] = g["home"]

# ----------------------------------------------------------------------------- players on this week's FBS teams
cur, prior = shares(S, W), shares(S - 1)
oc, dc = team_td_split(S, W); op, dp = team_td_split(S - 1)
r26, t26 = EV[S]; r26, t26 = r26[r26.week < W], t26[t26.week < W]
team_of = {}
for df_ in (cur, prior[prior.team.isin(tinfo)]):
    for pid, team in zip(df_.pid, df_.team):
        if team in tinfo and team in FBS: team_of.setdefault(pid, team)
live = assemble(cur, prior, team_of)
live["pos"] = live.pid.map(POS)
live = live[live.pos.isin(SKILL) & ((live.n > 0) | ((live.pn >= 3) & live.psame))]
# who is active: touched the ball in either of the team's last two games
recent = {}
gw = pbp[S][pbp[S].week < W].groupby("posteam").week.apply(lambda s: sorted(s.unique())[-2:])
touch = pd.concat([r26[["pid", "posteam", "week"]], t26[["pid", "posteam", "week"]]]).drop_duplicates()
for pid, team in zip(live.pid, live.team):
    wks = gw.get(team, [])
    recent[pid] = bool(len(touch[(touch.pid == pid) & (touch.posteam == team) & touch.week.isin(wks)]))
# listed starter proxy: the QB who threw the most passes in the team's latest game
starter = {}
lastpa = pbp[S][(pbp[S].week < W) & pbp[S].is_pa]
if len(lastpa):
    lw = lastpa.groupby("posteam").week.max()
    for team, wk in lw.items():
        x = lastpa[(lastpa.posteam == team) & (lastpa.week == wk)].passer.value_counts()
        if len(x): starter[team] = x.index[0]
for g in slate:
    g["home_qb"], g["away_qb"] = starter.get(g["home"]), starter.get(g["away"])
    g["home_qb_name"], g["away_qb_name"] = NAME.get(g["home_qb"]), NAME.get(g["away_qb"])

pc_live = blend_pace(S, W)
qp_c, qp_p = passers(S, W), passers(S - 1)
teams = {}
uo = {}
for y, wgt in ((S - 1, 1.0), (S, 2.0)):
    p = pbp[y] if y < S else pbp[y][pbp[y].week < W]
    if not len(p): continue
    for key in ("posteam", "defteam"):
        g = p.groupby(key).agg(r=("rtd", "sum"), pa=("ptd", "sum"), gp=("game_id", "nunique")) * wgt
        uo[key] = g if key not in uo else uo[key].add(g, fill_value=0)
UO = pd.DataFrame({"rush": uo["posteam"].r / uo["posteam"].gp, "pass": uo["posteam"].pa / uo["posteam"].gp}).loc[lambda d: d.index.isin(FBS)]
UD = pd.DataFrame({"rush": uo["defteam"].r / uo["defteam"].gp, "pass": uo["defteam"].pa / uo["defteam"].gp}).loc[lambda d: d.index.isin(FBS)]
for t, opp in tinfo.items():
    _, off, _ = team_r(t, opp, oc, dc, op, dp, 0); _, _, de = team_r(opp, t, oc, dc, op, dp, 0)
    units = None
    if t in UO.index and t in UD.index:
        units = dict(rushO=round(float((UO["rush"] < UO.loc[t, "rush"]).mean()), 3), passO=round(float((UO["pass"] < UO.loc[t, "pass"]).mean()), 3),
                     rushD=round(float((UD["rush"] > UD.loc[t, "rush"]).mean()), 3), passD=round(float((UD["pass"] > UD.loc[t, "pass"]).mean()), 3))
    pcv = pc_live.get(t, {})
    teams[t] = dict(opp=opp, offR=round(off, 4), defR=round(de, 4), units=units, conf=CONF.get(t), fbs=t in FBS,
                    rush_pg=round(pcv.get("rush_pg", 38), 2), pass_pg=round(pcv.get("pass_pg", 32), 2),
                    def_ypc=round(pcv.get("def_ypc", LG_YPC), 3), def_ypa=round(pcv.get("def_ypa", LG_YPA), 3))

wl = (r26.groupby(["pid", "week"]).agg(car=("rz", "size"), rzc=("rz", "sum"), rtd=("rtd", "sum"))
      .join(t26.groupby(["pid", "week"]).agg(tgt=("rz", "size"), rzt=("rz", "sum"), ctd=("ptd", "sum")), how="outer").fillna(0).reset_index())
logs = {}
for r in wl.itertuples():
    logs.setdefault(r.pid, []).append([int(r.week), int(r.car), int(r.tgt), int(r.rzc + r.rzt), int(r.rtd + r.ctd), None])

players = []
for r in live.itertuples():
    qc, qp = qp_c[qp_c.pid == r.pid], qp_p[qp_p.pid == r.pid]
    players.append(dict(id=r.pid, name=NAME.get(r.pid, r.pid), pos=r.pos, team=r.team, num=JERSEY.get(r.pid, 0), rookie=False,
                        n=int(r.n), td=int(r.td), pn=int(r.pn), psame=bool(r.psame), ptd=int(r.ptd),
                        c=[round(getattr(r, "c_" + m), 4) for m in METRICS], p=[round(getattr(r, "p_" + m), 4) for m in METRICS],
                        sn=0, smean=0, slast=0, spn=0, sprior=0, rzp=None if r.pn == 0 else 0, depth=None, inj=None, recent=recent.get(r.pid, False),
                        ru=[int(r.c_car), int(r.c_ryd), int(r.p_car), int(r.p_ryd)], rc=[int(r.c_tg), int(r.c_cyd), int(r.p_tg), int(r.p_cyd)],
                        qb=[int(qc.att.sum()), int(qc.pyd.sum()), int(qp.att.sum()), int(qp.pyd.sum())] if r.pos == "QB" else None,
                        log=logs.get(r.pid, [])))

# ----------------------------------------------------------------------------- this season's receipts
history = []
if len(pbp[S]):
    ctx = season_ctx(S)
    for wk in [w for w in done_weeks if w < W or (demo and w <= W)]:
        f = week_frame(S, wk, *ctx, require_history=False)
        if f is None or not len(f): continue
        f = f.assign(r=f[f"r_{P['beta']}"]); f["p"] = predict_glm(design(f, P), b_live)
        f = f[f.p >= 0.04].sort_values("p", ascending=False); top = f.head(20)
        ab = lambda t: TEAMS.get(t, {}).get("abbr", t)
        history.append(dict(week=wk, n=int(len(f)), tds=int(f.y.sum()), exp=round(float(f.p.sum()), 1), top20_exp=round(float(top.p.sum()), 1), top20_hit=int(top.y.sum()),
                            rows=[[NAME.get(x.pid, x.pid), x.pos, x.team, ab(x.opp), int(round(x.p * 1000)), int(x.y)] for x in f.itertuples()]))
    print("Receipts:", [(h["week"], f"{h['top20_hit']}/20 vs {h['top20_exp']}") for h in history])

colors = {t: [v.get("c1", "#555"), v.get("c2", "#999"), v.get("short", t)] for t, v in TEAMS.items() if t in FBS or t in tinfo}
abbr = {t: v.get("abbr", t) for t, v in TEAMS.items()}
data = dict(season=S, week=W, demo=demo, built=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            through=max([w for w in done_weeks if w < W] or [0]), inj_week=None, A=round(A, 5), LG_R=round(LG_R, 4), params=P, coef=coef,
            backtest=backtest, history=history, props=PROPS, games=slate, teams=teams, colors=colors, abbr=abbr, players=players)
tpl = open(os.path.join(ROOT, "template.html"), encoding="utf-8").read()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w", encoding="utf-8").write(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), default=lambda o: None).replace("</", "<\\/")))
print(f"Wrote {args.out} ({len(players)} players, {len(slate)} games{', demo' if demo else ''})")

if os.environ.get("CFB_DIAG"):
    tt2 = te.assign(p=p_full)
    for lo, hi in ((1, 3), (4, 8), (9, 16)):
        x = tt2[(tt2.week >= lo) & (tt2.week <= hi)]; tp = x.sort_values("p", ascending=False).groupby("week").head(20)
        print(f"weeks {lo}-{hi}: top20 pred {tp.p.mean():.3f} act {tp.y.mean():.3f} | all pred {x.p.mean():.3f} act {x.y.mean():.3f}")

if os.environ.get("CFB_DIAG2"):
    tt3 = te.assign(p=p_full)
    for pos in ["QB", "RB", "WR", "TE"]:
        x = tt3[tt3.pos == pos]; hi = x[x.p >= 0.5]
        print(f"{pos}: all pred {x.p.mean():.3f} act {x.y.mean():.3f} | p>=50%: n={len(hi)} pred {hi.p.mean():.3f} act {hi.y.mean():.3f}")
