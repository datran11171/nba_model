"""Predict a player's stat line against a given opponent, home or away.

Examples
  python predict.py player "Stephen Curry" --opp Lakers --home
  python predict.py player "Jokic" --opp Celtics --away --date 2026-12-25
  python predict.py player "Anthony Edwards" --next          # next scheduled game
  python predict.py game Lakers Warriors                      # home team first; both rosters

Predictions assume the player plays; they're the model's expected value plus an
80% range calibrated on 2025-26/2026-27 games the model never trained on.
"""
import argparse
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd

from nba_model.data import ARTIFACT_DIR, DATA_DIR, prepare
from nba_model.features import TARGETS, assemble

LABELS = {"pts": "Points", "reb": "Rebounds", "ast": "Assists", "fg3m": "3PM", "stl": "Steals",
          "blk": "Blocks", "tov": "Turnovers", "min": "Minutes"}


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def load_teams() -> pd.DataFrame:
    sched = pd.read_csv(DATA_DIR / "LeagueSchedule26_27.csv")
    t = pd.concat([
        sched[["homeTeamId", "homeTeamCity", "homeTeamName"]].set_axis(["teamId", "city", "name"], axis=1),
        sched[["awayTeamId", "awayTeamCity", "awayTeamName"]].set_axis(["teamId", "city", "name"], axis=1),
    ]).drop_duplicates("teamId")
    t["full"] = t.city + " " + t.name
    return t.reset_index(drop=True)


def find_team(q: str, teams: pd.DataFrame) -> pd.Series:
    nq = _norm(q)
    for col in ("name", "full", "city"):
        hit = teams[teams[col].map(_norm) == nq]
        if len(hit) == 1:
            return hit.iloc[0]
    hit = teams[teams.full.map(_norm).str.contains(nq, regex=False)]
    if len(hit) == 1:
        return hit.iloc[0]
    sys.exit(f"Team '{q}' is {'ambiguous: ' + ', '.join(hit.full) if len(hit) else 'not found'}.")


def find_player(q: str, info: pd.DataFrame, pg: pd.DataFrame) -> int:
    last_game = pg.groupby("personId").date.max()
    cand = info[info.personId.isin(last_game.index)].copy()
    cand["full"] = (cand.firstName.fillna("") + " " + cand.lastName.fillna("")).map(_norm)
    cand["last"] = last_game.reindex(cand.personId).values
    nq = _norm(q)
    hit = cand[cand.full == nq]
    if hit.empty:
        hit = cand[cand.full.str.contains(nq, regex=False)]
    if hit.empty:
        sys.exit(f"No player matching '{q}'.")
    if len(hit) > 1:
        recent = hit[hit["last"] >= pg.date.max() - pd.Timedelta(days=400)]
        if len(recent) == 1:
            return int(recent.personId.iloc[0])
        names = ", ".join(f"{r.firstName} {r.lastName}" for r in (recent if len(recent) else hit).itertuples())
        sys.exit(f"'{q}' matches several players: {names}. Be more specific.")
    return int(hit.personId.iloc[0])


def load_schedule() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(DATA_DIR.glob("LeagueSchedule*.csv"))]
    s = pd.concat(frames)[["gameDateTimeEst", "homeTeamId", "awayTeamId"]]
    # Some old preseason rows carry "+00:00" offsets; only the calendar date matters here.
    s["date"] = pd.to_datetime(s.gameDateTimeEst.astype(str).str[:10])
    return s.drop_duplicates().sort_values("date")


def _season_of(date: pd.Timestamp) -> int:
    return date.year if date.month >= 8 else date.year - 1


def build_rows(games, pg, tg, info, sched):
    """games: list of dicts(personId, teamId, oppId, home, date, gid). Returns feature rows."""
    synth_p, synth_t = [], {}
    for g in games:
        synth_p.append({"personId": g["personId"], "gameId": g["gid"], "date": g["date"],
                        "season": _season_of(g["date"]), "playoff": 0, "teamId": g["teamId"],
                        "oppId": g["oppId"], "home": g["home"], "starter": np.nan})
        for team in (g["teamId"], g["oppId"]):
            synth_t[(g["gid"], team)] = {"gameId": g["gid"], "teamId": team, "date": g["date"],
                                         "season": _season_of(g["date"])}
    ids = {g["personId"] for g in games}
    p = pd.concat([pg[pg.personId.isin(ids)], pd.DataFrame(synth_p)], ignore_index=True)
    t = pd.concat([tg, pd.DataFrame(list(synth_t.values()))], ignore_index=True)
    X = assemble(p, t, info)
    X = X[X.gameId < 0].copy()
    # Rest = days since the team's previous game (played or scheduled), assuming the
    # player suits up for it. Otherwise a game weeks away looks like a return from injury.
    # (assemble sorts rows by player, so match on personId rather than position)
    for g in games:
        i = X.index[X.personId == g["personId"]]
        past = pd.concat([tg.loc[tg.teamId == g["teamId"], "date"],
                          sched.loc[(sched.homeTeamId == g["teamId"]) | (sched.awayTeamId == g["teamId"]), "date"]])
        past = past[past < g["date"]]
        if len(past):
            gap = (g["date"] - past.max()).days
            X.loc[i, "rest_days"] = min(gap, 10)
            X.loc[i, "b2b"] = float(gap == 1)
    return X.set_index(["personId", "gameId"])


def next_meeting(team_id, opp_id, upcoming, home=None):
    """First scheduled game between the teams (optionally with team_id at home/away), or None."""
    m = upcoming[((upcoming.homeTeamId == team_id) & (upcoming.awayTeamId == opp_id)) |
                 ((upcoming.awayTeamId == team_id) & (upcoming.homeTeamId == opp_id))]
    if home is not None:
        m = m[(m.homeTeamId == team_id) == bool(home)]
    return m.iloc[0] if len(m) else None


def rotation(team_id, pg, tg, last_n=5):
    """Players whose latest game was for this team, within the team's last `last_n` games."""
    team_games = tg[tg.teamId == team_id].sort_values("date").tail(last_n).gameId
    latest = pg.sort_values("date").groupby("personId").tail(1)
    return [int(p) for p in latest[(latest.teamId == team_id) & latest.gameId.isin(team_games)].personId]


def predict(X: pd.DataFrame, bundle) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    feats = X[bundle["features"]].astype("float32")
    for t in TARGETS:
        pred = np.clip(bundle["models"][t].predict(feats), 0, None)
        table = bundle["intervals"][t]
        edges = np.array([e for e, _, _ in table])
        idx = np.clip(np.searchsorted(edges, pred, side="right") - 1, 0, len(table) - 1)
        out[t] = pred
        out[f"{t}_lo"] = np.clip(pred + np.array([table[i][1] for i in idx]), 0, None)
        out[f"{t}_hi"] = pred + np.array([table[i][2] for i in idx])
    return out


def print_player(name, team, opp, home, date, x, p):
    venue = "vs" if home else "@"
    print(f"\n{name} ({team.full}) {venue} {opp.full} — {date:%a %b %d, %Y} ({'home' if home else 'away'})")
    if x.rest_days == x.rest_days:
        print(f"Rest: {int(x.rest_days)} day(s){'  [back-to-back]' if x.b2b == 1 else ''}"
              f"   Prior games vs {opp['name']}: {int(x.opp_n)} (model uses up to the last 12)")
    print(f"\n  {'Stat':<10}{'Predicted':>10}{'80% range':>14}{'Last 10':>9}{'Season':>8}{'vs Opp':>8}")
    for t in TARGETS:
        vs = x.get(f"{t}_vs_opp", np.nan)
        szn = x[f"{t}_szn"]
        rng = f"{p[t + '_lo']:.0f}–{p[t + '_hi']:.0f}"
        print(f"  {LABELS[t]:<10}{p[t]:>10.1f}{rng:>14}"
              f"{x[f'{t}_l10']:>9.1f}{szn if szn == szn else np.nan:>8.1f}"
              f"{'' if vs != vs else f'{vs:.1f}':>8}")
    pra = p.pts + p.reb + p.ast
    print(f"\n  PTS+REB+AST: {pra:.1f}")
    print(f"  Venue effect (vs player's usual, pts): {x.pts_venue_resid:+.2f}   "
          f"Matchup effect vs {opp['name']} (pts): {x.pts_opp_resid:+.2f}   "
          f"Opp def rating vs league: {x.opp_drtg_rel:+.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("player")
    a.add_argument("name")
    a.add_argument("--opp")
    v = a.add_mutually_exclusive_group()
    v.add_argument("--home", action="store_true")
    v.add_argument("--away", action="store_true")
    a.add_argument("--date", help="YYYY-MM-DD (default: next scheduled meeting)")
    a.add_argument("--next", action="store_true", help="use the player's next scheduled game")
    b = sub.add_parser("game")
    b.add_argument("home_team")
    b.add_argument("away_team")
    b.add_argument("--date")
    args = ap.parse_args()

    bundle_path = ARTIFACT_DIR / "model.joblib"
    if not bundle_path.exists():
        sys.exit("No trained model found. Run: python train.py")
    bundle = joblib.load(bundle_path)
    pg, tg, info = prepare()
    teams, sched = load_teams(), load_schedule()
    last_data = pg.date.max()
    upcoming = sched[sched.date > last_data]

    if args.mode == "player":
        pid = find_player(args.name, info, pg)
        hist = pg[pg.personId == pid]
        team = teams[teams.teamId == hist.teamId.iloc[-1]].iloc[0]
        if args.next:
            nxt = upcoming[(upcoming.homeTeamId == team.teamId) | (upcoming.awayTeamId == team.teamId)]
            if nxt.empty:
                sys.exit("No upcoming games in the schedule files.")
            g = nxt.iloc[0]
            home = int(g.homeTeamId == team.teamId)
            opp = teams[teams.teamId == (g.awayTeamId if home else g.homeTeamId)].iloc[0]
            date = g.date
        else:
            if not args.opp:
                sys.exit("Give --opp TEAM or --next.")
            opp = find_team(args.opp, teams)
            if opp.teamId == team.teamId:
                sys.exit(f"{args.name} plays for the {opp.full}.")
            meet = next_meeting(team.teamId, opp.teamId, upcoming,
                                home=int(args.home) if (args.home or args.away) else None)
            if args.home or args.away:
                home = int(args.home)
            else:
                home = int(meet.homeTeamId == team.teamId) if meet is not None else 1
            date = pd.Timestamp(args.date) if args.date else (meet.date if meet is not None else last_data + pd.Timedelta(days=2))
        if date <= hist.date.max():
            sys.exit(f"Date must be after the player's last game in the data ({hist.date.max():%Y-%m-%d}).")
        X = build_rows([{"personId": pid, "teamId": team.teamId, "oppId": opp.teamId,
                         "home": home, "date": date, "gid": -1}], pg, tg, info, sched)
        P = predict(X, bundle)
        name = info.loc[info.personId == pid, ["firstName", "lastName"]].iloc[0]
        print_player(f"{name.firstName} {name.lastName}", team, opp, home, date, X.iloc[0], P.iloc[0])
        if hist.date.max() < last_data - pd.Timedelta(days=30):
            print(f"\n  Note: last game in data was {hist.date.max():%Y-%m-%d}; prediction may be stale.")
        return

    # game mode: rotation players of both teams
    ht, at = find_team(args.home_team, teams), find_team(args.away_team, teams)
    if args.date:
        date = pd.Timestamp(args.date)
    else:
        m = upcoming[(upcoming.homeTeamId == ht.teamId) & (upcoming.awayTeamId == at.teamId)]
        date = m.iloc[0].date if len(m) else last_data + pd.Timedelta(days=2)
    games = []
    for team, opp, home in ((ht, at, 1), (at, ht, 0)):
        games += [{"personId": p, "teamId": team.teamId, "oppId": opp.teamId, "home": home,
                   "date": date, "gid": -1} for p in rotation(team.teamId, pg, tg)]
    X = build_rows(games, pg, tg, info, sched)
    P = predict(X, bundle).reset_index()
    P = P.merge(info[["personId", "firstName", "lastName"]], on="personId")
    P = P.merge(pd.DataFrame(games)[["personId", "teamId"]], on="personId")
    print(f"\n{at.full} @ {ht.full} — {date:%a %b %d, %Y}")
    for team in (ht, at):
        print(f"\n{team.full} ({'home' if team.teamId == ht.teamId else 'away'})")
        print(f"  {'Player':<26}{'MIN':>6}{'PTS':>6}{'REB':>6}{'AST':>6}{'3PM':>6}{'STL':>6}{'BLK':>6}{'TOV':>6}")
        for r in P[P.teamId == team.teamId].sort_values("min", ascending=False).itertuples():
            print(f"  {r.firstName + ' ' + r.lastName:<26}{r.min:>6.1f}{r.pts:>6.1f}{r.reb:>6.1f}{r.ast:>6.1f}"
                  f"{r.fg3m:>6.1f}{r.stl:>6.1f}{r.blk:>6.1f}{r.tov:>6.1f}")


if __name__ == "__main__":
    main()
