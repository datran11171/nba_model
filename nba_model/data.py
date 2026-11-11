"""Load the raw CSVs in nba_data/ and turn them into two clean tables.

player_games: one row per player per game played (DNPs dropped).
team_games:   one row per team per game, with the team's own offense and
              what it allowed on defense (overall and by opponent position).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "nba_data"
ARTIFACT_DIR = ROOT / "artifacts"

FIRST_SEASON = 2004  # 2004-05 onward: modern-ish era, advanced stats available
# 3rd digit of the 10-char gameId: 2=regular season, 4=playoffs, 5=play-in, 6=NBA Cup final
COMPETITIVE_TYPES = {"2", "4", "5", "6"}

PLAYER_STAT_COLS = {
    "numMinutes": "min",
    "points": "pts",
    "reboundsTotal": "reb",
    "assists": "ast",
    "threePointersMade": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
    "fieldGoalsAttempted": "fga",
    "freeThrowsAttempted": "fta",
    "threePointersAttempted": "fg3a",
    "plusMinusPoints": "pm",
}


def _parse_minutes(s: pd.Series) -> pd.Series:
    """Minutes come as floats (26.28) or clock strings ("17:42")."""
    s = s.astype("string")
    clock = s.str.contains(":", na=False)
    out = pd.to_numeric(s.where(~clock), errors="coerce")
    parts = s[clock].str.split(":", expand=True)
    if len(parts):
        out[clock] = pd.to_numeric(parts[0], errors="coerce") + pd.to_numeric(parts[1], errors="coerce") / 60
    return out.astype(float)


def _game_meta(game_id: pd.Series) -> pd.DataFrame:
    s = game_id.astype("int64").astype(str).str.zfill(10)
    yy = s.str[3:5].astype(int)
    return pd.DataFrame(
        {"gtype": s.str[2], "season": np.where(yy < 50, 2000 + yy, 1900 + yy)},
        index=game_id.index,
    )


def load_players_info() -> pd.DataFrame:
    p = pd.read_csv(DATA_DIR / "Players.csv")
    pos = np.select([p.center == 1, p.forward == 1, p.guard == 1], ["C", "F", "G"], default="")
    return pd.DataFrame(
        {
            "personId": p.personId.astype("int64"),
            "firstName": p.firstName,
            "lastName": p.lastName,
            "birthDate": pd.to_datetime(p.birthDate, errors="coerce"),
            "height": pd.to_numeric(p.heightInches, errors="coerce"),
            "weight": pd.to_numeric(p.bodyWeightLbs, errors="coerce"),
            "pos": pos,
        }
    )


def build_player_games() -> pd.DataFrame:
    cols = ["firstName", "lastName", "personId", "gameId", "gameDateTimeEst", "home",
            "playerteamId", "opponentteamId", "startingPosition", *PLAYER_STAT_COLS]
    df = pd.read_csv(DATA_DIR / "PlayerStatistics.csv", usecols=cols, low_memory=False)
    df = df.dropna(subset=["gameId", "personId"])
    df["gameId"] = df.gameId.astype("int64")
    df = df.join(_game_meta(df.gameId))
    df = df[df.gtype.isin(COMPETITIVE_TYPES) & (df.season >= FIRST_SEASON)].copy()

    df = df.rename(columns=PLAYER_STAT_COLS)
    df["min"] = _parse_minutes(df["min"])
    df = df[df["min"] > 0]  # played in the game
    for c in PLAYER_STAT_COLS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Some 2021-22 rows lack team ids; recover them from Games.csv via the home flag.
    games = pd.read_csv(DATA_DIR / "Games.csv", usecols=["gameId", "gameDateTimeEst", "hometeamId", "awayteamId"])
    games["gameId"] = games.gameId.astype("int64")
    df = df.merge(games.rename(columns={"gameDateTimeEst": "gdate"}), on="gameId", how="left")
    is_home = df.home == 1
    df["playerteamId"] = df.playerteamId.fillna(df.hometeamId.where(is_home, df.awayteamId))
    df["opponentteamId"] = df.opponentteamId.fillna(df.awayteamId.where(is_home, df.hometeamId))
    df["gameDateTimeEst"] = df.gameDateTimeEst.fillna(df.gdate)
    df = df.dropna(subset=["playerteamId", "opponentteamId", "home", "gameDateTimeEst"])

    usage = pd.read_csv(DATA_DIR / "PlayerStatisticsExtended.csv",
                        usecols=["personId", "gameId", "usagePercentage"])
    usage = usage.dropna().astype({"personId": "int64", "gameId": "int64"})
    usage = usage.drop_duplicates(["personId", "gameId"]).rename(columns={"usagePercentage": "usg"})

    out = pd.DataFrame(
        {
            "personId": df.personId.astype("int64"),
            "gameId": df.gameId,
            "date": pd.to_datetime(df.gameDateTimeEst, format="mixed").dt.normalize(),
            "season": df.season.astype(int),
            "playoff": df.gtype.isin(["4", "5"]).astype(int),
            "teamId": df.playerteamId.astype("int64"),
            "oppId": df.opponentteamId.astype("int64"),
            "home": df.home.astype(int),
            "starter": df.startingPosition.notna().astype(int),
            **{c: df[c] for c in PLAYER_STAT_COLS.values()},
        }
    )
    # startingPosition is only a reliable starter flag in some seasons (pre-2017 bench
    # players also carry a position; 2021-22 has none). Null it where the share of
    # "starters" is implausible (~10 of ~21 players per game is normal).
    rate = out.groupby("season").starter.transform("mean")
    out["starter"] = out.starter.astype(float).where(rate.between(0.35, 0.6))
    out = out.merge(usage, on=["personId", "gameId"], how="left")
    out = out.drop_duplicates(["personId", "gameId"])
    return out.sort_values(["personId", "date", "gameId"]).reset_index(drop=True)


def build_team_games(player_games: pd.DataFrame, players_info: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: own pace/offense and defense allowed (overall + per position)."""
    ts = pd.read_csv(
        DATA_DIR / "TeamStatisticsExtended.csv",
        usecols=["gameId", "teamId", "opponentTeamId", "gameDateTimeEst", "home", "win", "teamScore",
                 "opponentScore", "reboundsTotal", "assists", "threePointersMade", "steals", "blocks",
                 "turnovers", "pace", "offensiveRating", "defensiveRating", "effectiveFieldGoalPercentage"],
        low_memory=False,
    )
    ts = ts.dropna(subset=["gameId", "teamId", "opponentTeamId"])
    ts["gameId"] = ts.gameId.astype("int64")
    ts = ts.join(_game_meta(ts.gameId))
    ts = ts[ts.gtype.isin(COMPETITIVE_TYPES) & (ts.season >= FIRST_SEASON)]
    ts = ts.astype({"teamId": "int64", "opponentTeamId": "int64"}).drop_duplicates(["gameId", "teamId"])

    own = ts[["gameId", "teamId", "reboundsTotal", "assists", "threePointersMade", "steals", "blocks",
              "turnovers", "effectiveFieldGoalPercentage"]]
    opp = own.rename(columns=lambda c: c if c == "gameId" else "opp_" + c).rename(columns={"opp_teamId": "opponentTeamId"})
    t = ts.merge(opp, on=["gameId", "opponentTeamId"], how="left")

    tg = pd.DataFrame(
        {
            "gameId": t.gameId,
            "teamId": t.teamId,
            "date": pd.to_datetime(t.gameDateTimeEst, format="mixed").dt.normalize(),
            "season": t.season.astype(int),
            "win": t.win,
            "pace": t.pace,
            "ortg": t.offensiveRating,
            "drtg": t.defensiveRating,
            "pts_for": t.teamScore,
            # what this team's defense allowed
            "allow_pts": t.opponentScore,
            "allow_reb": t.opp_reboundsTotal,
            "allow_ast": t.opp_assists,
            "allow_fg3m": t.opp_threePointersMade,
            "allow_efg": t.opp_effectiveFieldGoalPercentage,
            "allow_tov": t.opp_turnovers,  # turnovers this defense forced
            "def_stl": t.steals,
            "def_blk": t.blocks,
        }
    )

    # Per-36 production allowed to guards / forwards / centers.
    pg = player_games.merge(players_info[["personId", "pos"]], on="personId", how="left")
    pg = pg[pg.pos.isin(["G", "F", "C"])]
    agg = pg.groupby(["gameId", "oppId", "pos"])[["min", "pts", "reb", "ast", "fg3m"]].sum()
    for c in ["pts", "reb", "ast", "fg3m"]:
        agg[c] = agg[c] / agg["min"].clip(lower=1) * 36
    agg = agg.drop(columns="min").unstack("pos")
    agg.columns = [f"allow_{stat}36_{pos}" for stat, pos in agg.columns]
    agg = agg.reset_index().rename(columns={"oppId": "teamId"})
    tg = tg.merge(agg, on=["gameId", "teamId"], how="left")
    return tg.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)


def prepare(force: bool = False):
    """Build (or load cached) player_games, team_games, players_info."""
    ARTIFACT_DIR.mkdir(exist_ok=True)
    paths = {k: ARTIFACT_DIR / f"{k}.parquet" for k in ("player_games", "team_games", "players_info")}
    if not force and all(p.exists() for p in paths.values()):
        return tuple(pd.read_parquet(p) for p in paths.values())
    info = load_players_info()
    pg = build_player_games()
    tg = build_team_games(pg, info)
    for name, frame in zip(paths, (pg, tg, info)):
        frame.to_parquet(paths[name], index=False)
    return pg, tg, info
