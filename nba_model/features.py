"""Pre-game feature engineering.

Every feature for a game uses only games strictly before it (shift(1) before
any rolling/expanding window), so the same code builds training rows and rows
for a hypothetical upcoming game appended to the history.
"""
import numpy as np
import pandas as pd

TARGETS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "min"]
FORM_STATS = TARGETS + ["fga", "fta", "fg3a", "usg", "pm"]
SPLIT_STATS = ["pts", "reb", "ast", "fg3m", "min"]  # home/away and vs-opponent splits

TEAM_STATS = ["pace", "ortg", "drtg", "pts_for", "allow_pts", "allow_reb", "allow_ast",
              "allow_fg3m", "allow_efg", "allow_tov", "def_stl", "def_blk", "win"]
POS_STATS = ["pts36", "reb36", "ast36", "fg3m36"]


class _Cols(dict):
    """dict of columns supporting f[[a, b]] = 2-D array; avoids fragmented frames."""

    def __init__(self, index):
        super().__init__()
        self.index = index

    def __setitem__(self, key, value):
        if isinstance(key, list):
            value = np.asarray(value)
            for i, k in enumerate(key):
                super().__setitem__(k, value[:, i])
        else:
            super().__setitem__(key, np.asarray(value))

    def __getitem__(self, key):
        return pd.Series(super().__getitem__(key), index=self.index)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(dict(self.items()), index=self.index)


def _shifted(df: pd.DataFrame, keys, cols) -> pd.DataFrame:
    return df.groupby(keys, sort=False)[cols].shift(1)


def _group_rolling(values: pd.DataFrame, groups: pd.Series, window: int) -> pd.DataFrame:
    """Rolling mean of already-shifted values within groups (vectorised via cumsum)."""
    v = values.fillna(0)
    n = values.notna().astype(float)
    cs, cn = v.groupby(groups).cumsum(), n.groupby(groups).cumsum()
    cs_lag = cs.groupby(groups).shift(window).fillna(0)
    cn_lag = cn.groupby(groups).shift(window).fillna(0)
    return (cs - cs_lag) / (cn - cn_lag).replace(0, np.nan)


def _group_ewm(values: pd.DataFrame, groups: pd.Series, halflife: float) -> pd.DataFrame:
    out = values.groupby(groups).ewm(halflife=halflife, ignore_na=True).mean()
    return out.reset_index(level=0, drop=True).sort_index()


def player_features(pg: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """pg must be sorted by personId, date. Returns feature frame aligned with pg."""
    pg = pg.reset_index(drop=True)
    pid = pg.personId
    f = _Cols(pg.index)

    prev = _shifted(pg, "personId", FORM_STATS)
    for w in (5, 10, 20):
        r = _group_rolling(prev, pid, w)
        f[[f"{c}_l{w}" for c in FORM_STATS]] = r.values
    ewm = _group_ewm(prev, pid, 8)
    f[[f"{c}_ewm" for c in FORM_STATS]] = ewm.values

    ps = pid.astype(str) + "_" + pg.season.astype(str)
    prev_s = prev.copy()
    # season-to-date: restart at each season (first game of season has no prior)
    prev_s[_shifted(pg.assign(ps=ps), "ps", ["season"]).season.isna().values] = np.nan
    cs = prev_s.fillna(0).groupby(ps).cumsum()
    cn = prev_s.notna().astype(float).groupby(ps).cumsum()
    f[[f"{c}_szn" for c in FORM_STATS]] = (cs / cn.replace(0, np.nan)).values
    f["szn_games"] = pg.groupby(ps).cumcount()
    f["career_games"] = pg.groupby("personId").cumcount()

    # last season's averages (carry into the new one)
    season_avg = pg.groupby(["personId", "season"])[TARGETS].mean()
    season_avg.index = season_avg.index.set_levels(season_avg.index.levels[1] + 1, level=1)
    last = pg[["personId", "season"]].merge(
        season_avg.add_suffix("_lastszn").reset_index(), on=["personId", "season"], how="left")
    f[[f"{c}_lastszn" for c in TARGETS]] = last[[f"{c}_lastszn" for c in TARGETS]].values

    # per-minute production rates
    for c in ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov"):
        f[f"{c}_per36_l20"] = f[f"{c}_l20"] / f["min_l20"].clip(lower=1) * 36

    # role / schedule context
    starter_prev = _shifted(pg, "personId", ["starter"]).starter
    f["started_prev"] = starter_prev.values
    f["starter_l10"] = _group_rolling(starter_prev.to_frame(), pid, 10).values.ravel()
    gap = pg.groupby("personId").date.diff().dt.days
    f["rest_days"] = gap.clip(upper=10).values
    f["b2b"] = (gap == 1).astype(float).where(gap.notna()).values
    team_change = pg.groupby("personId").teamId.shift(1).ne(pg.teamId) & f["career_games"].gt(0)
    f["games_with_team"] = pg.groupby([pid, team_change.cumsum()]).cumcount().values

    # Residual vs. the player's own form -> venue and opponent effects.
    # These say "how much better/worse than usual" rather than raw averages,
    # so a player's growth over the years doesn't contaminate the split.
    resid = pg[SPLIT_STATS] - ewm[SPLIT_STATS].values
    for key, name, window, shrink in (("home", "venue", 40, 8), ("oppId", "opp", 12, 4)):
        grp = pid.astype(str) + "_" + pg[key].astype(str)
        rprev = resid.groupby(grp).shift(1)
        n = rprev.notna().astype(float).groupby(grp).cumsum()
        n = n - n.groupby(grp).shift(window).fillna(0)
        s = rprev.fillna(0).groupby(grp).cumsum()
        s = s - s.groupby(grp).shift(window).fillna(0)
        f[[f"{c}_{name}_resid" for c in SPLIT_STATS]] = (s / (n + shrink)).values  # shrunk toward 0
        f[f"{name}_n"] = n[SPLIT_STATS[0]].values
    # raw career average vs this opponent (for display + extra signal)
    grp = pid.astype(str) + "_" + pg.oppId.astype(str)
    oprev = pg.groupby(grp)[["pts", "reb", "ast"]].shift(1)
    f[["pts_vs_opp", "reb_vs_opp", "ast_vs_opp"]] = _group_rolling(oprev, grp, 12).values

    # static player info
    m = pg[["personId", "date"]].merge(info[["personId", "birthDate", "height", "weight", "pos"]],
                                       on="personId", how="left")
    f["age"] = ((m.date - m.birthDate).dt.days / 365.25).values
    f["height"] = m.height.values
    f["weight"] = m.weight.values
    f["pos"] = m.pos.map({"G": 0, "F": 1, "C": 2}).values

    f["home"] = pg.home.values
    f["playoff"] = pg.playoff.values
    f["season"] = pg.season.values
    return f.frame()


def team_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Pre-game team form, keyed by (gameId, teamId)."""
    tg = tg.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)
    pos_cols = [f"allow_{s}_{p}" for s in POS_STATS for p in "GFC"]
    cols = TEAM_STATS + pos_cols
    prev = _shifted(tg, "teamId", cols)
    out = tg[["gameId", "teamId", "season"]].copy()
    out[[f"{c}_ewm" for c in cols]] = _group_ewm(prev, tg.teamId, 10).values
    # season-to-date win%
    key = tg.teamId.astype(str) + "_" + tg.season.astype(str)
    w = tg.groupby(key).win.shift(1)
    out["win_pct_szn"] = (w.groupby(key).cumsum() / w.notna().groupby(key).cumsum().replace(0, np.nan)).values

    # League baseline per season (prior games only) so defense is relative to era.
    lg = tg.sort_values(["date", "gameId"])
    lg_cols = ["pace", "drtg", "allow_pts"] + pos_cols
    daily = lg.groupby(["season", "date"])[lg_cols].agg(["sum", "count"])
    cum = daily.groupby(level="season").cumsum().groupby(level="season").shift(1)
    base = pd.DataFrame({c: cum[(c, "sum")] / cum[(c, "count")] for c in lg_cols})
    prior_season = tg.groupby("season")[lg_cols].mean()
    prior_season.index += 1
    base = base.reset_index()
    fill = base[["season"]].merge(prior_season.reset_index(), on="season", how="left")
    base[lg_cols] = base[lg_cols].fillna(fill[lg_cols])
    base = base.rename(columns={c: f"lg_{c}" for c in lg_cols})
    out = out.merge(tg[["gameId", "teamId", "date"]], on=["gameId", "teamId"]).merge(
        base, on=["season", "date"], how="left")
    return out.drop(columns=["date"])


def assemble(pg: pd.DataFrame, tg: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Full feature matrix (+targets/ids) for every row of pg."""
    pg = pg.sort_values(["personId", "date", "gameId"]).reset_index(drop=True)
    pf = player_features(pg, info)
    tf = team_features(tg)

    ids = pg[["personId", "gameId", "date", "teamId", "oppId", *TARGETS]]
    X = pd.concat([ids, pf], axis=1)

    own = tf[["gameId", "teamId", "pace_ewm", "ortg_ewm", "pts_for_ewm", "win_pct_szn"]]
    own = own.rename(columns=lambda c: c if c in ("gameId", "teamId") else "team_" + c)
    X = X.merge(own, on=["gameId", "teamId"], how="left")

    opp = tf.drop(columns=["season"]).rename(
        columns=lambda c: "oppId" if c == "teamId" else (c if c == "gameId" else "opp_" + c))
    X = X.merge(opp, on=["gameId", "oppId"], how="left")

    # opponent's allowed production to *this player's* position, relative to league
    new = {}
    for s in POS_STATS:
        val = np.select([X.pos == 0, X.pos == 1, X.pos == 2],
                        [X[f"opp_allow_{s}_{p}_ewm"] for p in "GFC"], default=np.nan)
        lg = np.select([X.pos == 0, X.pos == 1, X.pos == 2],
                       [X[f"opp_lg_allow_{s}_{p}"] for p in "GFC"], default=np.nan)
        new[f"opp_allow_{s}_mypos"] = val
        new[f"opp_allow_{s}_mypos_rel"] = val - lg
    for c in ("pace", "drtg", "allow_pts"):
        new[f"opp_{c}_rel"] = X[f"opp_{c}_ewm"] - X[f"opp_lg_{c}"]
    X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
    drop = [c for c in X.columns if c.startswith("opp_lg_")] + \
           [f"opp_allow_{s}_{p}_ewm" for s in POS_STATS for p in "GFC"]
    return X.drop(columns=drop)


ID_COLS = ["personId", "gameId", "date", "teamId", "oppId"]


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ID_COLS and c not in TARGETS]
