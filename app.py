"""Web app for the NBA player performance model.

Run:  .venv/bin/streamlit run app.py
"""
import json

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from nba_model.data import ARTIFACT_DIR, prepare
from nba_model.features import TARGETS
from predict import LABELS, build_rows, load_schedule, load_teams, next_meeting, predict, rotation

SERIES = "#2a78d6"   # categorical slot 1 (reads on light and dark surfaces)
NEUTRAL = "#8a8984"  # model range / prediction marks
HEADLINE = ["pts", "reb", "ast", "fg3m", "min"]
SHORT = {"pts": "PTS", "reb": "REB", "ast": "AST", "fg3m": "3PM", "stl": "STL", "blk": "BLK",
         "tov": "TOV", "min": "MIN"}

st.set_page_config(page_title="NBA Player Predictor", page_icon="🏀", layout="wide")


@st.cache_resource(show_spinner="Loading model and data…")
def load():
    if not (ARTIFACT_DIR / "model.joblib").exists():
        return None
    bundle = joblib.load(ARTIFACT_DIR / "model.joblib")
    pg, tg, info = prepare()
    teams, sched = load_teams(), load_schedule()
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    last_data = pg.date.max()
    latest = pg.sort_values("date").groupby("personId").tail(1)
    active = latest[latest.date >= last_data - pd.Timedelta(days=400)]
    active = active.merge(info[["personId", "firstName", "lastName"]], on="personId")
    active = active.merge(teams[["teamId", "full"]], on="teamId")
    active["name"] = active.firstName.fillna("") + " " + active.lastName.fillna("")
    active = active.sort_values("name").set_index("personId")
    return dict(bundle=bundle, pg=pg, tg=tg, info=info, teams=teams.set_index("teamId"), sched=sched,
                metrics=metrics, last_data=last_data, upcoming=sched[sched.date > last_data], active=active)


D = load()
if D is None:
    st.error("No trained model found. Run `caffeinate -i .venv/bin/python train.py --rebuild-data` first.")
    st.stop()
teams, active, last_data, upcoming = D["teams"], D["active"], D["last_data"], D["upcoming"]


@st.cache_data(show_spinner="Predicting…")
def run(games: tuple):
    """games: tuple of (personId, teamId, oppId, home, 'YYYY-MM-DD')."""
    rows = [{"personId": p, "teamId": t, "oppId": o, "home": h, "date": pd.Timestamp(d), "gid": -1}
            for p, t, o, h, d in games]
    X = build_rows(rows, D["pg"], D["tg"], D["info"], D["sched"])
    return X, predict(X, D["bundle"])


def team_name(tid, short=False):
    return str(teams.loc[tid, "name" if short else "full"]) if tid in teams.index else "?"


def default_date(meet):
    return (meet.date if meet is not None else last_data + pd.Timedelta(days=2)).date()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🏀 NBA Player Predictor")
    st.write(f"Data through **{last_data:%b %d, %Y}**. Predictions assume the player plays — "
             "check the injury report. The model doesn't know about injuries or teammates being out.")
    st.subheader("Accuracy on unseen games")
    st.caption("Average miss per game, 2025-26 and 2026-27 season so far. Lower is better.")
    m = D["metrics"]
    st.dataframe(pd.DataFrame({
        "Stat": [LABELS[t] for t in TARGETS],
        "Model": [round(m[t]["model_mae"], 2) for t in TARGETS],
        "Last-10 avg": [round(m[t]["last10_avg_mae"], 2) for t in TARGETS],
    }), hide_index=True, width="stretch")
    st.caption("Retrain after updating `nba_data/`: `caffeinate -i .venv/bin/python train.py --rebuild-data`")

tab_player, tab_game = st.tabs(["Player vs team", "Full game"])

# ---------------------------------------------------------------- player tab
def player_tab():
    ids = list(active.index)
    curry = active.index[active.name == "Stephen Curry"]
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    pid = c1.selectbox("Player", ids, index=ids.index(curry[0]) if len(curry) else 0,
                       format_func=lambda p: f"{active.loc[p, 'name']} · {team_name(int(active.loc[p, 'teamId']), True)}")
    tid = int(active.loc[pid, "teamId"])
    opp_ids = [t for t in teams.sort_values("full").index if t != tid]
    nxt = upcoming[(upcoming.homeTeamId == tid) | (upcoming.awayTeamId == tid)]
    nxt_opp = None if nxt.empty else int(nxt.iloc[0].awayTeamId if nxt.iloc[0].homeTeamId == tid else nxt.iloc[0].homeTeamId)
    oid = c2.selectbox("Opponent", opp_ids, index=opp_ids.index(nxt_opp) if nxt_opp in opp_ids else 0,
                       format_func=team_name, key=f"opp_{pid}")
    first = next_meeting(tid, oid, upcoming)
    venue = c3.radio("Venue", ["Home", "Away"], horizontal=True,
                     index=0 if first is None or first.homeTeamId == tid else 1, key=f"venue_{pid}_{oid}")
    home = int(venue == "Home")
    meet = next_meeting(tid, oid, upcoming, home=home)
    date = c4.date_input("Game date", value=default_date(meet), min_value=(last_data + pd.Timedelta(days=1)).date(),
                         key=f"date_{pid}_{oid}_{home}")
    if meet is not None and pd.Timestamp(date) == meet.date:
        st.caption(f"Next scheduled {venue.lower()} game vs the {team_name(oid, True)}.")
    elif meet is None:
        st.caption(f"No {venue.lower()} game vs the {team_name(oid, True)} left on the schedule — predicting a hypothetical one.")

    hist = D["pg"][D["pg"].personId == pid]
    if pd.Timestamp(date) <= hist.date.max():
        st.warning(f"Pick a date after the player's last game ({hist.date.max():%b %d, %Y}).")
        return

    X, P = run(((pid, tid, oid, home, str(date)),))
    x, p = X.iloc[0], P.iloc[0]

    st.subheader(f"{active.loc[pid, 'name']} {'vs' if home else '@'} {team_name(oid)}")
    rest = "" if x.rest_days != x.rest_days else f" · {int(x.rest_days)} day(s) rest" + (" · back-to-back" if x.b2b == 1 else "")
    st.caption(f"{pd.Timestamp(date):%A, %B %d, %Y} · {team_name(tid)} {'at home' if home else 'on the road'}{rest}")
    if hist.date.max() < last_data - pd.Timedelta(days=30):
        st.warning(f"This player's last game in the data was {hist.date.max():%b %d, %Y}, so the prediction may be stale.")

    cols = st.columns(len(HEADLINE) + 1)
    for col, t in zip(cols, HEADLINE):
        col.metric(LABELS[t], f"{p[t]:.1f}", f"{p[t] - x[f'{t}_l10']:+.1f} vs last 10", delta_color="off",
                   help=f"80% range: {p[t + '_lo']:.0f}–{p[t + '_hi']:.0f}")
    cols[-1].metric("PTS+REB+AST", f"{p.pts + p.reb + p.ast:.1f}",
                    f"{p.pts + p.reb + p.ast - x.pts_l10 - x.reb_l10 - x.ast_l10:+.1f} vs last 10", delta_color="off")

    left, right = st.columns([5, 4])
    with left:
        st.markdown("**Prediction vs. the player's averages**")
        st.dataframe(pd.DataFrame({
            "Stat": [LABELS[t] for t in TARGETS],
            "Predicted": [round(p[t], 1) for t in TARGETS],
            "80% range": [f"{p[t + '_lo']:.0f} – {p[t + '_hi']:.0f}" for t in TARGETS],
            "Last 10": [round(x[f"{t}_l10"], 1) for t in TARGETS],
            "Season": [None if x[f"{t}_szn"] != x[f"{t}_szn"] else round(x[f"{t}_szn"], 1) for t in TARGETS],
            f"vs {team_name(oid, True)}": [round(x[f"{t}_vs_opp"], 1) if f"{t}_vs_opp" in x and x[f"{t}_vs_opp"] == x[f"{t}_vs_opp"]
                                           else None for t in TARGETS],
        }), hide_index=True, width="stretch")
        st.caption("The 80% range is where the real result landed 8 times in 10 for similar predictions on unseen games. "
                   f"vs {team_name(oid, True)} = average over the last {int(x.opp_n)} meetings (up to 12).")

    with right:
        st.markdown("**What moved the prediction**")

        def effect(v, unit="pts"):
            return "no data" if v != v else f"{v:+.1f} {unit}"
        st.markdown(
            f"- **{venue} games:** {effect(x.pts_venue_resid)} vs. the player's usual form\n"
            f"- **Against the {team_name(oid, True)}:** {effect(x.pts_opp_resid)} vs. usual "
            f"({int(x.opp_n)} recent meetings, smaller samples count less)\n"
            f"- **{team_name(oid, True)} defense:** {effect(x.opp_drtg_rel, 'pts/100 poss.')} vs. league average "
            f"({'worse' if x.opp_drtg_rel > 0 else 'better'} than average)\n"
            f"- **Points allowed to this position:** {effect(x.opp_allow_pts36_mypos_rel, 'per 36 min')} vs. league\n"
            f"- **Opponent pace:** {effect(x.opp_pace_rel, 'possessions')} vs. league"
        )
        st.caption("These are the inputs the model sees, not exact contributions. It weighs them against recent form, "
                   "which matters most.")

    # recent games chart: actual stat per game, with the model's range for the next game
    st.markdown("**Recent games**")
    stat = st.segmented_control("Stat", [t for t in TARGETS], default="pts", format_func=lambda t: LABELS[t],
                                key="chart_stat", label_visibility="collapsed") or "pts"
    recent = hist.tail(15).copy()
    recent["label"] = [f"{d:%b %d} {'vs' if h else '@'} {team_name(o, True)}" for d, h, o in
                       zip(recent.date, recent.home, recent.oppId)]
    recent["value"] = recent[stat]
    nxt_label = f"Next: {'vs' if home else '@'} {team_name(oid, True)}"
    order = list(recent.label) + [nxt_label]
    pred_df = pd.DataFrame({"label": [nxt_label], "value": [p[stat]], "lo": [p[stat + "_lo"]], "hi": [p[stat + "_hi"]]})
    xenc = alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=120))
    yenc = alt.Y("value:Q", title=LABELS[stat], scale=alt.Scale(zero=True))
    line = alt.Chart(recent).mark_line(color=SERIES, strokeWidth=2).encode(x=xenc, y=yenc)
    dots = alt.Chart(recent).mark_point(color=SERIES, filled=True, size=70).encode(
        x=xenc, y=yenc, tooltip=[alt.Tooltip("label:N", title="Game"), alt.Tooltip("value:Q", title=LABELS[stat]),
                                 alt.Tooltip("min:Q", title="Minutes", format=".0f")])
    band = alt.Chart(pred_df).mark_rule(color=NEUTRAL, strokeWidth=10, opacity=0.35, strokeCap="round").encode(
        x=alt.X("label:N", sort=order), y="lo:Q", y2="hi:Q")
    pred_pt = alt.Chart(pred_df).mark_point(color=NEUTRAL, filled=True, size=110, shape="diamond").encode(
        x=alt.X("label:N", sort=order), y="value:Q",
        tooltip=[alt.Tooltip("label:N", title="Game"), alt.Tooltip("value:Q", title="Predicted", format=".1f"),
                 alt.Tooltip("lo:Q", title="80% low", format=".0f"), alt.Tooltip("hi:Q", title="80% high", format=".0f")])
    st.altair_chart((band + line + dots + pred_pt).properties(height=300), width="stretch")
    st.caption("Blue: actual results in the player's last 15 games. Gray diamond and bar: the prediction and its 80% range.")

    log_cols = {"date": "Date", "opp": "Opponent", "min": "MIN", "pts": "PTS", "reb": "REB", "ast": "AST",
                "fg3m": "3PM", "stl": "STL", "blk": "BLK", "tov": "TOV"}

    def game_log(df):
        df = df.sort_values("date", ascending=False).copy()
        df["opp"] = [f"{'vs' if h else '@'} {team_name(o, True)}" for h, o in zip(df.home, df.oppId)]
        df["date"] = df.date.dt.strftime("%Y-%m-%d")
        df["min"] = df["min"].round(0)
        return df[list(log_cols)].rename(columns=log_cols)

    g1, g2 = st.columns(2)
    g1.markdown("**Last 10 games**")
    g1.dataframe(game_log(hist.tail(10)), hide_index=True, width="stretch")
    g2.markdown(f"**Last 10 games vs the {team_name(oid, True)}**")
    vs = hist[hist.oppId == oid].tail(10)
    if vs.empty:
        g2.info("No previous games against this team.")
    else:
        g2.dataframe(game_log(vs), hide_index=True, width="stretch")

# ---------------------------------------------------------------- game tab
def game_tab():
    order_ids = list(teams.sort_values("full").index)
    c1, c2, c3 = st.columns([3, 3, 2])
    gs = upcoming.iloc[0] if len(upcoming) else None
    ht = c1.selectbox("Home team", order_ids, format_func=team_name,
                      index=order_ids.index(gs.homeTeamId) if gs is not None else 0)
    away_ids = [t for t in order_ids if t != ht]
    at = c2.selectbox("Away team", away_ids, format_func=team_name,
                      index=away_ids.index(gs.awayTeamId) if gs is not None and gs.awayTeamId in away_ids else 0)
    meet = next_meeting(ht, at, upcoming, home=1)
    gdate = c3.date_input("Game date", value=default_date(meet), min_value=(last_data + pd.Timedelta(days=1)).date(),
                          key=f"gdate_{ht}_{at}")
    st.caption("Next scheduled meeting at this venue." if meet is not None and pd.Timestamp(gdate) == meet.date
               else "Not on the schedule — predicting a hypothetical game.")

    games = tuple((p, int(t), int(o), h, str(gdate))
                  for t, o, h in ((ht, at, 1), (at, ht, 0)) for p in rotation(t, D["pg"], D["tg"]))
    if not games:
        st.info("No recent players found for these teams.")
        return
    X, P = run(games)
    P = P.reset_index()
    P["teamId"] = P.personId.map({g[0]: g[1] for g in games})
    P["Player"] = active.reindex(P.personId).name.values
    P["PRA"] = P.pts + P.reb + P.ast

    st.subheader(f"{team_name(at)} @ {team_name(ht)}")
    st.caption(f"{pd.Timestamp(gdate):%A, %B %d, %Y} · everyone who played for each team in its last 5 games, "
               "assuming they all play. Sorted by predicted minutes.")
    cols = ["Player", "min", "pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "PRA"]
    for col, t, where in zip(st.columns(2), (ht, at), ("home", "away")):
        col.markdown(f"**{team_name(t)}** ({where})")
        df = P[P.teamId == t].sort_values("min", ascending=False)[cols].rename(columns=SHORT)
        col.dataframe(df.round(1), hide_index=True, width="stretch",
                      height=min(38 + 35 * len(df), 600))


with tab_player:
    player_tab()
with tab_game:
    game_tab()
