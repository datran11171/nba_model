# nba_model

Predicts a player's stat line (points, rebounds, assists, 3PM, steals, blocks,
turnovers, minutes) for a game against a specific opponent, at home or away.

## Setup

Put the dataset CSVs in `nba_data/` (gitignored), then:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
caffeinate -i .venv/bin/python train.py --rebuild-data   # ~30 min; caffeinate keeps a Mac awake
```

`--rebuild-data` re-reads the CSVs. Use it whenever `nba_data/` is updated with new games.
Without it, training reuses the cleaned tables cached in `artifacts/`.

## Web app (easiest)

```bash
.venv/bin/streamlit run app.py
```

This opens http://localhost:8501 in your browser.

- **Player vs team:** pick a player, an opponent, home or away, and a date (it defaults to the
  next scheduled meeting). You get the predicted stat line with 80% ranges, the player's
  averages for comparison, what moved the prediction, and a chart of recent games next to
  the prediction.
- **Full game:** pick two teams and get predictions for both rotations.

## Command line

```bash
.venv/bin/python predict.py player "Stephen Curry" --opp Lakers --home
.venv/bin/python predict.py player "Jokic" --opp Celtics --away --date 2026-12-25
.venv/bin/python predict.py player "Anthony Edwards" --next     # next game on the schedule
.venv/bin/python predict.py game Lakers Warriors                # home team first, both rotations
```

Without `--date`, the next scheduled meeting from `LeagueSchedule26_27.csv` is used.
Output shows the prediction, an 80% range, and the player's last-10, season, and
vs-this-opponent averages for comparison. It also shows the venue and matchup
effects the model applied.

Predictions assume the player plays. The model doesn't know about injuries, rest
decisions, or teammates being out.

## How it works

One gradient-boosted model per stat (scikit-learn `HistGradientBoostingRegressor`,
Poisson loss for counts), trained on ~580k player-games from 2004-05 onward.
Every feature uses only games before the one being predicted:

- **Recent form:** last 5/10/20 games, exponentially weighted average, season-to-date, last season, per-36 rates, usage
- **Home/away split:** how the player performs at this venue *relative to their own form at the time*, shrunk toward zero when the sample is small
- **Matchup history:** the same relative measure over the last 12 games against this opponent
- **Opponent defense:** defensive rating, pace, points/rebounds/assists/3s allowed, and per-36 production allowed to the player's position (G/F/C), each compared with the league average for that season
- **Context:** rest days, back-to-backs, age, starter rate, own team's pace and offense

Code: [nba_model/data.py](nba_model/data.py) (cleaning), [nba_model/features.py](nba_model/features.py)
(features), [train.py](train.py), [predict.py](predict.py).

## Accuracy

Trained through 2024-25, tested on 2025-26 and 2026-27 to date (games the model never saw).
Mean absolute error compared with simply using the player's recent average:

| Stat | Model MAE | Last-10 avg MAE | Season avg MAE |
|---|---|---|---|
| Points | **4.51** | 4.71 | 4.73 |
| Rebounds | **1.86** | 1.94 | 1.93 |
| Assists | **1.32** | 1.37 | 1.37 |
| 3PM | **0.88** | 0.90 | 0.90 |
| Steals | **0.72** | 0.74 | 0.73 |
| Blocks | 0.52 | 0.51 | **0.51** |
| Turnovers | **0.89** | 0.92 | 0.92 |
| Minutes | **4.81** | 5.29 | 5.37 |

The 80% ranges are calibrated from the errors on those test seasons, so they reflect
how far off predictions really were on unseen games. Single-game stats are noisy: a typical points prediction is off by about 4.5. Most of
that is game-to-game randomness that no pre-game model can remove. Full numbers are
in `artifacts/metrics.json` after training.
