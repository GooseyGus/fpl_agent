# fpl_hq

All-in-one Fantasy Premier League analysis home: data gathering, expert-opinion
consensus, statistics, projections, optimisation, backtesting, and reports.

## Map

Data flows one way: **gather → consensus / stats / projections → optimiser → reports**, with
backtest checking every stage against what actually happened.

| Folder | What lives there |
|---|---|
| `fpl_hq/paths.py` | The single source of truth for every file location. Change the season here. |
| `fpl_hq/gather/` | Fetching: FPL API players and fixtures, YouTube transcripts, web/RSS/Reddit text |
| `fpl_hq/consensus/` | Opinion mining: LLM pipelines that turn transcripts and articles into player stances, plus Bradley-Terry rankings |
| `fpl_hq/stats/` | Statistical analysis (form, fixtures, team strength). *Empty for now* |
| `fpl_hq/projections/` | Expected-points models. *Empty for now: to be extracted from `notebooks/research/expected_points.ipynb`* |
| `fpl_hq/ml/` | Machine-learning models. *Empty for now* |
| `fpl_hq/optimiser/` | PuLP squad selection and transfers: constraints, objective, squad state, runners |
| `fpl_hq/backtest/` | Scores signals and models against final points |
| `fpl_hq/reports/` | Deterministic consensus report and CSV |
| `pipelines/` | Multi-step entry points that chain the stages together |
| `apps/` | Streamlit tools (name-resolution review; future dashboard) |
| `notebooks/pipeline/` | Notebooks you run each week (statistics, transcripts, text) |
| `notebooks/research/` | Exploration: expected points, neural network, old squad-selection notebook |
| `tests/` | `pytest`: currently checks every module imports |
| `data/` | Not in git, apart from `data/reference/` (see below) |
| `archive/` | Not in git. Old FPL projects kept for reference, each with its own git history |

### Data layout

```
data/
├── reference/              tracked in git: element/team lookups, name_aliases.json (hand-curated), text_sources.json
├── history/                past seasons: vaastav/ (2016-26), fpl_api/ snapshots, history_past.csv
└── 2026-27/
    ├── gameweeks/gwN/      transcripts/, text/, data/, *_cache/   (per-gameweek inputs, caches, derived JSON)
    ├── season_data/        season-long player and fixture tables
    ├── reports/gw_N/       consensus CSV + markdown report
    ├── optimiser/          squads and optimiser reports
    └── backtest/           backtest outputs
```

Next season: set `SEASON` in `fpl_hq/paths.py`. Don't copy the repo.

## Setup

```zsh
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Create `.env` with `OPENAI_API_KEY=...` (plus `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` for Reddit search).

Run every `python -m fpl_hq...` command **from the repo root**. On macOS, iCloud can mark
files inside `.venv` as hidden. Python then skips the editable-install `.pth` file, so
`fpl_hq` only imports when the repo root is your working directory. Notebooks, `apps/` and
`pipelines/` handle this themselves.

## Weekly workflow

1. **Statistics**: open `notebooks/pipeline/statistics_processor.ipynb`, set `gameweek`, and run all cells.
2. **Transcripts**: list YouTube URLs in `data/2026-27/gameweeks/gwN/transcripts/transcript_urls.txt`, then:
   ```zsh
   python -m fpl_hq.gather.fetch_transcript --gameweek N --method text_file
   ```
3. **Articles**: list URLs in `data/2026-27/gameweeks/gwN/text/text_urls.txt`, or fill the folder from RSS:
   ```zsh
   python -m fpl_hq.gather.fetch_text --gameweek N --method text_file --min-year 2026
   python -m fpl_hq.gather.fill_text_folder --query fpl --gameweek N --limit 5
   ```
4. **Consensus**: run `notebooks/pipeline/transcripts_processor.ipynb` and `notebooks/pipeline/text_processor.ipynb`.
   Optionally review name resolution:
   ```zsh
   streamlit run apps/review_name_resolution.py -- --gameweek N
   python -m fpl_hq.consensus.transcripts.5_comparisons --gameweek N   # head-to-head comparisons for Bradley-Terry
   ```
5. **Report**:
   ```zsh
   python -m fpl_hq.reports.fpl_report --gameweek N
   ```
   This writes `data/2026-27/reports/gw_N/gw_N_consensus.csv` and `gw_N_report.md`.
6. **Optimise**:
   ```zsh
   python -m fpl_hq.optimiser.run_transfer_optimiser --gameweek N             # your real squad: transfers that maximise consensus
   python -m fpl_hq.optimiser.run_optimiser --gameweek N --no-open-report
   python -m fpl_hq.optimiser.run_dual_optimiser --gameweek N              # API ep_next vs custom gus_ep_next
   python pipelines/run_optimiser_week.py [N]                              # refresh data, then run the dual optimiser
   ```

## Backtesting

```zsh
python -m fpl_hq.backtest.run_backtest
```

This scores consensus, Bradley-Terry, and baseline signals (ownership, net transfers, form,
form + fixture) against final points for every finished gameweek with extractions. Final
points come from the FPL `event/{gw}/live` endpoint and are cached per gameweek.

## How the consensus report works

- **Inputs**: transcript and text extractions, the aggregated player form, and `data/reference/element_lookup.json`.
- **Exact player join**: every recommendation must match exactly one canonical name. The report stops and lists any unmatched names.
- **Consensus**: one stance (1–5) per source, the mean of its mentions, so repeat mentions add no votes. The mean across sources is shrunk toward neutral with a two-source prior, alongside source and mention counts.
- **No report LLM**: report generation makes no OpenAI request. LLM use stays upstream, where speech and articles become structured recommendations.

## Cost and caching

Each transcript costs OpenAI calls the first time it's analysed. Results are cached under
`data/2026-27/gameweeks/gwN/*_cache/`, keyed by model, prompt version, and text. Delete a
cache folder to force re-analysis, for example after a prompt change.
