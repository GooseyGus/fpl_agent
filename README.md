# FPL Advice Report

This project combines the FPL data created by `data_generator.ipynb` with manually collected YouTube transcripts to produce a Markdown decision-support report.

## One-time setup

1. Activate the existing virtual environment:

   ```zsh
   source .venv/bin/activate
   ```

2. Install the report dependencies:

   ```zsh
   pip install -r requirements.txt
   ```

3. Create `.env` from `.env.example`, then put your own OpenAI API key in `.env`. Keep this file private; it is excluded from Git.

## Weekly workflow

1. Open `data_generator.ipynb`.
2. Set `gameweek` to the upcoming deadline and run every cell from top to bottom.
3. Put the relevant YouTube transcript files in `gw/gw[N]/transcripts/`.
   To fetch them from YouTube URLs, create `gw/gwN/transcripts/transcript_urls.txt`
   with one URL or video ID per line, then run:

   ```zsh
   python notebooks/transcripts_processor/fetch_transcript.py --gameweek N --method text_file
   ```

   Blank lines and lines beginning with `#` are ignored. The command removes
   duplicate video IDs, skips transcripts already present in the folder, and waits
   30-90 seconds between requests to reduce the chance of YouTube blocking the batch.
   For website articles, put one URL per line in `gw/gwN/text/text_urls.txt` and run:

   ```zsh
   python notebooks/text_processor/fetch_text.py --gameweek N --method text_file --min-year 2026
   ```

   Add `--min-year 2026` to skip pre-2026 articles in the batch list.

   The RSS batch collector now writes article/forum text into `gw/gw[N]/text/`; the downstream pipeline only needs plain text files.
4. Generate the report:

   ```zsh
   python fpl_report.py --gameweek N
   ```

The command writes both the full Markdown report to `reports/gw_N_report.md`
and a two-column `player_id,Consensus` table to `reports/gw_N_consensus.csv`.

### Review transcript name resolution

After running the transcript name-resolution step, launch the local review dashboard:

```zsh
streamlit run notebooks/transcripts_processor/review_name_resolution.py -- --gameweek N
```

The dashboard starts with unresolved mentions in `gw/gwN/transcript_cache/name_resolution_cache/` and records resumable decisions in `gw/gwN/data/gw_N_name_resolution_reviews.json`. Once every unresolved mention has been reviewed, it presents the existing mappings from `lookup/name_aliases.json` in a random order so each can be confirmed, corrected, or removed. Confirmed and corrected mention mappings are also written to the alias file; unsure and not-a-player decisions are kept only in the review file. Alias changes are part of the resolver cache key, so rerunning the name-resolution notebook step regenerates results with the updated aliases.

### Extract head-to-head comparisons

After the transcript notebook has written `gw_N_transcript_selection.json`, extract explicit player comparisons ("Bruno is as good as Palmer this week") for Bradley-Terry scoring:

```zsh
python -m notebooks.transcripts_processor.5_comparisons --gameweek N
```

This makes one OpenAI call per transcript, cached in `gw/gwN/transcript_cache/comparison_cache/`, and writes `gw/gwN/data/gw_N_transcript_comparisons.json`. Comparisons whose quote is not verbatim in the transcript, or whose players do not resolve exactly to the roster, are dropped and counted.

## Backtesting

Score consensus, Bradley-Terry, and baseline signals (ownership, net transfers, form, form + fixture) against final points for every finished gameweek with extractions:

```zsh
python -m backtest.run_backtest
```

Final points come from the FPL `event/{gw}/live` endpoint and are cached as `gw/gwN/data/gw_N_final_points.json`; the saved `api_data_player_stats` files can be captured mid-gameweek. Bradley-Terry strengths (`reports/report_scripts/bradley_terry.py`) anchor every 1-5 stance to a neutral reference player and refine the order with comparisons, because comparisons on their own only rank players within the shortlists creators discuss.

## Collecting non-YouTube text

The default source shortlist lives in `lookup/text_sources.json`. That is the right place to keep websites, RSS feeds, and any future subreddit targets because it gives discovery one editable source of truth.

- Extract a known web page into the text folder:

   ```zsh
   python notebooks/text_processor/fetch_text.py url 'https://example.com/article' --gameweek N --name article_name
   ```

- Search free RSS feeds for candidate articles:

   ```zsh
   python scripts/discover_text.py rss --query saka --gameweek N --limit 10
   ```

  `--gameweek` prefers posts whose title or summary explicitly mentions that gameweek. Add `--strict-gameweek` if you want to drop ambiguous matches entirely.

- Fetch the linked article text from an RSS entry:

   ```zsh
   python notebooks/text_processor/fetch_text.py rss-item 'https://www.fantasyfootballscout.co.uk/feed/' --match 'Saka' --gameweek N --strict-gameweek
   ```

- Fill a whole gameweek folder in one command:

   ```zsh
   python scripts/fill_text_folder.py --query fpl --gameweek N --limit 5
   ```

   This searches the curated RSS shortlist, filters by the fixture-derived publish window for the target gameweek, skips URLs containing `ucl`, writes the results into `gw/gwN/text/`, and updates the source manifest automatically.

- Run a first-pass LLM cleaner over the saved article text:

   ```zsh
   python notebooks/text_processor/1_filter_text.py --gameweek N
   ```

   This reads every file in `gw/gwN/text/`, keeps only the FPL-relevant content, and writes the cleaned copies to `gw/gwN/text_clean/`. The raw source files stay untouched, and repeated runs reuse a per-file cache in `gw/gwN/text_cache/`.

- Search Reddit through the official free Reddit API:

   ```zsh
   python scripts/discover_text.py reddit --query wildcard --subreddit FantasyPL --limit 10
   ```

   Reddit search requires `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in your shell environment or `.env`. Direct Reddit post extraction is also available with:

   ```zsh
   python notebooks/text_processor/fetch_text.py reddit 'https://www.reddit.com/r/FantasyPL/comments/POST_ID/post_title/' --gameweek N
   ```

Each saved text fetch also updates `gw/gwN/data/gw_N_source_manifest.json`, which records the output file, fetch time, source type, original URL, and any source-specific metadata such as feed URL, Reddit subreddit, or title.

## How the report works

- **Inputs**: the report reads only `gw_N_transcript_extractions.json`, `gw_N_aggregated_player_form.json`, and `lookup/element_lookup.json`.
- **Exact player join**: `element_lookup.json` is inverted into canonical name to player ID. Every final recommendation must exactly match one canonical name, and every resulting ID must exist in the aggregate. The report stops with all unmatched names listed if either contract is broken.
- **Grouping**: recommendations are grouped by player ID, preserving every source's stance, explanation, and supporting quote.
- **Consensus**: the report shows the raw mean stance, source and mention counts, stance distribution, and standard deviation. Its ranking score shrinks small samples toward neutral using a two-mention prior.
- **Output**: one row per source-mentioned player, containing the FPL player ID and consensus score, ordered from highest to lowest consensus.
- **No report LLM**: report generation is deterministic and makes no OpenAI request. LLM usage remains upstream in transcript processing, where unstructured speech is converted into reviewed structured recommendations.

The report is decision support rather than a point prediction. Check the included evidence, injuries, and press-conference news before acting.

## Notes on cost and caching

Each transcript triggers two OpenAI calls (clean, then extract) the first time it's analysed. Cached results live in `gw/gwN/transcript_cache/`, keyed by a hash of the model, prompt version, and text — delete that folder (or an individual gameweek's) to force a re-analysis, e.g. after a prompt change.
