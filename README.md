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
   The RSS batch collector now writes article/forum text into `gw/gw[N]/text/`; the downstream pipeline only needs plain text files.
4. Generate the report:

   ```zsh
   python fpl_report.py --gameweek N
   ```

   For a data-only report that does not make an OpenAI request:

   ```zsh
   python fpl_report.py --gameweek N --no-llm
   ```

The report is written to `reports/gw_N_report.md`.

## Collecting non-YouTube text

The default source shortlist lives in `lookup/text_sources.json`. That is the right place to keep websites, RSS feeds, and any future subreddit targets because it gives discovery one editable source of truth.

- Extract a known web page into the transcript folder:

   ```zsh
   python scripts/fetch_text.py url 'https://example.com/article' --gameweek N --name article_name
   ```

- Search free RSS feeds for candidate articles:

   ```zsh
  python scripts/discover_text.py rss --query saka --gameweek N --limit 10
   ```

  `--gameweek` prefers posts whose title or summary explicitly mentions that gameweek. Add `--strict-gameweek` if you want to drop ambiguous matches entirely.

- Fetch the linked article text from an RSS entry:

   ```zsh
  python scripts/fetch_text.py rss-item 'https://www.fantasyfootballscout.co.uk/feed/' --match 'Saka' --gameweek N --strict-gameweek
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
   python scripts/fetch_text.py reddit 'https://www.reddit.com/r/FantasyPL/comments/POST_ID/post_title/' --gameweek N
   ```

Each saved text fetch also updates `gw/gwN/data/gw_N_source_manifest.json`, which records the output file, fetch time, source type, original URL, and any source-specific metadata such as feed URL, Reddit subreddit, or title.

## How the report works

- **FPL data (used for "Players To Consider" and the data score)**: recent points, xGI per 90, minutes, availability, next-fixture difficulty, and a 5-gameweek fixture difficulty run.
- **Transcript evidence (analyst score)**: for each transcript, OpenAI is called twice — once to strip ads/intros/filler while preserving FPL-relevant sentences, then again to extract structured `buy`, `consider`, `avoid`, `sell`, `hold`, and `captain` recommendations from the cleaned text. Results are cached on disk per source (`gw/gwN/transcript_cache/`), so re-running a report for the same gameweek only re-calls the API for new or changed transcripts.
- **Decay**: a player's analyst score is pulled toward neutral (50) if they have fewer than 3 independent transcript mentions, so a single confident opinion can't look as strong as a corroborated consensus.
- **Disagreement**: when transcripts disagree on a player, the report shows the split of supportive vs bearish mentions.
- **Combined score**: 50% data score + 50% analyst score, used for "Players To Consider".
- **Players To Avoid**: driven purely by transcript sentiment (analyst score < 50), not the data score — the data score is shown alongside for context only.
- **Name matching**: transcript player names are resolved to FPL IDs via exact match, unique surname match, a small alias table for common nicknames/acronyms (e.g. "KDB", "TAA", "Harland"), then phonetic fuzzy matching.
- **Your squad**: the report evaluates the most recent publicly available team picks. FPL does not expose the upcoming squad through its public API before the deadline.

The report is decision support rather than a point prediction. Check the included evidence, injuries, and press-conference news before acting.

## Notes on cost and caching

Each transcript triggers two OpenAI calls (clean, then extract) the first time it's analysed. Cached results live in `gw/gwN/transcript_cache/`, keyed by a hash of the model, prompt version, and text — delete that folder (or an individual gameweek's) to force a re-analysis, e.g. after a prompt change.
