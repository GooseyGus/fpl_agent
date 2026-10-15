# fpl_hq

Fantasy Premier League analysis package. `README.md` has the full folder map, data layout and weekly workflow.

## Commands

- Setup: `pip install -r requirements.txt && pip install -e .`
- Tests: `pytest` (smoke test that every module imports, so run it after moving or renaming files)
- Run modules as `python -m fpl_hq.<subpackage>.<module>` **from the repo root**. iCloud can hide the editable-install `.pth` file in `.venv`, so imports only work from there.

## Conventions

- All file locations come from `fpl_hq/paths.py`. Never hard-code paths. The season is set by `SEASON` there.
- `data/` is not in git and is regenerated, except `data/reference/`. `name_aliases.json` is hand-curated, so don't overwrite it.
- `archive/` is legacy code for reference only. Don't edit it or import from it.
- Scripts in `apps/` and `pipelines/` add the repo root to `sys.path` themselves.
- LLM (OpenAI) calls belong only in `fpl_hq/consensus/`. Report generation must stay deterministic.
- OpenAI results are cached in `data/<season>/gameweeks/gwN/*_cache/`. Avoid anything that clears these, because re-running costs money.
- Secrets live in `.env` (`OPENAI_API_KEY`, Reddit keys). Never print or commit them.
