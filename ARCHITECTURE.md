# cc-tool architecture

A local, single-user tool that turns credit-card statement PDFs into an
accumulating, categorized spending history. Runs entirely on your machine:
no account, no cloud dependency, no API key required for the core flow.

## Pipeline

```
PDF --> parse --> reconcile --> categorize --> store --> dashboard
```

1. **Parse** (`deterministic.py`): extracts transaction rows (date,
   descriptor, amount, type) from a statement PDF, via `GenericStatementParser`
   -- the only parser this tool has, by design (see "Why no per-issuer
   parsers, no LLM" below). Total detection is reconciliation-driven rather
   than phrase-matched: `_find_total` collects every dollar amount sitting
   near a generic keyword (total/purchase/balance/charges/debits/amount due)
   and accepts whichever one matches the already-extracted rows' sum to the
   cent, so a brand-new issuer's own wording for "total" doesn't need to be
   known in advance. A small exact-phrase list (`_TOTAL_PATTERNS`) is kept as
   a fallback for the rare case where nothing near those keywords reconciles.
2. **Reconcile** (`reconcile.py`): sums the parsed `purchase` rows only (not
   refunds or payments -- the issuer's own printed total is a purchases-only
   figure, so comparing anything else against it produces false failures) and
   checks that sum against the statement's printed total. A statement that
   fails reconciliation is still stored, but flagged for review rather than
   trusted silently.
3. **Categorize** (`categorizer.py`): a local `distilbert-base-uncased`
   classifier, fine-tuned on real merchant descriptors
   (`training/distilbert-uncased-trainer.ipynb`), assigns a spending category to every
   `purchase` row plus a confidence score (its softmax probability). A
   per-merchant cache sits in front of it as pure memoization -- classify a
   normalized merchant once, reuse the answer forever (or until a user
   override replaces it).
4. **Store** (`storage.py`): a single local SQLite file. `statements`
   (deduped by content hash) and `transactions` accumulate across every PDF
   ever imported, which is what makes month-over-month reporting possible.
   `learned_categories` is the current per-merchant answer, with a source
   precedence of `seed < model < user` -- a user correction is never
   silently overwritten by a later model guess.
5. **Dashboard** (`webapp.py` + `static/index.html`): a FastAPI app serving
   one static page. Shows category totals by month across the full history,
   an activity feed of import progress, and the "All merchants" browser.

## Ingestion

Two paths into the same pipeline, both always available:

- **Manual upload** via the dashboard's upload button.
- **Watched drop folder** (`watchdog`): drop a PDF into a folder and it's
  auto-imported in the background.

Both are hash-deduped by `importer.py`, so re-processing the same file is a
no-op. Because uploads are saved into the watched folder (so both paths share
one code path), an upload also fires the watcher for the same file; a lock in
`import_pdf` (and another around `DistilBertCategorizer`'s model-loading
cache) serializes these so they can't race each other into a duplicate
import or a concurrent model load.

## The model + "All merchants"

The categorizer is not assumed to be right. Its own accuracy (~72% test
accuracy / macro F1 on Overture-derived training data, which per prior
evaluation is close to the ceiling set by label noise in that dataset) means
some fraction of categorizations will be wrong, so every categorization
carries a confidence score (the model's softmax probability) alongside it,
purely as information for a human -- not as a gate.

There is no separate confidence-gated review queue: `storage.all_merchants()`
and the dashboard's "All merchants" browser show *every* purchase merchant,
always, with its category, confidence (or "confirmed" once a human has set
it), and total spend, searchable by name. Correcting any merchant writes back
as `source="user"`, which the source-precedence rule (`seed < model < user`)
then protects from ever being overwritten by a future model guess for that
merchant. `DEFAULT_REVIEW_THRESHOLD` in `storage.py` still exists as a
calibration reference point (see `tests/calibrate_confidence.py`) for how
much to trust a given confidence value, but nothing in the live app uses it
to filter what a human can see.

This means the system's accuracy compounds over time: every merchant a human
looks at is permanently resolved, and only genuinely new merchants ever need
a second look -- the human just decides when to look, rather than the app
deciding for them.

## Why no per-issuer parsers, no LLM

This tool has to work on a stranger's own statement PDF, from their own bank,
on their own machine -- not just the handful of issuers its own author
happens to have. That rules out two tempting designs:

- **Per-issuer parsers** (there used to be `CanadianTireParser`, `SimpliiParser`,
  `RogersBankParser`, plus a `DeterministicStatementParser`/`AutoStatementParser`
  escalation tier -- removed). Each one only covers the issuer it was written
  for; a new user's bank just doesn't work until someone writes a parser for
  it. That's not a fixable gap, it's the shape of the approach.
- **An LLM fallback** (there used to be `FuelixStatementParser` and
  `OllamaStatementParser` in `parser.py` -- also removed, along with the whole
  file once nothing else used it). Beyond needing an API key or a local model
  server most users won't have, an LLM extracting dollar amounts can silently
  drop a row, flip a sign, or grab the wrong printed total -- exactly the kind
  of error that shouldn't be possible on data where correctness matters to
  the cent.

Instead, `GenericStatementParser` is issuer-agnostic by construction, and the
one place it used to need issuer-specific knowledge -- recognizing the total
line's exact wording -- was replaced with the reconciliation-driven scheme
described above. Verified against 25 real statements across 4 issuers with
completely different total phrasing, and by masking out every previously-known
exact phrase to confirm the generic-keyword pass alone (not the fallback list)
recovers the total (`tests/test_deterministic.py`).

## Explicitly out of scope right now

- **Hosting this for other people**: multi-user auth, a hosted database,
  discarding raw PDFs for privacy, dropping the drop-folder in favor of
  upload-only -- all real architecture changes if this ever needs to serve
  multiple people from one running instance, as opposed to each person running
  their own local copy. Deferred until the single-instance-per-user case works
  well.

## Model artifact

The trained model is not checked into git. `DistilBertCategorizer` resolves
where to load it from, in order:

1. `CC_TOOL_MODEL_PATH`, if set.
2. `~/.cc_tool/models/distilbert-merchant`, if present locally (e.g. after
   training via `training/distilbert-uncased-trainer.ipynb` and copying its saved
   output folder there).
3. Otherwise, the public Hugging Face Hub repo
   `categorizer.HF_MODEL_REPO_ID` (currently
   `Dluvhugging/cc-tool-merchant-distilbert`) -- `transformers` downloads and
   caches it under `~/.cache/huggingface` on first use, so a fresh install
   with no local model still categorizes correctly with no setup step.

`DistilBertCategorizer` validates at load time that the model's label set
matches `categories.CATEGORY_SET`, so a retrained model with a different
category list fails loudly instead of silently mismatching.

After retraining, push the new weights (and the training data, for
reproducibility) with `python scripts/push_to_hub.py` -- see that script's
docstring. The training data itself lives at the dataset repo
`Dluvhugging/cc-tool-merchant-training-data` on the Hub, sourced from
`data/cc_merchants_overture.csv` (built by `training/build_overture_dataset.py`,
which folds in `training/well_known_merchants.py`'s hand corrections).

## Testing

`tests/` has two different things in it:

- **`test_*.py`** (run via `pytest tests/`, needs the `dev` extra --
  `pip install -e ".[dev]"`): real unit tests against synthetic statement
  text, not real PDFs -- a real statement is a personal financial document and
  those were deliberately kept out of the repo. Covers reconciliation
  (including the refund-handling regression), the reconciliation-driven total
  detection and its exact-phrase fallback, date disambiguation, the itemized-
  appendix and total-for-transaction lookahead logic, storage's source
  precedence, and importer's hash dedup / non-fatal degradation.
- **`calibrate_confidence.py`**: not a pytest test (no `test_` prefix, not
  collected). A manual script against 100 hand-labeled real rows
  (`merchants_gold.csv`) that needs a trained model on disk -- see "Model
  artifact" above and the script's own docstring.

## Local file locations

| What | Location | Override |
|---|---|---|
| SQLite database | `~/.cc_tool/cc_tool.db` | `CC_TOOL_DB` |
| Merchant cache (CLI, no DB) | `~/.cc_tool/merchant_categories.json` | `CC_TOOL_CACHE` |
| Model folder | `~/.cc_tool/models/distilbert-merchant` | `CC_TOOL_MODEL_PATH` |
| Watched drop folder | `~/.cc_tool/inbox` | `cc-tool serve --drop` |
