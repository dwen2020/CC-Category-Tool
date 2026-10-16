# cc-tool

A tool to upload credit-card statement PDFs and track spending across
categories -- especially useful if you have several different credit cards.

Runs entirely on your machine: no account, no cloud dependency, no API key
required. It turns your statements into an accumulating, categorized spending
history.

- Parses statement PDFs from any issuer -- there's no per-bank code, and no
  LLM in the parsing path (see "Why no per-issuer parsers, no LLM" in
  `ARCHITECTURE.md`).
- Checks every parsed statement against its own printed total before trusting
  it; anything that doesn't reconcile is still stored, but flagged for you to
  look at rather than silently accepted.
- Categorizes each purchase into one of twelve spending categories (Groceries,
  Dining, Transport, Travel, Shopping, Utilities, Health, Entertainment, Home,
  Services, Financial, Other) using a small local model -- any correction you
  make is remembered permanently for that merchant.
- A local web dashboard shows spending by category over time, plus a
  searchable list of every merchant it's seen.
- A watched drop folder auto-imports new statements, or upload them by hand.

See `ARCHITECTURE.md` for how it works internally, and `BUILDING.md` for the
story of how it got to this design (including the approaches that didn't
work out).

## Install

This isn't published to PyPI, so grab the source first:

```
git clone https://github.com/dwen2020/CC-Category-Tool.git
cd CC-Category-Tool
pip install .
```

(or `pip install -e .` for an editable install if you're developing on it).

This installs a `cc-tool` command. Categorization needs a trained model,
which is not bundled in the package: the first time you run it, `cc-tool`
downloads the public model from the Hugging Face Hub automatically -- no
extra setup step. If you'd rather use your own locally-trained model, point
`CC_TOOL_MODEL_PATH` at its folder (see "Model artifact" in
`ARCHITECTURE.md`).

## Usage

**The web dashboard is the main way to use cc-tool -- most people only ever
need this one command:**

```
cc-tool serve
```

This starts a UI at `http://127.0.0.1:8765` and watches `~/.cc_tool/inbox`
(override with `--drop`): drop a statement PDF in there, or use the upload
button in the dashboard, and it's parsed, categorized, and stored
automatically. From the dashboard you can see spending by category over
time, browse every merchant it's seen, and correct a miscategorized one with
a click.

The rest of the CLI exists alongside it for scripting or terminal use, and
reads/writes the exact same local data as the dashboard:

```
cc-tool parse statement.pdf          # parse one PDF, print extracted rows
cc-tool import statement.pdf         # parse + categorize + store it
cc-tool import ~/Statements/         # import every PDF in a folder
cc-tool report                       # spending by category, by month
cc-tool set "some merchant" Dining   # correct/pin a merchant's category
```

## Data

Everything lives locally under `~/.cc_tool/` (override with `CC_TOOL_DB`,
`CC_TOOL_CACHE`, `CC_TOOL_MODEL_PATH` -- see `ARCHITECTURE.md`'s "Local file
locations" table). Nothing is uploaded anywhere; the only network call this
tool ever makes is the one-time model download on first use.

## Development

```
pip install -e ".[dev]"
pytest tests/
```

Training the categorization model, and pushing a retrained model to the Hub,
are covered in `training/` and `scripts/push_to_hub.py`.
