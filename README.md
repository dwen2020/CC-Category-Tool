# cc-tool

A local, single-user tool that turns credit-card statement PDFs into an
accumulating, categorized spending history. Runs entirely on your machine: no
account, no cloud dependency, no API key required. See `ARCHITECTURE.md` for
how it works internally.

## Install

```
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

```
cc-tool parse statement.pdf          # parse one PDF, print extracted rows
cc-tool import statement.pdf         # parse + categorize + store it
cc-tool import ~/Statements/         # import every PDF in a folder
cc-tool report                       # spending by category, by month
cc-tool set "some merchant" Dining   # correct/pin a merchant's category
cc-tool serve                        # local web dashboard + watched drop folder
```

`cc-tool serve` starts a dashboard at `http://127.0.0.1:8765` and watches
`~/.cc_tool/inbox` (override with `--drop`) -- drop a statement PDF in there
and it's imported automatically.

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
