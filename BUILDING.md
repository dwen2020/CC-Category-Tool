# How this was built

This is the story of how cc-tool got to its current shape, including the
approaches that didn't work out. `ARCHITECTURE.md` describes what the system
does today; this document is about how it got there and why.

## Why this exists

This started as a personal project: a way to see spending by category across
several credit cards without manually reconciling statements every month, and
a chance to learn how to fine-tune a small language model for a real task
rather than a toy dataset.

## Parsing statements: starting with an LLM, ending without one

The very first design for this tool used an LLM to read each statement PDF
and extract the transaction rows -- a reconciliation step (checking that the
extracted rows summed to the statement's own printed total) was there from
day one, to catch cases where the model dropped a row or misread an amount.

That design worked for personal use, but it doesn't hold up once the goal
becomes "someone else can run this on their own statements." A cloud LLM
call means every user needs an API key (a real cost and setup barrier), and
even a well-behaved LLM can silently drop a row, flip a sign, or grab the
wrong total on data where correctness needs to be exact to the cent. The
project went through a phase of issuer-specific parsers (one per bank) plus
an LLM fallback for anything those didn't cover -- better for accuracy, but
it doesn't scale either: a new user's bank just doesn't work until someone
writes a parser for it.

The eventual approach drops both: a single generic parser handles layout
extraction for any statement, and instead of needing to know each issuer's
exact wording for "total" in advance, it collects every dollar amount near a
generic keyword (total, purchase, balance, charges, debits, amount due) and
accepts whichever one matches the already-extracted rows' own sum to the
cent. Reconciliation stopped being just a pass/fail check and became the
mechanism that finds the total in the first place. This was verified against
real statements across several issuers with completely different wording for
their totals, plus a stress test that masked out every known phrase to
confirm the generic pass alone -- not a fallback list -- could still find it.

## Categorizing merchants: what didn't work first

The categorization model went through several rounds of "this looked
promising, then didn't."

A synthetically generated training set was the first attempt -- and looked
great in early evaluation, until it became clear the generator had leaked
the category into the merchant name itself ("Maple Dental" scores as Health
because the word "Dental" is right there), so the benchmark was measuring
the generator, not the model. A public dataset of labeled transactions
looked promising for scale, but turned out to hide only a few hundred
distinct real merchant names behind millions of templated rows, and it
didn't transfer to real Canadian merchant descriptors at all.

The training data that actually worked comes from real business listings
(Overture Maps), labeled by their real business category rather than
anything derived from the name -- a much more honest test of whether the
model learned anything.

On the model side, a few encoder backbones were compared before settling on
DistilBERT, and a fine-tuned generative model (Qwen, via LoRA) was also
tried as a genuine alternative -- it scored higher in that comparison. It
didn't ship, for a few reasons together: the main point of this stage of the
project was learning to fine-tune BERT specifically, a generative model is
slower to run locally at import time than a small classifier's single
forward pass, and its outputs were less consistent than a closed-set
classification head that can only ever pick one of the twelve real
categories.

## Making it work for someone besides the author

Everything above was originally built for one person's own statements. Once
the goal became "let other people use this," two designs that were already
in place had to come back out: the issuer-specific parsers and the LLM
fallback, for the reasons above. What replaced them (the reconciliation-
driven parser) had to be validated harder than "it works on my statements"
-- hence the stress test that hid every known phrase and checked the generic
mechanism still worked.

A test suite followed for the same reason: manual spot-checking against real
statements doesn't scale as a way to keep confidence that a change to the
parser or reconciliation logic hasn't broken something. The tests run
against synthetic statement text rather than real PDFs, since a real
statement is a personal financial document that shouldn't need to exist in
this repository for the tests to work.

## Shipping it

The last piece was distribution. The tool itself packages as a normal
Python package (`pip install .`), but the trained model is ~250MB of
weights that can't reasonably live in git. It's hosted on the Hugging Face
Hub instead, and the tool downloads it automatically on first use if no
local model is found -- so installing the package is enough to get a working
categorizer, with no separate model-training step required for someone who
just wants to use the tool.
