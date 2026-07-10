"""Throwaway: dump what extract_text_from_pdf feeds the LLM. Delete when done."""

import sys

from cc_tool.parser import extract_text_from_pdf

path = sys.argv[1] if len(sys.argv) > 1 else "simplii.pdf"
text = extract_text_from_pdf(open(path, "rb").read())
print(f"=== {path}: {len(text)} chars (~{len(text)//4} tokens est.) ===\n")
print(text)
