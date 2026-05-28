"""ICL pilot: pick a serialization format and run the in-context-learning loop.

Two formats are supported:
  - sr_words: shift-reduce action sequence with whitespace-split words in
    place of subword COPY tokens. Format matches
    `actions_to_strings(tree.to_shift_reduce(include_text=True))` joined by
    single spaces. Decoding: whitespace-tokenize the model output, run
    `strings_to_actions` with a reduce-token map built from the GUM relation
    inventory, then `RstTree.from_shift_reduce`.
  - sexp: s-expression serialization matching `RstTree.to_sexp(format=
    "iudex")`: internal nodes are `(NUC:relation child1 child2)` with NUC in
    {NS, SN, NN}, leaves are `(EDU text)` with literal parens in EDU text
    escaped to `-LRB-`/`-RRB-`. There's no `from_sexp` in
    `iudex.rst.data.tree`. The reader below targets exactly that one output
    format, so we're inverting an existing format, not inventing one.

This file is intentionally disposable, so comments lean liberal. The shared
loader, eval loop, metrics, and CLI plumbing live in `_common.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repo root is importable when running this script directly (same as
# `_common.py`).
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _common import (  # noqa: E402
    TaskBundle,
    doc_text,
    fmt_relation_inventory,
    fmt_sexp_relation_inventory,
    load_corpora,
    run_eval,
)
from iudex.common.log import console, rule  # noqa: E402
from iudex.rst.data.tree import (  # noqa: E402
    Reduce,
    RstTree,
    Shift,
    actions_to_strings,
    strings_to_actions,
)


# ---------------------------------------------------------------------------
# sr_words format
# ---------------------------------------------------------------------------


SR_WORDS_TASK_DESCRIPTION = """\
You are doing Rhetorical Structure Theory (RST) discourse parsing. RST analyzes
a document as a binary tree over Elementary Discourse Units (EDUs, roughly
clause-level text spans). Internal nodes are labeled with a relation (e.g.
"causal-cause", "joint-list") and a nuclearity pattern:
  - NS: left child is the nucleus, right child is the satellite
  - SN: left child is the satellite, right child is the nucleus
  - NN: multinuclear, both children are nuclei (used for symmetric relations)

You will serialize the parse as a SHIFT-REDUCE action sequence. Reading left
to right:
  - Emit the next EDU's whitespace-split words, then a "<shift>" token, to
    push that EDU onto the parse stack.
  - Emit a single "<reduce_*>" token to combine the top two stack items into
    a parent node. The reduce token names which (nuclearity, relation) pair
    is assigned to the new node.

For a document with N EDUs, the action sequence contains exactly N "<shift>"
tokens and exactly N-1 "<reduce_*>" tokens.

The only legal reduce tokens are these (one per `(nuclearity, relation)`
combination over the GUM relation inventory):

{relation_inventory}

Output ONLY the linearized action sequence, no commentary, no extra lines.
"""


def _build_reduce_token_map(relation_types) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for rel, kind in relation_types:
        nucs = ("NN",) if kind == "multinuc" else ("NS", "SN")
        for nuc in nucs:
            tok = Reduce(nuc=nuc, rel=rel).to_token()
            out[tok] = (nuc, rel)
    return out


def build_prefix_sr_words(bundle: TaskBundle) -> str:
    inv = fmt_relation_inventory(bundle.relation_types)
    head = SR_WORDS_TASK_DESCRIPTION.format(relation_inventory=inv)
    parts = [head, f"\nHere are {len(bundle.icl_sample)} worked examples.\n"]
    for i, tree in enumerate(bundle.icl_sample, 1):
        doc_text_str = " ".join(tree.edu_strings)
        actions = tree.to_shift_reduce(include_text=True)
        out_str = " ".join(actions_to_strings(actions))
        parts.append(f"--- Example {i} ---")
        parts.append(f"INPUT: {doc_text_str}")
        parts.append(f"OUTPUT: {out_str}")
        parts.append("")
    return "\n".join(parts)


def parse_response_sr_words(response: str, gold: RstTree, bundle: TaskBundle) -> RstTree:
    # The model may wrap its answer in code fences or include a leading line
    # like "OUTPUT:". Strip lightly. We don't try to be clever about repair.
    text = response.strip()
    for marker in ("OUTPUT:", "Output:", "```text", "```"):
        if text.startswith(marker):
            text = text[len(marker) :].lstrip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    # Some models wrap each line. Whitespace-tokenize the whole blob.
    tokens = text.split()
    reduce_token_map = _build_reduce_token_map(bundle.relation_types)
    actions = strings_to_actions(tokens, reduce_token_map)

    # Truncate to the action prefix that uses exactly N shifts and N-1 reduces
    # if the model over-generated. Simpler than full repair. Most failure modes
    # are "good prefix + garbage tail".
    n_shifts = 0
    n_reduces = 0
    for a in actions:
        if isinstance(a, Shift):
            n_shifts += 1
        elif isinstance(a, Reduce):
            n_reduces += 1
    if n_shifts == 0 and n_reduces == 0:
        raise ValueError("no shift/reduce actions in model output")

    return RstTree.from_shift_reduce(actions, relation_types=bundle.relation_types)


# ---------------------------------------------------------------------------
# sexp format
# ---------------------------------------------------------------------------


SEXP_TASK_DESCRIPTION = """\
You are doing Rhetorical Structure Theory (RST) discourse parsing. RST analyzes
a document as a binary tree over Elementary Discourse Units (EDUs, roughly
clause-level text spans). Internal nodes are labeled with a relation (e.g.
"causal-cause", "joint-list") and a nuclearity pattern:
  - NS: left child is the nucleus, right child is the satellite
  - SN: left child is the satellite, right child is the nucleus
  - NN: multinuclear, both children are nuclei (used for symmetric relations)

You will serialize the parse as an S-expression. Each internal node has the
form `(NUC:relation CHILD1 CHILD2)` with exactly two children. Each leaf has
the form `(EDU surface text)`. Literal parentheses inside EDU text are
escaped to `-LRB-` and `-RRB-`.

The whole document is one S-expression. For a document with N EDUs, the
expression contains exactly N `(EDU ...)` leaves and exactly N-1 internal
nodes.

The only legal head tags on internal nodes are:

{relation_inventory}

Output ONLY the S-expression, no commentary, no markdown fences.
"""


def build_prefix_sexp(bundle: TaskBundle) -> str:
    inv = fmt_sexp_relation_inventory(bundle.relation_types)
    head = SEXP_TASK_DESCRIPTION.format(relation_inventory=inv)
    parts = [head, f"\nHere are {len(bundle.icl_sample)} worked examples.\n"]
    for i, tree in enumerate(bundle.icl_sample, 1):
        doc_text_str = " ".join(tree.edu_strings)
        out_str = tree.to_sexp(format="iudex")
        parts.append(f"--- Example {i} ---")
        parts.append(f"INPUT: {doc_text_str}")
        parts.append(f"OUTPUT: {out_str}")
        parts.append("")
    return "\n".join(parts)


def _tokenize_sexp(s: str) -> List[str]:
    """Token types: `(`, `)`, or any maximal run of non-whitespace non-paren
    characters. This is the standard tiny-lisp tokenizer. It's safe here
    because EDU text has its parens escaped to `-LRB-`/`-RRB-`."""
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "(" or c == ")":
            out.append(c)
            i += 1
            continue
        j = i
        while j < n and not s[j].isspace() and s[j] not in "()":
            j += 1
        out.append(s[i:j])
        i = j
    return out


def _parse_one(tokens: List[str], pos: int):
    """Recursive-descent. Returns (node, new_pos). Node shape mirrors
    `RstTree._build_binary_tree` so we can reuse downstream logic:
      - leaves: ("edu", text)
      - internal: ("node", nuc, rel, left, right)
    """
    if pos >= len(tokens) or tokens[pos] != "(":
        raise ValueError(f"sexp: expected '(' at token {pos}, got {tokens[pos : pos + 1]!r}")
    pos += 1
    if pos >= len(tokens):
        raise ValueError("sexp: unexpected EOF after '('")
    head = tokens[pos]
    pos += 1
    if head == "EDU":
        # Collect text tokens until the matching ')'. Multi-word EDUs are
        # space-rejoined. Nested parens cannot occur in this format because
        # EDU text has its parens escaped.
        text_tokens: List[str] = []
        depth = 0
        while pos < len(tokens):
            t = tokens[pos]
            if t == "(":
                depth += 1
                text_tokens.append(t)
            elif t == ")":
                if depth == 0:
                    pos += 1
                    text = " ".join(text_tokens)
                    # Undo PTB-style escaping
                    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
                    return ("edu", text), pos
                depth -= 1
                text_tokens.append(t)
            else:
                text_tokens.append(t)
            pos += 1
        raise ValueError("sexp: unclosed EDU leaf")
    # Internal node. head must look like NUC:relation
    if ":" not in head:
        raise ValueError(f"sexp: internal node head missing ':': {head!r}")
    nuc, rel = head.split(":", 1)
    if nuc not in ("NS", "SN", "NN"):
        raise ValueError(f"sexp: unknown nuclearity {nuc!r}")
    left, pos = _parse_one(tokens, pos)
    right, pos = _parse_one(tokens, pos)
    if pos >= len(tokens) or tokens[pos] != ")":
        raise ValueError(f"sexp: expected ')' to close internal node at {pos}")
    pos += 1
    return ("node", nuc, rel, left, right), pos


def _flatten_to_actions(node) -> Tuple[List[Tuple[int, str, str]], List[str]]:
    """Walk the nested tuple to produce (parsing_actions, edu_strings) suitable
    for `RstTree.from_parsing_actions`. parsing_actions uses post-order
    `(split_index, nuc, rel)` where split_index is the index of the first EDU
    in the right child (matching `RstTree.parsing_actions`).
    """
    edus: List[str] = []

    def visit(n) -> Tuple[List[int], List[Tuple[int, str, str]]]:
        if n[0] == "edu":
            idx = len(edus)
            edus.append(n[1])
            return [idx], []
        _, nuc, rel, left, right = n
        # First recurse so we know the children's index ranges, but emit the
        # parent's action BEFORE the children's. `from_parsing_actions`
        # iterates in reversed order, so we need top-down (parent-first) order
        # in the list, which makes the reverse bottom-up, matching the merge
        # order `from_parsing_actions` requires.
        l_idx, l_acts = visit(left)
        r_idx, r_acts = visit(right)
        split = r_idx[0]
        return (l_idx + r_idx, [(split, nuc, rel)] + l_acts + r_acts)

    _, actions = visit(node)
    return actions, edus


def _validate_parsing_actions(actions, edus, rel_set):
    if len(actions) != len(edus) - 1:
        raise ValueError(f"sexp: expected {len(edus) - 1} internal nodes, got {len(actions)}")
    # Relation presence check. Fail loud if the model invented a label.
    for _, _, rel in actions:
        if rel not in rel_set:
            raise ValueError(f"sexp: unknown relation {rel!r}")


def parse_response_sexp(response: str, gold: RstTree, bundle: TaskBundle) -> RstTree:
    text = response.strip()
    for marker in ("OUTPUT:", "Output:", "```text", "```sexp", "```scheme", "```lisp", "```"):
        if text.startswith(marker):
            text = text[len(marker) :].lstrip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    # Trim anything before the first '(' (some models add a preamble)
    first = text.find("(")
    if first == -1:
        raise ValueError("no '(' in model output")
    text = text[first:]
    # Trim trailing junk after the matching root close-paren.
    depth = 0
    end = None
    for i, c in enumerate(text):
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("unbalanced parens in model output")
    text = text[:end]

    tokens = _tokenize_sexp(text)
    node, pos = _parse_one(tokens, 0)
    if pos != len(tokens):
        # Trailing junk that the depth-walker missed (shouldn't happen, but be
        # defensive).
        pass

    actions, edus = _flatten_to_actions(node)
    rel_set = {r for r, _ in bundle.relation_types}
    _validate_parsing_actions(actions, edus, rel_set)

    if len(edus) != len(gold.edu_strings):
        # Surface this here for a clearer error than the generic mismatch
        # later.
        raise ValueError(f"sexp: parsed {len(edus)} EDUs but gold has {len(gold.edu_strings)}")

    return RstTree.from_parsing_actions(actions, edus, relation_types=bundle.relation_types)


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------


FORMATS = {
    "sr_words": (build_prefix_sr_words, parse_response_sr_words),
    "sexp": (build_prefix_sexp, parse_response_sexp),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    p = argparse.ArgumentParser(prog="icl_pilot")
    p.add_argument(
        "--format",
        choices=sorted(FORMATS.keys()),
        required=True,
        help="serialization format for the ICL prompt and the decoder",
    )
    p.add_argument("--model", default="claude-opus-4-7", help="litellm model string")
    p.add_argument("--limit", type=int, default=None, help="only process the first N dev docs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=5)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="assemble the prompt for the first dev doc and print it; skip API calls",
    )
    args = p.parse_args()

    build_prefix, parse_response = FORMATS[args.format]

    if args.dry_run:
        bundle = load_corpora(seed=args.seed, k=args.k)
        if args.limit is not None:
            bundle.dev_pairs = bundle.dev_pairs[: args.limit]
        rule(f"ICL pilot DRY RUN: {args.format} via {args.model}")
        prefix = build_prefix(bundle)
        if not bundle.dev_pairs:
            console.print(prefix)
            console.print("\n[no dev docs after limit; printed prefix only]")
            return
        path, gold = bundle.dev_pairs[0]
        dtext = doc_text(gold)
        full_prompt = prefix + f"\n--- Now parse this document ---\nINPUT: {dtext}\nOUTPUT: "
        console.print(full_prompt)
        console.print(
            f"\n[dry-run: prompt chars={len(full_prompt)}, ~{len(full_prompt) // 4} tokens; doc_id={Path(path).stem}]"
        )
        return

    run_eval(
        fmt=args.format,
        model=args.model,
        build_prefix=build_prefix,
        parse_response=parse_response,
        limit=args.limit,
        seed=args.seed,
        k=args.k,
    )


if __name__ == "__main__":
    main()
