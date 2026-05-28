"""Round-trip invariants for `RstTree.to_shift_reduce` / `from_shift_reduce`
and the `actions_to_strings` / `strings_to_actions` helpers.

Runs over every tree in RST-DT and GUM. Equality is structural via
`RstTree.__eq__` (canonical span sets).
"""

import os
from pathlib import Path

import pytest

from iudex.rst.data.reader import read_rst_dir
from iudex.rst.data.tree import (
    Reduce,
    RstTree,
    Shift,
    actions_to_strings,
    strings_to_actions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CORPORA = [
    ("rstdt", REPO_ROOT / "data" / "rstdt"),
    ("gum", REPO_ROOT / "data" / "gum_12.1.0"),
]


def _all_trees():
    """Yield (corpus_name, split, filename, tree) for every tree in
    RST-DT and GUM. Skips silently if a corpus directory is absent so the
    suite stays runnable in partial environments."""
    for corpus_name, corpus_root in CORPORA:
        if not corpus_root.is_dir():
            continue
        for split in ("train", "dev", "test"):
            split_dir = corpus_root / split
            if not split_dir.is_dir():
                continue
            for filepath, tree in read_rst_dir(str(split_dir)):
                yield corpus_name, split, os.path.basename(filepath), tree


@pytest.fixture(scope="module")
def all_trees():
    out = list(_all_trees())
    if not out:
        pytest.skip("No RST-DT or GUM data found under data/.")
    return out


def _reduce_map_for(tree: RstTree) -> dict:
    """Derive a complete reduce_token_map by enumerating every (nuc, rel)
    pair that could appear in this tree's shift-reduce serialization."""
    # The action sequence itself names exactly the (nuc, rel) pairs needed
    # for this tree. Building the map from the tree's own actions is
    # sufficient for round-trip testing.
    actions = tree.to_shift_reduce(include_text=False)
    return {r.to_token(): (r.nuc, r.rel) for r in actions if isinstance(r, Reduce)}


def test_action_roundtrip_with_text(all_trees):
    """Path 1: include_text=True → from_shift_reduce reconstructs both
    structure and EDU text without an explicit `edus` list."""
    failures = []
    for corpus, split, name, tree in all_trees:
        actions = tree.to_shift_reduce(include_text=True)
        try:
            reconstructed = RstTree.from_shift_reduce(actions)
        except Exception as e:
            failures.append((corpus, split, name, f"raised: {e!r}"))
            continue
        if reconstructed != tree:
            failures.append((corpus, split, name, "tree != reconstructed"))
            continue
        if reconstructed.edu_strings != tree.edu_strings:
            failures.append((corpus, split, name, "edu_strings differ"))
    if failures:
        raise AssertionError(f"{len(failures)} failures, first 5: {failures[:5]}")


def test_action_roundtrip_explicit_edus(all_trees):
    """Path 2: include_text=False with explicit `edus=tree.edu_strings`."""
    failures = []
    for corpus, split, name, tree in all_trees:
        actions = tree.to_shift_reduce(include_text=False)
        try:
            reconstructed = RstTree.from_shift_reduce(actions, edus=tree.edu_strings)
        except Exception as e:
            failures.append((corpus, split, name, f"raised: {e!r}"))
            continue
        if reconstructed != tree:
            failures.append((corpus, split, name, "tree != reconstructed"))
            continue
        if reconstructed.edu_strings != tree.edu_strings:
            failures.append((corpus, split, name, "edu_strings differ"))
    if failures:
        raise AssertionError(f"{len(failures)} failures, first 5: {failures[:5]}")


def test_string_roundtrip(all_trees):
    """Path 3: actions_to_strings → strings_to_actions → from_shift_reduce.
    This is the path the seq2seq parser actually exercises."""
    failures = []
    for corpus, split, name, tree in all_trees:
        actions = tree.to_shift_reduce(include_text=True)
        strings = actions_to_strings(actions)
        reduce_map = _reduce_map_for(tree)
        try:
            actions2 = strings_to_actions(strings, reduce_map)
            reconstructed = RstTree.from_shift_reduce(actions2)
        except Exception as e:
            failures.append((corpus, split, name, f"raised: {e!r}"))
            continue
        if reconstructed != tree:
            failures.append((corpus, split, name, "tree != reconstructed"))
            continue
        if reconstructed.edu_strings != tree.edu_strings:
            failures.append((corpus, split, name, "edu_strings differ"))
        if actions != actions2:
            failures.append((corpus, split, name, "action lists differ"))
    if failures:
        raise AssertionError(f"{len(failures)} failures, first 5: {failures[:5]}")


def test_shifts_balanced(all_trees):
    """Sanity: every tree's SR sequence has n shifts and n-1 reduces, where
    n is len(tree.edus)."""
    failures = []
    for corpus, split, name, tree in all_trees:
        actions = tree.to_shift_reduce()
        n_shift = sum(1 for a in actions if isinstance(a, Shift))
        n_reduce = sum(1 for a in actions if isinstance(a, Reduce))
        n_edu = len(tree.edus)
        if n_shift != n_edu:
            failures.append((corpus, split, name, f"{n_shift} shifts, {n_edu} edus"))
        if n_reduce != n_edu - 1:
            failures.append((corpus, split, name, f"{n_reduce} reduces, expected {n_edu - 1}"))
    if failures:
        raise AssertionError(f"{len(failures)} failures, first 5: {failures[:5]}")
