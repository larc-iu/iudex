"""Round-trip invariants for `RstTree.to_sexp` / `from_sexp` (plan style).

Runs every (traversal_order, include_text) combination over every tree in
RST-DT and GUM. Equality is structural via `RstTree.__eq__` (canonical span
sets); EDU surface strings must match positionally.
"""

import os
from pathlib import Path

import pytest

from iudex.rst.data.reader import read_rst_dir
from iudex.rst.data.tree import RstTree

REPO_ROOT = Path(__file__).resolve().parents[1]

CORPORA = [
    ("rstdt", REPO_ROOT / "data" / "rstdt"),
    ("gum", REPO_ROOT / "data" / "gum_12.1.0_notok"),
]


def _all_trees():
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


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_roundtrip_with_text(all_trees, traversal_order):
    """`include_text=True` reconstructs both structure and EDU text without
    an explicit `edus` argument."""
    failures = []
    for corpus, split, name, tree in all_trees:
        s = tree.to_sexp(traversal_order=traversal_order, include_text=True)
        try:
            reconstructed = RstTree.from_sexp(s, traversal_order=traversal_order)
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


@pytest.mark.parametrize("traversal_order", ["preorder", "postorder"])
def test_roundtrip_without_text(all_trees, traversal_order):
    """`include_text=False` with explicit `edus=tree.edu_strings` reconstructs
    structure; surface forms come from the caller-supplied list."""
    failures = []
    for corpus, split, name, tree in all_trees:
        s = tree.to_sexp(traversal_order=traversal_order, include_text=False)
        try:
            reconstructed = RstTree.from_sexp(s, traversal_order=traversal_order, edus=tree.edu_strings)
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


def test_preorder_postorder_structural_equivalence(all_trees):
    """The two traversal orders produce trees that round-trip to the same
    structure. Cross-check that pre→from_pre and post→from_post agree."""
    failures = []
    for corpus, split, name, tree in all_trees:
        s_pre = tree.to_sexp(traversal_order="preorder", include_text=True)
        s_post = tree.to_sexp(traversal_order="postorder", include_text=True)
        t_pre = RstTree.from_sexp(s_pre, traversal_order="preorder")
        t_post = RstTree.from_sexp(s_post, traversal_order="postorder")
        if t_pre != t_post:
            failures.append((corpus, split, name, "pre vs post structural mismatch"))
        if t_pre.edu_strings != t_post.edu_strings:
            failures.append((corpus, split, name, "pre vs post edu_strings mismatch"))
    if failures:
        raise AssertionError(f"{len(failures)} failures, first 5: {failures[:5]}")
