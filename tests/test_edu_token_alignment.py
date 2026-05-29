"""Verify `align_edus_to_tokens` tiles tokens across EDUs on the real corpora,
and that it fixes the old overlap-method drift. CPU only, tokenizer only."""

import os

import pytest

from iudex.rst.data.reader import read_rst_dir
from iudex.rst.parsers.common.seqgen import align_edus_to_tokens

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TOKENIZER_NAME = "google/t5gemma-2-1b-1b"
GUM_TRAIN_CAP = 60  # cap train split for speed; dev/test run in full

CORPORA = {
    "gum": "gum_12.1.0_notok",
    "rstdt": "rstdt",
}


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _reconstruct_text(edus):
    parts = []
    for i, edu in enumerate(edus):
        if i == 0:
            parts.append(edu.text)
            continue
        prefix = edu.prefix if edu.prefix is not None else " "
        parts.append(prefix + edu.text)
    return "".join(parts)


def _char_ends(edus):
    ends = []
    cursor = 0
    for i, edu in enumerate(edus):
        if i > 0:
            prefix = edu.prefix if edu.prefix is not None else " "
            cursor += len(prefix)
        cursor += len(edu.text)
        ends.append(cursor)
    return ends


def _old_overlap_total(tokenizer, text, edus):
    """Sum of old overlap-slice lengths; != len(input_ids) signals drift."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    total = 0
    char_cursor = 0
    for i, edu in enumerate(edus):
        if i > 0:
            prefix = edu.prefix if edu.prefix is not None else " "
            char_cursor += len(prefix)
        char_start = char_cursor
        char_cursor += len(edu.text)
        char_end = char_cursor
        first = last = None
        for j, (tcs, tce) in enumerate(offsets):
            if tcs < char_end and tce > char_start:
                if first is None:
                    first = j
                last = j
        if first is not None:
            total += last + 1 - first
    return total, len(enc["input_ids"])


def _load(corpus_dir, split, cap=None):
    d = os.path.join(DATA_ROOT, corpus_dir, split)
    if not os.path.isdir(d):
        return None
    trees = read_rst_dir(d)
    if cap is not None:
        trees = trees[:cap]
    return trees


def test_alignment_tiles_and_fixes_drift(tokenizer):
    any_corpus = False
    for name, corpus_dir in CORPORA.items():
        splits = {
            "train": _load(corpus_dir, "train", cap=GUM_TRAIN_CAP if name == "gum" else None),
            "dev": _load(corpus_dir, "dev"),
            "test": _load(corpus_dir, "test"),
        }
        if all(v is None for v in splits.values()):
            print(f"[{name}] absent, skipping")
            continue
        any_corpus = True

        for split, trees in splits.items():
            if trees is None:
                continue
            old_drift = new_drift = 0
            for path, tree in trees:
                edus = tree.edus
                text = _reconstruct_text(edus)

                input_ids, spans = align_edus_to_tokens(tokenizer, text, edus)

                # (1) spans tile range(len(input_ids))
                assert spans[0][0] == 0, path
                for i in range(len(spans) - 1):
                    assert spans[i][1] == spans[i + 1][0], (path, i)
                assert spans[-1][1] == len(input_ids), path
                assert sum(e - s for s, e in spans) == len(input_ids), path

                # (2) char spans recover EDU text
                ends = _char_ends(edus)
                cs = 0
                for i, edu in enumerate(edus):
                    if i > 0:
                        prefix = edu.prefix if edu.prefix is not None else " "
                        cs += len(prefix)
                    ce = cs + len(edu.text)
                    assert text[cs:ce] == edu.text, (path, i)
                    cs = ce
                    assert ce == ends[i]

                # new method never drifts
                if sum(e - s for s, e in spans) != len(input_ids):
                    new_drift += 1
                # (3) old overlap method drift count
                old_total, n_ids = _old_overlap_total(tokenizer, text, edus)
                if old_total != n_ids:
                    old_drift += 1

            print(f"[{name}/{split}] docs={len(trees)} old_drift={old_drift} new_drift={new_drift}")
            assert new_drift == 0

    if not any_corpus:
        pytest.skip("no corpora present")
