"""Registry of known RST corpora for model-card rendering.

Keys are directory prefixes — `lookup("data/rstdt/train")` returns the
RST-DT entry. Fields beyond `name` are optional; missing ones are omitted
from the card.
"""

from __future__ import annotations

DATASETS: dict[str, dict] = {
    "data/rstdt": {
        "name": "RST Discourse Treebank (RST-DT)",
        "url": "https://catalog.ldc.upenn.edu/LDC2002T07",
        "language": "English",
        "description": (
            "385 WSJ articles from the Penn Treebank, annotated in RST with a fine-grained relation inventory. "
            "RST-DT is the traditional English benchmark for RST parsing."
        ),
        "citation_text": (
            "Carlson, Lynn, Daniel Marcu, and Mary Ellen Okurovsky. 2001. "
            "Building a Discourse-Tagged Corpus in the Framework of Rhetorical Structure Theory. "
            "In Proceedings of the Second SIGdial Workshop on Discourse and Dialogue."
        ),
        "citation_bibtex": (
            "@inproceedings{carlson-etal-2001-building,\n"
            "    title = {Building a Discourse-Tagged Corpus in the Framework of {R}hetorical "
            "{S}tructure {T}heory},\n"
            "    author = {Carlson, Lynn and Marcu, Daniel and Okurovsky, Mary Ellen},\n"
            "    booktitle = {Proceedings of the Second {SIG}dial Workshop on Discourse and Dialogue},\n"
            "    year = {2001},\n"
            "    url = {https://aclanthology.org/W01-1605/},\n"
            "}"
        ),
    },
    "data/gum_12.1.0": {
        "name": "GUM 12.1.0 (Georgetown University Multilayer corpus)",
        "url": "https://gucorpling.org/gum/",
        "language": "English",
        "description": (
            "A multi-genre English corpus (academic, biography, fiction, interview, news, reddit, "
            "speech, textbook, vlog, voyage, whow) annotated for RST among many other layers. "
        ),
        "citation_text": (
            "Zeldes, Amir, Tatsuya Aoyama, Yang Liu, Siyao Peng, Debopam Das and Luke Gessler. 2025. "
            "eRST: A Signaled Graph Theory of Discourse Relations and Organization. "
            "Computational Linguistics 51(1), 23–72."
        ),
        "citation_bibtex": (
            "@article{zeldes-etal-2025-erst,\n"
            "    title = {e{RST}: A Signaled Graph Theory of Discourse Relations and Organization},\n"
            "    author = {Zeldes, Amir and Aoyama, Tatsuya and Liu, Yang Janet and Peng, Siyao "
            "and Das, Debopam and Gessler, Luke},\n"
            "    journal = {Computational Linguistics},\n"
            "    volume = {51},\n"
            "    number = {1},\n"
            "    year = {2025},\n"
            "    address = {Cambridge, MA},\n"
            "    publisher = {MIT Press},\n"
            "    url = {https://aclanthology.org/2025.cl-1.3/},\n"
            "    doi = {10.1162/coli_a_00538},\n"
            "    pages = {23--72},\n"
            "}"
        ),
    },
}


def lookup(path: str) -> dict | None:
    """Return the dataset entry whose directory-prefix key is a prefix of `path`.

    `path` is typically the value of `cfg.train_dir`. Longest match wins.
    Returns None if nothing matches — callers should fall back to showing the
    raw path.
    """
    if not path:
        return None
    best_key = ""
    for key in DATASETS:
        if path.startswith(key) and len(key) > len(best_key):
            best_key = key
    return DATASETS[best_key] if best_key else None
