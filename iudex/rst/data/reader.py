import os
import xml.etree.ElementTree as ET
from glob import glob
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree

from iudex.rst.data.tree import RstPpEdge, RstPpNode, RstPpTree

logger = getLogger(__name__)


def _extract_one(filepath, elt, name, nullable=False):
    target = elt.findall(name)
    if not nullable and len(target) != 1:
        raise ValueError(f"rs4 file {filepath} does not have exactly one <{name}> element")
    elif nullable and len(target) == 0:
        return None
    else:
        return target[0]


def _drop_keys(d, ks):
    return {k: v for k, v in d.items() if k not in ks}


def _read_rs4_into_dict(filepath: str) -> Dict[str, Any]:
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ET.parse(f, parser=parser)
    document = dict()

    header = _extract_one(filepath, tree, "header")
    relations = _extract_one(filepath, header, "relations")
    document["relation_inventory"] = [r.attrib for r in relations.findall("rel")]

    body = _extract_one(filepath, tree, "body")
    terminals = body.findall("segment")
    document["terminals"] = [{"text": t.text, "type": "terminal", **t.attrib} for t in terminals]
    nonterminals = body.findall("group")
    document["nonterminals"] = [n.attrib for n in nonterminals]
    secedges = _extract_one(filepath, body, "secedges", nullable=True)
    document["secondary_edges"] = [] if secedges is None else [e.attrib for e in secedges.findall("secedge")]
    return document


def _validate_dict(filepath: str, d: Dict[str, Any]) -> None:
    nodes = d["terminals"] + d["nonterminals"]
    ids = [n["id"] for n in nodes]
    if not len(set(ids)) == len(ids):
        raise ValueError(f"Document {filepath} does not have unique IDs for each node")
    for node in nodes:
        if "parent" in node and node["parent"] not in ids:
            raise ValueError(f"Document {filepath} has edge with non-existent parent {node['parent']}")
    roots = [n for n in nodes if "parent" not in n]
    if len(roots) != 1:
        raise ValueError(f"Document {filepath} does not have exactly one root")


def _process_dict(d: Dict[str, Any]) -> Tuple[List[RstPpNode], List[RstPpEdge]]:
    terminals = [RstPpNode(**_drop_keys(n, ["parent", "relname"])) for n in d["terminals"]]
    nonterminals = [RstPpNode(**_drop_keys(n, ["parent", "relname"])) for n in d["nonterminals"]]
    nodes = terminals + nonterminals

    terminal_edges = [
        RstPpEdge(source=n["parent"], target=n["id"], relation=n["relname"])
        for n in d["terminals"]
        if "parent" in n
    ]
    nonterminal_edges = [
        RstPpEdge(source=n["parent"], target=n["id"], relation=n["relname"])
        for n in d["nonterminals"]
        if "parent" in n
    ]
    secondary_edges = [
        RstPpEdge(source=e["source"], target=e["target"], relation=e["relname"], secondary=True)
        for e in d["secondary_edges"]
    ]
    edges = terminal_edges + nonterminal_edges + secondary_edges
    return nodes, edges


def read_rst_file(
    filepath: str,
    binarize: bool = True,
    relation_types: Tuple[Tuple[str, str], ...] = None,
) -> RstPpTree:
    """Read an RS3 or RS4 file and return an RstPpTree."""
    logger.info(f"Reading {filepath}")
    d = _read_rs4_into_dict(filepath)
    _validate_dict(filepath, d)
    nodes, edges = _process_dict(d)
    return RstPpTree(nodes, edges, binarize=binarize, relation_types=relation_types)


def read_rst_dir(
    directory: str,
    binarize: bool = True,
    relation_types: Tuple[Tuple[str, str], ...] = None,
) -> List[Tuple[str, RstPpTree]]:
    """Read all RS3/RS4 files from a directory, returning (filepath, tree) pairs."""
    paths = sorted(glob(str(Path(directory) / "*.rs3")))
    paths += sorted(glob(str(Path(directory) / "*.rs4")))
    results = []
    for p in paths:
        tree = read_rst_file(p, binarize=binarize, relation_types=relation_types)
        results.append((p, tree))
    return results
