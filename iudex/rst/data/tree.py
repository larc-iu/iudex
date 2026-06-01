import copy
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Mapping, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shift-reduce action types
# ---------------------------------------------------------------------------
# Bottom-up RST shift-reduce serializes a binary tree as a flat sequence of
# Shift (push next EDU onto the stack) and Reduce (combine top two stack
# items) actions. These dataclasses are the structured form; serialization
# to the bracketed special-token strings the seq2seq tokenizer wants lives
# in `actions_to_strings` / `strings_to_actions` below.
#
# Why a structured form: the (nuc, rel) -> token-string mapping squashes
# non-alphanumerics, so it's not invertible without a lookup table.
# Round-trips through structured actions are invertible by construction.


@dataclass(frozen=True)
class Shift:
    # Populated by `RstTree.to_shift_reduce(include_text=True)`; left as
    # None for the action-only serialization. Stored as a single space-
    # separated string (RstTree's storage convention), not pre-tokenized.
    edu_text: Optional[str] = None

    def to_token(self) -> str:
        return "<shift>"


@dataclass(frozen=True)
class Reduce:
    nuc: str  # "NS" | "SN" | "NN"
    rel: str  # original (pre-normalization) relation name

    def to_token(self) -> str:
        # Lowercased nuclearity + relation with non-alphanumerics squashed
        # to '_'. Lossy, hence the parser-side `reduce_token_map`.
        return f"<reduce_{self.nuc.lower()}_{re.sub(r'[^a-z0-9]+', '_', self.rel.lower())}>"


ShiftReduceAction = Union[Shift, Reduce]


def actions_to_strings(actions: List[ShiftReduceAction]) -> List[str]:
    """Flatten an action list into the bracketed-token string sequence the
    seq2seq tokenizer consumes. Whitespace-splits any `Shift.edu_text` and
    interleaves the resulting tokens before each `<shift>`. Action-only
    serializations (every `Shift.edu_text is None`) produce a pure action
    sequence."""
    out: List[str] = []
    for a in actions:
        if isinstance(a, Shift):
            if a.edu_text is not None:
                out.extend(a.edu_text.split(" "))
            out.append(a.to_token())
        elif isinstance(a, Reduce):
            out.append(a.to_token())
        else:
            raise TypeError(f"actions_to_strings: unknown action type {type(a).__name__!r}")
    return out


def strings_to_actions(
    tokens: List[str],
    reduce_token_map: Mapping[str, Tuple[str, str]],
) -> List[ShiftReduceAction]:
    """Inverse of `actions_to_strings`. `reduce_token_map` maps known
    `<reduce_*>` token strings back to `(nuc, rel)` pairs (built by the
    seq2seq parser from `cfg.relation_types`). Anything that isn't
    `<shift>` or a key in the map is treated as a surface token belonging
    to the next `Shift` (joined with spaces into its `edu_text`)."""
    out: List[ShiftReduceAction] = []
    pending: List[str] = []
    for t in tokens:
        if t == "<shift>":
            # Always non-None: "" is a valid degenerate EDU (rare; happens
            # when a decoded source subword collapses to whitespace and
            # `.split()` drops it).
            out.append(Shift(edu_text=" ".join(pending)))
            pending = []
        elif t in reduce_token_map:
            nuc, rel = reduce_token_map[t]
            out.append(Reduce(nuc=nuc, rel=rel))
        else:
            pending.append(t)
    if pending:
        raise ValueError(
            f"strings_to_actions: {len(pending)} surface token(s) after the last action; "
            f"the sequence is malformed. Trailing tokens: {pending[:5]!r}..."
        )
    return out


def _ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _ddict2dict(v)
    return dict(d)


_SEXP_LABEL_RE = re.compile(r"^(NS|SN|NN):.+$")


def _is_label(tok: str) -> bool:
    return bool(_SEXP_LABEL_RE.match(tok))


def _tokenize_sexp(text: str) -> List[str]:
    """Split an s-expression into tokens. Parentheses are always their own
    tokens; everything else is whitespace-separated. Inverse of the joining
    done by `RstTree.to_sexp` (plan style)."""
    out: List[str] = []
    buf: List[str] = []
    for ch in text:
        if ch in "()":
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    out.append(tok)
                buf = []
            out.append(ch)
        elif ch.isspace():
            if buf:
                tok = "".join(buf).strip()
                if tok:
                    out.append(tok)
                buf = []
        else:
            buf.append(ch)
    if buf:
        tok = "".join(buf).strip()
        if tok:
            out.append(tok)
    return out


def _join_surface(tokens: List[str]) -> str:
    """Reverse the `(`/`)` escaping done by `to_sexp`'s `escape`. Surface
    tokens are space-joined to form the EDU text."""
    return " ".join(t.replace("-LRB-", "(").replace("-RRB-", ")") for t in tokens)


@dataclass
class RstNode:
    id: str
    type: str
    text: Optional[str] = None
    # TODO(LDG): remove once rs3/rs4 formats are updated
    # Detokenized corpora (e.g. data/gum_12.1.0_notok) record the exact string
    # that precedes a segment's text but follows the previous segment. Absent
    # (None) means the legacy default of a single space; "" means the segments
    # are glued with no separator.
    prefix: Optional[str] = None

    @property
    def is_edu(self):
        return self.type == "terminal"

    def __eq__(self, other):
        return other.id == self.id and other.type == self.type and other.text == self.text


@dataclass
class RstEdge:
    source: str
    target: str
    relation: str
    secondary: bool = False

    @property
    def id(self):
        return f"{self.source}-{self.target}"

    def __eq__(self, other):
        return (
            self.source == other.source
            and self.target == other.target
            and self.relation == other.relation
            and self.secondary == other.secondary
        )


class RstTree:
    def __init__(
        self,
        nodes: List[RstNode],
        edges: List[RstEdge],
        binarize: bool = True,
        relation_types: Tuple[Tuple[str, str], ...] = None,
        relation_map: Optional[dict] = None,
    ):
        if len(set(n.id for n in nodes)) != len(nodes):
            raise ValueError("Nodes must have unique IDs")
        self.is_binary = binarize
        if binarize:
            binarize_tree(nodes, edges)
        assert_well_formed(nodes, edges)

        self._nodes = nodes
        self._edges = edges

        self._node_map = {n.id: n for n in nodes}
        self._edus = [n for n in nodes if n.is_edu]
        self._primary_edge_map = {e.id: e for e in edges if not e.secondary}
        self._secondary_edge_map = {e.id: e for e in edges if e.secondary}
        self._primary_edges = [e for e in edges if not e.secondary]
        self._secondary_edges = [e for e in edges if e.secondary]

        self._primary_adj_map = defaultdict(lambda: defaultdict(lambda: None))
        self._secondary_adj_map = defaultdict(lambda: defaultdict(lambda: None))
        self._build_adj_maps()

        self._relation_types = relation_types
        # Applied at output boundary (parsing_actions, relation_of, etc.) so
        # that binarization / multinuc-vs-satellite disambiguation (which keys
        # off relation-name distinctness) sees the raw labels.
        self._relation_map = relation_map

        has_parent = set()
        for e in edges:
            has_parent.add(e.target)
        root_list = list(set([n.id for n in nodes]).difference(has_parent))
        if len(root_list) != 1:
            raise ValueError("Tree must have exactly one root")
        self.root = self._node_map[root_list[0]]

    def _build_adj_maps(self):
        for e in self._primary_edge_map.values():
            self._primary_adj_map[e.source][e.target] = e
        self._primary_adj_map = _ddict2dict(self._primary_adj_map)
        for e in self._secondary_edge_map.values():
            self._secondary_adj_map[e.source][e.target] = e
        self._secondary_adj_map = _ddict2dict(self._secondary_adj_map)

    def parent_of(self, node_id):
        for potential_parent, children in self._primary_adj_map.items():
            for child in children:
                if child == node_id:
                    return potential_parent
        return None

    def _resolve_rel(self, rel: str) -> str:
        """Apply `relation_map` to a relation name leaving the tree. The
        structural marker `"span"` passes through. Any other label not in the
        map raises (silent passthrough would hide annotation drift).
        """
        if self._relation_map is None or rel == "span":
            return rel
        if rel not in self._relation_map:
            raise KeyError(
                f"Relation {rel!r} is not in this tree's `relation_map`. "
                f"Add it to the map or pass `relation_map=None` to disable mapping."
            )
        return self._relation_map[rel]

    def relation_of(self, node_id):
        for potential_parent, children in self._primary_adj_map.items():
            for child in children:
                if child == node_id:
                    return self._resolve_rel(self._primary_adj_map[potential_parent][node_id].relation)
        return None

    def parsing_actions(self, dfs: bool = True) -> List[Tuple[int, str, str]]:
        if not self.is_binary:
            raise NotImplementedError("Non-binary trees are not currently supported for this operation.")

        edu_yields = {}

        def compute_edu_yields(current):
            nonlocal edu_yields
            current_node = self._node_map[current]
            current_yield = [self._edus.index(current_node)] if current_node.is_edu else []
            if current not in self._primary_adj_map:
                edu_yields[current] = current_yield
                return current_yield
            children = self._primary_adj_map[current].keys()
            for child in children:
                current_yield.extend(compute_edu_yields(child))
            current_yield = sorted(current_yield)
            edu_yields[current] = current_yield
            return current_yield

        compute_edu_yields(self.root.id)

        edge_index = defaultdict(list)
        for e in self._primary_edge_map.values():
            if not e.secondary:
                edge_index[e.source].append(e)

        sequence = []
        queue = [self.root.id]

        def handle_satellite(sequence, edge, edu_yield):
            satellite_edu_yield = edu_yields[edge.target]
            satellite_is_left = all(x < max(edu_yield) for x in satellite_edu_yield)
            sequence.append(
                (
                    satellite_edu_yield[-1] + 1 if satellite_is_left else satellite_edu_yield[0],
                    "SN" if satellite_is_left else "NS",
                    self._resolve_rel(edge.relation),
                )
            )

        while len(queue) > 0:
            current = self._node_map[queue.pop(-1 if dfs else 0)]
            children = [e for e in self._primary_adj_map.get(current.id, {}).keys()]
            children = sorted(children, key=lambda e: min(edu_yields[e]))
            queue.extend(children)
            current_edges = edge_index[current.id]
            edu_yield = edu_yields[current.id]
            if current.type == "span":
                if len(current_edges) == 1:
                    continue
                # 1 nucleus (relname='span') + N satellites with distinct relations
                # is valid RST (rare in GUM, present in RST-DT). Emit one satellite
                # per action, outermost-first, so the `spans()` `bounds` accumulator
                # narrows each successive emit's enclosing range and the model sees
                # a distinct gold split for each.
                satellite_edges = [e for e in current_edges if e.relation != "span"]
                if len(satellite_edges) > 1:
                    nucleus_edge = next(e for e in current_edges if e.relation == "span")
                    nucleus_yield = edu_yields[nucleus_edge.target]

                    def dist_to_nucleus(edge):
                        y = edu_yields[edge.target]
                        if y[-1] < nucleus_yield[0]:
                            return nucleus_yield[0] - y[-1]
                        if y[0] > nucleus_yield[-1]:
                            return y[0] - nucleus_yield[-1]
                        return 0

                    satellite_edges = sorted(satellite_edges, key=dist_to_nucleus, reverse=True)
                for edge in satellite_edges:
                    handle_satellite(sequence, edge, edu_yield)
            elif current.type == "terminal":
                if len(current_edges) == 0:
                    continue
                edge = current_edges[0]
                handle_satellite(sequence, edge, edu_yield)
            else:
                satellite_relation = None
                if len(current_edges) == 3:
                    satellite_relation = Counter([e.relation for e in current_edges]).most_common(2)[1][0]
                    edge = [e for e in current_edges if e.relation == satellite_relation][0]
                    handle_satellite(sequence, edge, edu_yield)
                multinuc_edges = [e for e in current_edges if e.relation != satellite_relation]
                edge_yields = [edu_yields[e.target] for e in multinuc_edges]
                first_is_left = all(edge_yields[0][0] < x for x in edge_yields[1])
                sequence.append(
                    (
                        edge_yields[1][0] if first_is_left else edge_yields[0][0],
                        "NN",
                        self._resolve_rel(multinuc_edges[0].relation),
                    )
                )
        return sequence

    def spans(self) -> List[Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], str, str]]:
        spans = []
        edus = list(range(len(self.edus)))
        bounds = {0, len(edus)}

        def find_bound(i, left=False):
            while i not in bounds:
                i = i - 1 if left else i + 1
            return i

        for split, nuclearity, relation in self.parsing_actions():
            left, right = edus[find_bound(split, left=True) : split], edus[split : find_bound(split)]
            bounds.add(split)
            spans.append(((tuple(left), tuple(right)), nuclearity, relation))

        return spans

    def spans_with_ranges(self) -> List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], str, str]]:
        output = []
        for (left, right), nuclearity, relation in self.spans():
            output.append((((left[0], left[-1] + 1), (right[0], right[-1] + 1)), nuclearity, relation))
        return output

    def debinarize(self) -> "RstTree":
        """Return a new tree with binary multinuclear chains flattened to n-ary.

        A multinuc node nested directly under another multinuc, where the
        attaching edge and all of the child's own edges share one relation, is a
        binarization artifact (parsers can only emit binary splits, so an n-ary
        multinuc comes out as nested same-relation multinucs). Such a child is
        absorbed into its parent, recursively and regardless of branch shape
        (right/left/balanced). Mononuclear structure and genuinely nested
        *different*-relation multinucs (e.g. `List[A, Joint[B, C]]`) are left
        intact, since the relation guard only fires when labels match through.

        Non-mutating: `self` stays binary. The result is non-binary and so is
        serialization-only. Do not call `spans()`/`parsing_actions()` on it.
        """
        nodes = [copy.copy(n) for n in self._nodes]
        edges = [copy.copy(e) for e in self._edges]
        node_map = {n.id: n for n in nodes}

        def primary_out(node_id):
            return [e for e in edges if not e.secondary and e.source == node_id]

        merged = True
        while merged:
            merged = False
            for e in [e for e in edges if not e.secondary]:
                parent, child = node_map.get(e.source), node_map.get(e.target)
                if parent is None or child is None:
                    continue
                if parent.type != "multinuc" or child.type != "multinuc":
                    continue
                # Raw `e.relation` (relation_map is applied later, in relation_of):
                # merging on mapped labels could fuse two distinct fine relations.
                grandchild_edges = primary_out(child.id)
                if not grandchild_edges or any(ge.relation != e.relation for ge in grandchild_edges):
                    continue
                for ge in grandchild_edges:
                    ge.source = parent.id
                edges.remove(e)
                nodes.remove(child)
                del node_map[child.id]
                merged = True
                break

        return RstTree(
            nodes,
            edges,
            binarize=False,
            relation_types=self._relation_types,
            relation_map=self._relation_map,
        )

    def to_rs4_string(self) -> str:
        from lxml import etree as ET
        from lxml.builder import E

        tree = self.debinarize()
        relations = E("relations", *[E("rel", name=name, type=type) for name, type in tree._relation_types])
        header = E("header", *[relations])
        body_children = []
        for edu in tree.edus:
            parent = tree.parent_of(edu.id)
            relname = tree.relation_of(edu.id)
            # Omit parent/relname when None (single-EDU case, where the EDU is
            # itself the root): lxml's ElementMaker rejects None-valued attrs.
            if parent is not None:
                body_children.append(E("segment", edu.text, id=edu.id, parent=parent, relname=relname))
            else:
                body_children.append(E("segment", edu.text, id=edu.id))
        for node in tree.nonterminals:
            parent = tree.parent_of(node.id)
            relname = tree.relation_of(node.id)
            if parent is not None:
                body_children.append(
                    E(
                        "group",
                        id=node.id,
                        type=("multinuc" if node.type == "multinuc" else "span"),
                        parent=parent,
                        relname=relname,
                    )
                )
            else:
                body_children.append(E("group", id=node.id, type=("multinuc" if node.type == "multinuc" else "span")))
        body = E("body", *body_children)
        root = E("rst", *[header, body])
        return ET.tostring(root, encoding="utf-8", pretty_print=True).decode("utf-8")

    def _build_binary_tree(self):
        """Replay `parsing_actions` into a nested-tuple binary tree:
        leaves are `("edu", edu_text)`; internal nodes are
        `("node", nuclearity, relation, left, right)`, with children in text
        order. Multi-satellite span nodes (1 nucleus, N>1 satellites) come out
        as N nested binary splits, matching how `parsing_actions` decomposes
        them. Relations are already passed through `_resolve_rel`.
        """
        actions = self.parsing_actions()
        coverage: list = [("edu", t) for t in self.edu_strings]
        for right_index, nuc, rel in reversed(actions):
            left = coverage[right_index - 1]
            right = coverage[right_index]
            merged = ("node", nuc, rel, left, right)
            for i in range(len(coverage)):
                if coverage[i] is left or coverage[i] is right:
                    coverage[i] = merged
        return coverage[0]

    def to_sexp(
        self,
        traversal_order: str = "preorder",
        include_text: bool = True,
        format: Optional[str] = None,
        invert_escaping: bool = False,
    ) -> str:
        """Serialize as an S-expression. Internal nodes carry a `NUC:relation`
        label with NUC in {NS, SN, NN}. Two output styles:

        * Plan style (the default, when `format` is None). Used by the
          `seq2seq_sexp` / `decoder_only_sexp` parsers. Leaves render as
          `(<edu surface tokens>)` when `include_text=True` (parens inside the
          surface text are escaped to `-LRB-`/`-RRB-`), or as the bare token
          `<edu>` when `include_text=False` (callers supply EDU surface forms
          out-of-band, positionally in document order). Internal nodes render
          as `(NS:elaboration child1 child2)` for `traversal_order='preorder'`
          or `(child1 child2 NS:elaboration)` for `traversal_order='postorder'`.

        * Legacy style (when `format` is set). `format` selects between:
          - "iudex": `(EDU text)` leaves with `-LRB-`/`-RRB-` escaping.
          - "dis": `(text _!...!_)` RST-DT `.dis` fence (no escaping needed).
          - "index": bare 0-based EDU indices in place of leaves.
          The legacy modes ignore `traversal_order` and `include_text` (always
          pre-order, always text-in). `invert_escaping=True` swaps structural
          `(`/`)` for `-LRB-`/`-RRB-` (and disables leaf-text escaping).

        Round-trip with `from_sexp` is guaranteed for the plan style only.
        """
        if not self.is_binary:
            raise NotImplementedError("Non-binary trees are not currently supported for this operation.")
        if traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"Unknown traversal_order {traversal_order!r}; expected 'preorder' or 'postorder'.")
        if format is not None and format not in ("iudex", "dis", "index"):
            raise ValueError(f"Unknown format {format!r}; expected one of: iudex, dis, index, or None.")

        if invert_escaping:
            lpar, rpar = "-LRB-", "-RRB-"

            def escape(s: str) -> str:
                return s
        else:
            lpar, rpar = "(", ")"

            def escape(s: str) -> str:
                return s.replace("(", "-LRB-").replace(")", "-RRB-")

        if format is not None:
            counter = 0

            def render_legacy(node) -> str:
                nonlocal counter
                if node[0] == "edu":
                    text = node[1]
                    if format == "iudex":
                        return f"{lpar}EDU {escape(text)}{rpar}"
                    if format == "dis":
                        return f"{lpar}text _!{text}_!{rpar}"
                    idx = counter
                    counter += 1
                    return str(idx)
                _, nuc, rel, left, right = node
                return f"{lpar}{nuc}:{rel} {render_legacy(left)} {render_legacy(right)}{rpar}"

            return render_legacy(self._build_binary_tree())

        def render(node) -> str:
            if node[0] == "edu":
                if not include_text:
                    return "<edu>"
                text = escape(node[1])
                return f"{lpar}{text}{rpar}" if text else f"{lpar}{rpar}"
            _, nuc, rel, left, right = node
            label = f"{nuc}:{rel}"
            if traversal_order == "preorder":
                return f"{lpar}{label} {render(left)} {render(right)}{rpar}"
            return f"{lpar}{render(left)} {render(right)} {label}{rpar}"

        return render(self._build_binary_tree())

    @classmethod
    def from_sexp(
        cls,
        text: str,
        traversal_order: str = "preorder",
        edus: Optional[List[str]] = None,
        relation_types: Optional[Tuple[Tuple[str, str], ...]] = None,
    ) -> "RstTree":
        """Inverse of `to_sexp` (plan style only). `text` must be the output
        of `to_sexp(traversal_order=..., include_text=...)`. If the source had
        `include_text=False`, pass `edus` (positional list of surface strings,
        document order) and each `<edu>` placeholder is filled in.

        Leaf vs internal disambiguation: a span is internal iff its first
        (`preorder`) or last (`postorder`) token is a `NUC:relation` label. EDU
        surface text never starts/ends with such a token in practice (no
        `NS:`/`SN:`/`NN:` prefix), so the format is unambiguous.
        """
        if traversal_order not in ("preorder", "postorder"):
            raise ValueError(f"Unknown traversal_order {traversal_order!r}; expected 'preorder' or 'postorder'.")

        tokens = _tokenize_sexp(text)
        pos = [0]

        def parse_node() -> dict:
            # Dispatches on the next token: `<edu>` is a placeholder leaf,
            # `(` opens a span (leaf-with-text OR internal). Bare tokens are
            # not legal as standalone children of an internal node.
            if pos[0] >= len(tokens):
                raise ValueError("from_sexp: unexpected EOF where a child was expected.")
            tok = tokens[pos[0]]
            if tok == "<edu>":
                pos[0] += 1
                return {"kind": "leaf_placeholder"}
            if tok != "(":
                raise ValueError(f"from_sexp: expected '(' or '<edu>', got {tok!r}")
            return parse_paren()

        def parse_paren() -> dict:
            pos[0] += 1  # consume '('
            # Decide leaf-with-text vs internal by peeking. Pre-order: a label
            # right after '(' marks an internal node. Post-order: a label right
            # before the matching ')' marks an internal node. Otherwise the
            # span is a leaf containing only surface tokens.
            depth_start = pos[0]
            if traversal_order == "preorder" and pos[0] < len(tokens) and _is_label(tokens[pos[0]]):
                label = tokens[pos[0]]
                pos[0] += 1
                left = parse_node()
                right = parse_node()
                if pos[0] >= len(tokens) or tokens[pos[0]] != ")":
                    raise ValueError(f"from_sexp: expected ')' after internal node {label!r}.")
                pos[0] += 1
                return {"kind": "internal", "label": label, "left": left, "right": right}
            if traversal_order == "postorder":
                # Find the matching ')' by tracking depth; check the token
                # immediately before it.
                depth = 1
                scan = pos[0]
                while scan < len(tokens) and depth > 0:
                    if tokens[scan] == "(":
                        depth += 1
                    elif tokens[scan] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    scan += 1
                if scan >= len(tokens):
                    raise ValueError("from_sexp: unbalanced parens (postorder scan).")
                last_inner = tokens[scan - 1] if scan - 1 >= depth_start else None
                if last_inner is not None and _is_label(last_inner):
                    left = parse_node()
                    right = parse_node()
                    if pos[0] >= len(tokens) or not _is_label(tokens[pos[0]]):
                        raise ValueError("from_sexp: expected label before ')' in postorder internal node.")
                    label = tokens[pos[0]]
                    pos[0] += 1
                    if pos[0] >= len(tokens) or tokens[pos[0]] != ")":
                        raise ValueError(f"from_sexp: expected ')' after postorder label {label!r}.")
                    pos[0] += 1
                    return {"kind": "internal", "label": label, "left": left, "right": right}
            # Leaf with literal text. Consume surface tokens up to ')'.
            surfaces: List[str] = []
            while pos[0] < len(tokens) and tokens[pos[0]] != ")":
                if tokens[pos[0]] == "(":
                    raise ValueError("from_sexp: nested '(' inside a leaf span (malformed).")
                surfaces.append(tokens[pos[0]])
                pos[0] += 1
            if pos[0] >= len(tokens):
                raise ValueError("from_sexp: unbalanced parens (leaf scan).")
            pos[0] += 1  # consume ')'
            return {"kind": "leaf_text", "surfaces": surfaces}

        parsed = parse_node()
        if pos[0] != len(tokens):
            raise ValueError(f"from_sexp: trailing tokens after root: {tokens[pos[0] :][:5]!r}...")

        edu_texts: List[Optional[str]] = []
        actions: List[Tuple[int, str, str]] = []

        def collect(node) -> Tuple[int, int]:
            kind = node["kind"]
            if kind == "leaf_placeholder":
                edu_texts.append(None)
                idx = len(edu_texts) - 1
                return idx, idx + 1
            if kind == "leaf_text":
                edu_texts.append(_join_surface(node["surfaces"]))
                idx = len(edu_texts) - 1
                return idx, idx + 1
            lo_l, hi_l = collect(node["left"])
            lo_r, hi_r = collect(node["right"])
            if hi_l != lo_r:
                raise ValueError(f"from_sexp: discontinuous children {hi_l} vs {lo_r}")
            nuc, _, rel = node["label"].partition(":")
            if nuc not in ("NS", "SN", "NN") or not rel:
                raise ValueError(f"from_sexp: malformed label {node['label']!r}")
            actions.append((lo_r, nuc, rel))
            return lo_l, hi_r

        collect(parsed)
        # `parsing_actions` is depth-first (pre-order over the split index);
        # `from_parsing_actions` reverses it internally. Either DFS pre or post
        # works as long as `right_index` is consistent; we built it in
        # post-order (children before parent), so reverse to pre-order for
        # symmetry with `parsing_actions`.
        actions.reverse()

        # Fill in EDU surface forms.
        if edus is not None:
            if len(edus) != len(edu_texts):
                raise ValueError(
                    f"from_sexp: `edus` has {len(edus)} entries but the s-expression contains {len(edu_texts)} leaves."
                )
            final_edus: List[str] = []
            for i, (t, e) in enumerate(zip(edu_texts, edus)):
                if t is not None and t != e:
                    raise ValueError(
                        f"from_sexp: EDU {i} disagrees between in-stream text {t!r} and `edus[{i}]={e!r}`."
                    )
                final_edus.append(e)
        else:
            if any(t is None for t in edu_texts):
                raise ValueError(
                    "from_sexp: s-expression has `<edu>` placeholders but no `edus` argument was supplied."
                )
            final_edus = [t for t in edu_texts]  # type: ignore[misc]

        if not actions:
            # Single-EDU degenerate tree.
            edu_nodes = [RstNode("1", "terminal", final_edus[0])]
            return cls(edu_nodes, [], relation_types=relation_types)
        return cls.from_parsing_actions(actions, final_edus, return_tree=True, relation_types=relation_types)

    def to_shift_reduce(
        self,
        ltr: bool = True,
        include_text: bool = False,
    ) -> List[ShiftReduceAction]:
        """Serialize as a chain of shift/reduce actions. Each action is a
        `Shift()` (shift the next EDU onto the stack) or a `Reduce(nuc,
        rel)` (combine the top two stack items, with `nuc` in {NS, SN,
        NN}). With `ltr=True` (default) EDUs are shifted in text order;
        with `ltr=False` they are shifted right-to-left. Nuclearity labels
        always describe text order, so a right-to-left consumer must swap
        children when applying each reduce.

        With `include_text=True`, each `Shift` carries the surface text of
        the EDU it commits in its `edu_text` field; this is the form the
        seq2seq parser feeds to its tokenizer. Hard-requires `ltr=True`
        (right-to-left text emission is undefined).
        """
        if not self.is_binary:
            raise NotImplementedError("Non-binary trees are not currently supported for this operation.")
        if include_text and not ltr:
            raise ValueError("include_text=True requires ltr=True (RTL text emission is undefined).")

        output: List[ShiftReduceAction] = []

        def walk(node) -> None:
            if node[0] == "edu":
                output.append(Shift(edu_text=node[1] if include_text else None))
                return
            _, nuc, rel, left, right = node
            for c in [left, right] if ltr else [right, left]:
                walk(c)
            output.append(Reduce(nuc=nuc, rel=rel))

        walk(self._build_binary_tree())
        return output

    @classmethod
    def from_shift_reduce(
        cls,
        actions: List[ShiftReduceAction],
        edus: Optional[List[str]] = None,
        relation_types: Optional[Tuple[Tuple[str, str], ...]] = None,
    ) -> "RstTree":
        """Inverse of `to_shift_reduce`. Walks the action sequence with a
        node stack, building `RstNode`s and `RstEdge`s as each `Reduce`
        fires; structurally mirrors `from_parsing_actions` but consumes
        the bottom-up SR form rather than the post-order `(split_index,
        nuc, rel)` form.

        EDU surface text comes from each `Shift.edu_text` if all are
        populated; otherwise `edus` (positional list of strings) must be
        supplied. Disagreement between the two sources raises.
        """
        shifts = [a for a in actions if isinstance(a, Shift)]
        shift_texts = [s.edu_text for s in shifts]
        if edus is None:
            if any(t is None for t in shift_texts):
                raise ValueError(
                    "from_shift_reduce: actions contain Shifts without edu_text; "
                    "either populate Shift.edu_text or pass `edus` explicitly."
                )
            edu_strings: List[str] = list(shift_texts)
        else:
            if len(edus) != len(shifts):
                raise ValueError(
                    f"from_shift_reduce: `edus` has {len(edus)} entries but actions have {len(shifts)} Shifts."
                )
            for i, (e, t) in enumerate(zip(edus, shift_texts)):
                if t is not None and t != e:
                    raise ValueError(
                        f"from_shift_reduce: EDU {i} disagrees between explicit `edus[{i}]={e!r}` "
                        f"and `Shift.edu_text={t!r}`."
                    )
            edu_strings = list(edus)

        # Pre-allocate EDU nodes so their IDs (1..n) can't collide with
        # internal-node IDs (n+1, n+2, ...) generated as Reduces fire.
        edu_nodes = [RstNode(str(i + 1), "terminal", text) for i, text in enumerate(edu_strings)]
        nodes: List[RstNode] = list(edu_nodes)
        edges: List[RstEdge] = []
        stack: List[RstNode] = []
        edu_counter = 0

        for step, action in enumerate(actions):
            if isinstance(action, Shift):
                stack.append(edu_nodes[edu_counter])
                edu_counter += 1
            elif isinstance(action, Reduce):
                if len(stack) < 2:
                    raise ValueError(
                        f"from_shift_reduce: Reduce at step {step} needs ≥2 stack items, have {len(stack)}."
                    )
                right_node = stack.pop()
                left_node = stack.pop()
                if action.nuc in ("NS", "SN"):
                    if action.nuc == "SN":
                        nucleus, satellite = right_node, left_node
                    else:
                        nucleus, satellite = left_node, right_node
                    top = RstNode(f"{len(nodes) + 1}", type="span")
                    nodes.append(top)
                    edges.append(RstEdge(top.id, nucleus.id, "span"))
                    edges.append(RstEdge(nucleus.id, satellite.id, action.rel))
                elif action.nuc == "NN":
                    top = RstNode(f"{len(nodes) + 1}", type="multinuc")
                    nodes.append(top)
                    edges.append(RstEdge(top.id, left_node.id, action.rel))
                    edges.append(RstEdge(top.id, right_node.id, action.rel))
                else:
                    raise ValueError(f"from_shift_reduce: unknown nuclearity {action.nuc!r}")
                stack.append(top)
            else:
                raise TypeError(f"from_shift_reduce: unknown action type {type(action).__name__!r}")

        if len(stack) != 1:
            raise ValueError(f"from_shift_reduce: action sequence ended with {len(stack)} items on stack, expected 1.")
        if edu_counter != len(edu_strings):
            raise ValueError(f"from_shift_reduce: consumed {edu_counter} EDUs but {len(edu_strings)} were available.")
        return cls(nodes, edges, relation_types=relation_types)

    def subtrees_up_to(self, max_edus: int, *, min_edus: int = 2) -> List["RstTree"]:
        """Maximal-subtree partition of this tree under an EDU-count cap, for
        size-bucketed curriculum training. Walk the binary tree top-down and
        emit a node as a standalone `RstTree` as soon as its EDU leaf count is
        `<= max_edus` (then stop recursing into it), otherwise recurse into both
        children. Nodes with fewer than `min_edus` leaves are skipped (a 1-EDU
        "tree" carries no split to learn), so the emitted EDU windows tile
        `[0, len(edus))` except for those dropped singletons (partition-with-holes).

        A whole tree with `<= max_edus` leaves returns one subtree structurally
        equal to `self`. Emitted subtrees carry already-`_resolve_rel`-mapped
        relation names (they come from `_build_binary_tree`), so `relation_map`
        is not re-applied (only `_relation_types` is propagated). EDU `prefix`
        info is not preserved (the rebuilt EDUs hold text only), so callers that
        reconstruct text from a subtree get space-joined EDUs.
        """
        if not self.is_binary:
            raise NotImplementedError("Non-binary trees are not currently supported for this operation.")

        def leaf_count(node) -> int:
            return 1 if node[0] == "edu" else leaf_count(node[3]) + leaf_count(node[4])

        def to_actions(node) -> List[ShiftReduceAction]:
            out: List[ShiftReduceAction] = []

            def walk(n):
                if n[0] == "edu":
                    out.append(Shift(edu_text=n[1]))
                    return
                _, nuc, rel, left, right = n
                walk(left)
                walk(right)
                out.append(Reduce(nuc=nuc, rel=rel))

            walk(node)
            return out

        subtrees: List["RstTree"] = []

        def select(node) -> None:
            n_leaves = leaf_count(node)
            if n_leaves <= max_edus:
                if n_leaves >= min_edus:
                    subtrees.append(RstTree.from_shift_reduce(to_actions(node), relation_types=self._relation_types))
                return
            select(node[3])
            select(node[4])

        select(self._build_binary_tree())
        return subtrees

    @property
    def edus(self) -> List[RstNode]:
        return [n for n in self._node_map.values() if n.is_edu]

    @property
    def edu_strings(self) -> List[str]:
        return [edu.text for edu in self.edus]

    # TODO(LDG): remove once rs3/rs4 formats are updated
    @property
    def edu_prefixes(self) -> List[Optional[str]]:
        """Per-EDU `prefix` markers, in document order, aligned with `edu_strings`.
        `None` means the legacy single-space default; "" means glued. Only
        detokenized corpora populate these (see scripts/build_gum_notok.py)."""
        return [edu.prefix for edu in self.edus]

    @property
    def nonterminals(self) -> List[RstNode]:
        return [n for n in self._node_map.values() if not n.is_edu]

    @property
    def tokens(self) -> List[str]:
        tokens = []
        for edu_string in self.edu_strings:
            tokens += edu_string.split(" ")
        return tokens

    @classmethod
    def from_parsing_actions(
        cls,
        actions: List[Tuple[int, str, str]],
        edus: Union[List[str], List[RstNode]],
        return_tree=True,
        relation_types: Tuple[Tuple[str, str], ...] = None,
    ) -> Union["RstTree", Tuple[List[RstNode], List[RstEdge]]]:
        edus = [
            RstNode(str(i + 1), "terminal", edu if isinstance(edu, str) else edu.text) for i, edu in enumerate(edus)
        ]

        nodes, edges = [], []
        nodes.extend(edus)
        coverage_index = {i: edus[i] for i in range(len(edus))}
        for right_index, nuclearity, relation in reversed(actions):
            left_index = right_index - 1
            left_node = coverage_index[left_index]
            right_node = coverage_index[right_index]

            if nuclearity in ["NS", "SN"]:
                satellite_node = left_node if nuclearity == "SN" else right_node
                nucleus_node = right_node if nuclearity == "SN" else left_node
                top_node = RstNode(f"{len(nodes) + 1}", type="span")
                nodes.append(top_node)
                edges.append(RstEdge(top_node.id, nucleus_node.id, "span"))
                edges.append(RstEdge(nucleus_node.id, satellite_node.id, relation))
            elif nuclearity == "NN":
                top_node = RstNode(f"{len(nodes) + 1}", type="multinuc")
                nodes.append(top_node)
                edges.append(RstEdge(top_node.id, left_node.id, relation))
                edges.append(RstEdge(top_node.id, right_node.id, relation))
            else:
                raise ValueError(f"Unknown nuclearity: {nuclearity}")

            combined_edu_indexes = sorted(
                list({i for i, e in coverage_index.items() if e == left_node or e == right_node})
            )
            for edu_index in combined_edu_indexes:
                coverage_index[edu_index] = top_node
        if return_tree:
            return RstTree(nodes, edges, relation_types=relation_types)
        else:
            return nodes, edges

    def __eq__(self, other):
        return hasattr(other, "spans") and set(other.spans()) == set(self.spans())


def assert_well_formed(nodes, edges):
    node_ids = {n.id for n in nodes}
    for e in edges:
        if e.source not in node_ids:
            raise ValueError(f"Edge {e} refers to non-existent source")
        if e.target not in node_ids:
            raise ValueError(f"Edge {e} refers to non-existent target")


def get_root(nodes, edges):
    has_parent = set()
    for e in edges:
        has_parent.add(e.target)
    root_list = list(set([n.id for n in nodes]).difference(has_parent))
    if len(root_list) != 1:
        raise ValueError(f"Expected exactly one root, but found roots: {root_list}")
    return root_list[0]


def binarize_tree(nodes, edges):
    edge_index = defaultdict(list)
    for e in [e for e in edges if not e.secondary]:
        edge_index[e.source].append(e)

    new_node_count = 0

    def binarize_node(node_id, multinuc_edges, relation):
        nonlocal new_node_count, edges, nodes
        new_nodes = []
        for i in range(len(multinuc_edges) - 2):
            new_nodes.append(RstNode(id=f"binarized_{new_node_count}", type="multinuc"))
            new_node_count += 1
        nodes.extend(new_nodes)

        latest_parent_id = node_id
        for i in range(1, len(multinuc_edges) - 1):
            next_parent_id = new_nodes[i - 1].id
            edges.append(RstEdge(latest_parent_id, next_parent_id, relation))
            multinuc_edges[i].source = next_parent_id
            latest_parent_id = next_parent_id
        multinuc_edges[-1].source = latest_parent_id

    edu_yields = {}
    node_map = {n.id: n for n in nodes}
    edus = [n for n in nodes if n.is_edu]
    primary_adj_map = defaultdict(lambda: defaultdict(lambda: None))
    for e in edges:
        if not e.secondary:
            primary_adj_map[e.source][e.target] = e
    primary_adj_map = _ddict2dict(primary_adj_map)

    def compute_edu_yields(current):
        nonlocal edu_yields
        current_node = node_map[current]
        current_yield = [edus.index(current_node)] if current_node.is_edu else []
        if current not in primary_adj_map:
            edu_yields[current] = current_yield
            return current_yield
        children = primary_adj_map[current].keys()
        for child in children:
            current_yield.extend(compute_edu_yields(child))
        current_yield = sorted(current_yield)
        edu_yields[current] = current_yield
        return current_yield

    compute_edu_yields(get_root(nodes, edges))

    for n in nodes.copy():
        if n.type != "multinuc":
            continue
        child_edges = edge_index[n.id]
        edge_labels = sorted(list(Counter([e.relation for e in child_edges]).items()), key=lambda x: -x[1])
        if len(edge_labels) > 2:
            logger.warning(f"Multinuc node at {n.id} has more than two relation types: {edge_labels}.")
        if len(edge_labels) == 2 and edge_labels[1][1] != 1:
            logger.warning(
                f"Multinuc node at {n.id} has two relation types, of which one does not occur exactly once: {edge_labels}."
            )
        if edge_labels[0][1] == 1:
            logger.warning(f"Multinuc node at {n.id} appears to have only one nucleus: {edge_labels}.")
        multinuc_relation_type, multinuc_edge_count = edge_labels[0]
        if multinuc_edge_count == 2:
            continue
        edges_to_binarize = [e for e in child_edges if e.relation == multinuc_relation_type]
        edges_to_binarize = sorted(edges_to_binarize, key=lambda e: min(edu_yields[e.target]))
        binarize_node(n.id, edges_to_binarize, multinuc_relation_type)
