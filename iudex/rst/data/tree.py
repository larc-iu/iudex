import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _ddict2dict(v)
    return dict(d)


@dataclass
class RstPpNode:
    id: str
    type: str
    text: Optional[str] = None

    @property
    def is_edu(self):
        return self.type == "terminal"

    def __eq__(self, other):
        return other.id == self.id and other.type == self.type and other.text == self.text


@dataclass
class RstPpEdge:
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


class RstPpTree:
    def __init__(
        self,
        nodes: List[RstPpNode],
        edges: List[RstPpEdge],
        binarize: bool = True,
        relation_types: Tuple[Tuple[str, str], ...] = None,
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

    def relation_of(self, node_id):
        for potential_parent, children in self._primary_adj_map.items():
            for child in children:
                if child == node_id:
                    return self._primary_adj_map[potential_parent][node_id].relation
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
                    edge.relation,
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
                else:
                    edge = [e for e in current_edges if e.relation != "span"][0]
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
                    (edge_yields[1][0] if first_is_left else edge_yields[0][0], "NN", multinuc_edges[0].relation)
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

    def to_rs4_string(self) -> str:
        from lxml import etree as ET
        from lxml.builder import E

        relations = E("relations", *[E("rel", name=name, type=type) for name, type in self._relation_types])
        header = E("header", *[relations])
        body_children = []
        for edu in self.edus:
            body_children.append(
                E("segment", edu.text, id=edu.id, parent=self.parent_of(edu.id), relname=self.relation_of(edu.id))
            )
        for node in self.nonterminals:
            parent = self.parent_of(node.id)
            relname = self.relation_of(node.id)
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
                body_children.append(E("group", id=node.id, type="span"))
        body = E("body", *body_children)
        root = E("rst", *[header, body])
        return ET.tostring(root, encoding="utf-8", pretty_print=True).decode("utf-8")

    @property
    def edus(self) -> List[RstPpNode]:
        return [n for n in self._node_map.values() if n.is_edu]

    @property
    def edu_strings(self) -> List[str]:
        return [edu.text for edu in self.edus]

    @property
    def nonterminals(self) -> List[RstPpNode]:
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
        edus: Union[List[str], List[RstPpNode]],
        return_tree=True,
        relation_types: Tuple[Tuple[str, str], ...] = None,
    ) -> Union["RstPpTree", Tuple[List[RstPpNode], List[RstPpEdge]]]:
        edus = [
            RstPpNode(str(i + 1), "terminal", edu if isinstance(edu, str) else edu.text) for i, edu in enumerate(edus)
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
                top_node = RstPpNode(f"{len(nodes) + 1}", type="span")
                nodes.append(top_node)
                edges.append(RstPpEdge(top_node.id, nucleus_node.id, "span"))
                edges.append(RstPpEdge(nucleus_node.id, satellite_node.id, relation))
            elif nuclearity == "NN":
                top_node = RstPpNode(f"{len(nodes) + 1}", type="multinuc")
                nodes.append(top_node)
                edges.append(RstPpEdge(top_node.id, left_node.id, relation))
                edges.append(RstPpEdge(top_node.id, right_node.id, relation))
            else:
                raise ValueError(f"Unknown nuclearity: {nuclearity}")

            combined_edu_indexes = sorted(
                list({i for i, e in coverage_index.items() if e == left_node or e == right_node})
            )
            for edu_index in combined_edu_indexes:
                coverage_index[edu_index] = top_node
        if return_tree:
            return RstPpTree(nodes, edges, relation_types=relation_types)
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
            new_nodes.append(RstPpNode(id=f"binarized_{new_node_count}", type="multinuc"))
            new_node_count += 1
        nodes.extend(new_nodes)

        latest_parent_id = node_id
        for i in range(1, len(multinuc_edges) - 1):
            next_parent_id = new_nodes[i - 1].id
            edges.append(RstPpEdge(latest_parent_id, next_parent_id, relation))
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
            logger.warning(
                f"Multinuc node at {n.id} has more than two relation types: {edge_labels}."
            )
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
