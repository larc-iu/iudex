"""Training curriculum strategies for the RST parsers.

A `Curriculum` decides, per training phase, three things: the training trees to
fit on, the dev set to validate against (empty => the loop skips validation for
that phase), and the phase's epoch budget. The default `SimpleCurriculum` is a
no-op (one phase, full documents, full dev) that reproduces pre-curriculum
behavior exactly. `SubtreeSizeCurriculum` warms the model up on small subtrees
before full documents, to avoid the cold-start degeneracy the generative
s-expression parsers hit on long documents (see SEXP_CURRICULUM_FINDINGS.md,
after Hu & Wan 2023).

`Curriculum` is a `tonga.Registrable` so the strategy is config-selectable
(`curriculum: {type: ..., ...}`), the same pattern as `Detokenizer`.
`default_implementation` is set so the config round-trips through
`dataclasses.asdict` (which drops the `type` discriminator) and back via
`from_params`.
"""

from dataclasses import dataclass, field

from tonga import Registrable

from iudex.rst.data.tree import RstTree


@dataclass
class Phase:
    cap: int | None  # EDU-count cap for subtrees in this phase (None = full documents)
    epochs: int


class Curriculum(Registrable):
    default_implementation = "simple"

    def plan(self) -> list[Phase]:
        """The ordered phases for the run. Each `Phase` carries its own epoch
        budget, so the curriculum fully owns run length (there is no top-level
        `max_epochs`). Total run length is `sum(p.epochs for p in plan())`."""
        raise NotImplementedError

    def train_trees(self, all_trees: list[RstTree], phase: Phase) -> list[RstTree]:
        """Training trees for `phase` (a subset/transform of the full set)."""
        raise NotImplementedError

    def dev_pairs(self, all_dev: list[tuple[str, RstTree]], phase: Phase) -> list[tuple[str, RstTree]]:
        """Dev `(name, tree)` pairs to validate on during `phase`. An empty list
        means the loop skips validation, best-model updates, and patience for the
        whole phase (early phases on full-document dev are near-useless)."""
        raise NotImplementedError


@Curriculum.register("simple")
@dataclass
class SimpleCurriculum(Curriculum):
    """No curriculum: one phase of full documents for `epochs`, with full dev
    validated every epoch. The pre-curriculum behavior, with the run length set
    by `epochs` (which replaced the old top-level `max_epochs`)."""

    # `type` is a real field (not just the Registrable discriminator) so it
    # survives `dataclasses.asdict` and the config round-trips with >1 impl
    # registered (tonga still pops it for dispatch before constructing).
    type: str = "simple"
    # Run length. Real configs set this; the default is a short generic fallback
    # (parsers used to default max_epochs to 10-50, always overridden in practice).
    epochs: int = 10

    def plan(self) -> list[Phase]:
        return [Phase(cap=None, epochs=self.epochs)]

    def train_trees(self, all_trees: list[RstTree], phase: Phase) -> list[RstTree]:
        return all_trees

    def dev_pairs(self, all_dev, phase):
        return all_dev


@Curriculum.register("subtree_size")
@dataclass
class SubtreeSizeCurriculum(Curriculum):
    """Size-bucketed-subtree ramp. Each phase trains on the maximal subtrees of
    every document whose EDU span is `<= size_schedule[i]` (a `null`/None cap
    means full documents) for `phase_epochs[i]` epochs, then advances to the next
    (larger) cap. The final phase should be full documents (a `null` cap last).
    Validation runs only in full-document phases (subtree-phase dev is empty).

    `phase_epochs` sets the run length (total = its sum). Each phase runs its full
    `phase_epochs[i]` (no early stop) until a validating phase is reached.
    """

    # See SimpleCurriculum.type: a real field so asdict keeps the discriminator.
    type: str = "subtree_size"
    size_schedule: list[int | None] = field(default_factory=lambda: [8, 20, 60, None])
    phase_epochs: int | list[int] = 5

    def __post_init__(self):
        if not self.size_schedule:
            raise ValueError("curriculum.size_schedule must be non-empty")
        if any(c is not None and c < 2 for c in self.size_schedule):
            raise ValueError("curriculum.size_schedule caps must be >= 2 or null (full documents)")
        caps = [float("inf") if c is None else c for c in self.size_schedule]
        if caps != sorted(caps):
            raise ValueError("curriculum.size_schedule must be ascending with the null (full-doc) cap last")
        if isinstance(self.phase_epochs, list) and len(self.phase_epochs) != len(self.size_schedule):
            raise ValueError(
                f"curriculum.phase_epochs has {len(self.phase_epochs)} entries but "
                f"size_schedule has {len(self.size_schedule)}"
            )

    def _epochs(self) -> list[int]:
        if isinstance(self.phase_epochs, list):
            return list(self.phase_epochs)
        return [self.phase_epochs] * len(self.size_schedule)

    def plan(self) -> list[Phase]:
        return [Phase(cap=c, epochs=e) for c, e in zip(self.size_schedule, self._epochs())]

    def train_trees(self, all_trees: list[RstTree], phase: Phase) -> list[RstTree]:
        if phase.cap is None:
            return all_trees
        out: list[RstTree] = []
        for tree in all_trees:
            out.extend(tree.subtrees_up_to(phase.cap))
        return out

    def dev_pairs(self, all_dev, phase):
        return all_dev if phase.cap is None else []
