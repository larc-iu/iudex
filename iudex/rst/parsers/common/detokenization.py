"""Detokenizers for reconstructing natural text from word-tokenized corpus EDUs.

End-to-end-from-text parsers (those with a bundled segmenter) train on EDU
strings that are word-tokenized in the source corpus (punctuation, brackets and
dashes spaced off, e.g. `designed ,` or `( L2 )`). At inference
`predict_from_text` receives raw natural text, so without normalization the
segmenter sees an out-of-distribution tokenization and collapses to a single
EDU. Detokenizing the corpus EDU text at train time aligns the two forms.

`Detokenizer` is a `tonga.Registrable` so the strategy is config-selectable
(`detokenizer: {type: ..., ...}`); today the only implementation wraps
sacremoses. `default_implementation` is set so the config still round-trips
through `dataclasses.asdict` (which drops the `type` discriminator) and back via
`from_params`.
"""

from dataclasses import dataclass

from tonga import Registrable


class Detokenizer(Registrable):
    default_implementation = "sacremoses"

    def detokenize(self, text: str) -> str:
        raise NotImplementedError


@Detokenizer.register("sacremoses")
@dataclass
class SacreMosesDetokenizer(Detokenizer):
    """Moses detokenizer (sacremoses). `lang` selects the language-specific
    rules (the seam for multilingual support; English-only in practice today)."""

    lang: str = "en"

    def __post_init__(self):
        from sacremoses import MosesDetokenizer

        self._md = MosesDetokenizer(lang=self.lang)

    def detokenize(self, text: str) -> str:
        return self._md.detokenize(text.split())
