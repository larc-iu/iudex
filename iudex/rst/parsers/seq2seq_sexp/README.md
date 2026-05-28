# seq2seq_sexp

End-to-end RST parser via a fine-tuned encoder-decoder LM (default `google/t5gemma-2-1b-1b`).
Sister of `seq2seq_sr`: same backbone, same training recipe, same small-action-head trick, but
the decoder emits a nested s-expression serialization of the binary RST tree (`<sexp_open>` /
`<sexp_close>` / `<reduce_*>` labels) instead of a flat shift-reduce action chain. Source
subwords are either substituted via a single `<copy>` sentinel (`use_copy: true`) or emitted
verbatim in-stream (`use_copy: false`). Traversal order is configurable (`postorder` default,
`preorder` available); the post-order variant places each relation label after its two
children, which gives the decoder both subtrees' content before committing to a label.

A pushdown-automaton validity mask (`iudex.rst.parsers.common.sexp_constraints`) enforces
structural validity during decoding so any beam that survives produces a parseable s-expression.
Tree reconstruction goes through `RstTree.from_sexp`.
