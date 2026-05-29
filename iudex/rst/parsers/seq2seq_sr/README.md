# seq2seq_sr

End-to-end RST parser via a fine-tuned encoder-decoder LM (default `google/t5gemma-2-1b-1b`,
baseline `google/mt5-large`). The decoder emits a linearized bottom-up shift-reduce action
sequence with source tokens interleaved verbatim. `<shift>` marks both an EDU boundary and
the parser action that commits it to the stack. `<reduce_<nuc>_<rel>>` pops the top two stack
items and combines them. Both the EDU segmentation and the labeled tree are recovered from
this one string.

A custom `LogitsProcessor` enforces structural validity (stack/queue invariants) and
input-coverage validity (the emitted source-copy sub-sequence must equal the input subword
IDs verbatim) during beam search, so any beam that survives produces a parseable output.
