# decoder_only_sr

End-to-end RST parser via a fine-tuned decoder-only causal LM (default `google/gemma-3-1b-it`).
Single-stream sibling of `seq2seq_sr`: instead of a separate encoder pass, the source subwords
sit in front of the action stream under one causal mask, separated by a learned
`<|start_of_actions|>` token. The action vocabulary, `<copy>` substitution mechanism,
small replacement `lm_head`, PEFT wrapping with frozen pretrained embedding rows, validity
constraints, beam search, and gold-EDU forced decode are all carried over from `seq2seq_sr`.
