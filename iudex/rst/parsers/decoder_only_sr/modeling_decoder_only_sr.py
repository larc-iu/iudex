"""End-to-end RST parser via a fine-tuned decoder-only causal LM that emits
a linearized bottom-up shift-reduce action sequence with source tokens
interleaved verbatim. Single-stream sibling of `seq2seq_sr`.

Input layout (training and inference):

    [BOS] source_subwords [SEP] seen_actions [EOS]

where `[SEP]` is a learned `<|start_of_actions|>` token added through the
same special-token / `modules_to_save` flow as the action tokens. The
training loss masks the prefix `[BOS source SEP]` to -100 and only scores
the action portion. At inference the prefix is consumed once to seed the
KV cache, then actions are generated one at a time with COPY substituted
by the current source-cursor subword (mechanism reused verbatim from
seq2seq_sr).

Per-document inference only (the `_predict_batch_greedy` API is preserved
but loops row-by-row internally. The per-row variable-prefix-length
padding story isn't worth the speedup for smoke-test scale).
"""

import contextlib
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from iudex.common.log import warn
from iudex.rst.parsers.common.seqgen import (
    BEAM_LENGTH_PENALTY_ALPHA,
    ShiftReduceDecodeState,
    align_edus_to_tokens,
    empty_tree,
    gold_edu_source_ranges,
    mask_old_embedding_gradients,
    reconstruct_text,
    reorder_past_key_values,
    repair_actions,
)
from iudex.rst.data.tree import (
    Reduce,
    RstTree,
    Shift,
)
from iudex.rst.parsers.decoder_only_sr.configuration_decoder_only_sr import (
    DecoderOnlySRConfig,
)

logger = logging.getLogger(__name__)


class DecoderOnlySRParser(nn.Module):
    SEP_TOKEN: str = "<|start_of_actions|>"
    COPY_TOKEN: str = "<copy>"

    def __init__(self, config: DecoderOnlySRConfig, *, compile_encoder: bool = False):
        super().__init__()
        self.config = config
        # compile_encoder is accepted for parser-CLI uniformity but has no
        # effect here. The HF causal LM has its own compilation story.
        del compile_encoder

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            # Gemma-family tokenizers usually have <pad>, but fall back to eos
            # so generic causal-LM tokenizers (Llama, Qwen) work out of the box.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_dtype = torch.bfloat16 if config.amp else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, dtype=model_dtype)

        self.action_token_ids: dict[str, int] = {}
        self.shift_token_id: int | None = None
        self.reduce_token_ids: set[int] = set()
        self.reduce_token_map: dict[str, Tuple[str, str]] = {}
        self.sep_token_id: int | None = None
        if config.relation_types is not None:
            self._install_action_vocab()

        if config.peft is not None:
            self._install_peft(config.peft)

        if config.relation_types is not None:
            self._install_action_head()
            # Train only the newly-added (SEP + action) token embedding rows via
            # the shared gradient-mask helper (keeps the full embedding trainable,
            # zeroes pretrained-row gradients, never overrides the embedding
            # forward, so backbone-specific behavior like Gemma scaling is
            # preserved). See `mask_old_embedding_gradients`.
            masked = mask_old_embedding_gradients(self._underlying_model(), self._original_vocab_size)
            if masked is not None:
                n_total, n_new = masked
                logger.info(
                    f"Full {n_total}-row input embedding trainable; gradient zeroed on pretrained "
                    f"rows so only the {n_new} new action-token rows update."
                )

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config"):
                self.model.base_model.config.use_cache = False

    # -----------------------------------------------------------------
    # Action vocabulary installation
    # -----------------------------------------------------------------

    def _build_action_vocab(self) -> List[str]:
        assert self.config.relation_types is not None
        shift_token = Shift().to_token()
        reduces: list[str] = []
        self.reduce_token_map = {}
        for rel, kind in self.config.relation_types:
            nucs = ("NN",) if kind == "multinuc" else ("NS", "SN")
            for nuc in nucs:
                token = Reduce(nuc=nuc, rel=rel).to_token()
                reduces.append(token)
                self.reduce_token_map[token] = (nuc, rel)
        return [self.COPY_TOKEN, shift_token] + reduces

    def _install_peft(self, peft_cfg) -> None:
        """Wrap `self.model` in LoRA adapters. The input embedding is NOT in
        PEFT `modules_to_save` (that would duplicate the vocab x hidden matrix
        to train ~100 new rows). We keep the single embedding trainable and
        zero pretrained-row gradients instead (`mask_old_embedding_gradients`)."""
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=peft_cfg.r,
            lora_alpha=peft_cfg.alpha,
            lora_dropout=peft_cfg.dropout,
            target_modules=peft_cfg.target_modules,
            bias=peft_cfg.bias,
            use_dora=peft_cfg.dora,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def _set_grad_checkpointing(self, enabled: bool) -> None:
        method = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        fn = getattr(self.model, method, None)
        if callable(fn):
            fn()
            return
        for mod in self.model.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = enabled

    @contextlib.contextmanager
    def _inference_mode(self):
        """Disable gradient checkpointing AND force `config.use_cache=True`
        during prediction. Required: at `__init__` we set `use_cache=False`
        on the underlying config when `gradient_checkpointing=True` so the
        backward pass actually recomputes activations. Some submodules and
        PEFT wrappers consult `self.config.use_cache` directly to decide
        whether to populate `past_key_values`, so toggling only the kwarg
        on the forward call isn't enough. Leaving the config flag at
        False during predict silently kills the KV cache and forces every
        step to re-encode the prefix from scratch. Restores both flags on
        exit so a subsequent training step is unaffected."""
        gc_was_on = self.config.gradient_checkpointing
        prev_use_cache = getattr(self.model.config, "use_cache", None)
        base_cfg = getattr(getattr(self.model, "base_model", None), "config", None)
        prev_base_use_cache = getattr(base_cfg, "use_cache", None) if base_cfg is not None else None
        if gc_was_on:
            self._set_grad_checkpointing(False)
        if prev_use_cache is not None:
            self.model.config.use_cache = True
        if prev_base_use_cache is not None:
            base_cfg.use_cache = True
        try:
            yield
        finally:
            if gc_was_on:
                self._set_grad_checkpointing(True)
            if prev_use_cache is not None:
                self.model.config.use_cache = prev_use_cache
            if prev_base_use_cache is not None:
                base_cfg.use_cache = prev_base_use_cache

    def _install_action_vocab(self) -> None:
        action_vocab = self._build_action_vocab()
        # Snapshot vocab size BEFORE adding any new tokens (SEP + actions).
        # Both SEP and action tokens go past this boundary so the gradient
        # mask leaves them trainable while freezing the pretrained rows.
        self._original_vocab_size = len(self.tokenizer)
        existing = set(self.tokenizer.get_vocab().keys())
        all_new = [self.SEP_TOKEN, *action_vocab]
        new_tokens = [t for t in all_new if t not in existing]
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.sep_token_id = int(self.tokenizer.convert_tokens_to_ids(self.SEP_TOKEN))
        self.action_token_ids = {t: int(self.tokenizer.convert_tokens_to_ids(t)) for t in action_vocab}
        self.shift_token_id = self.action_token_ids[Shift().to_token()]
        self.copy_token_id = self.action_token_ids[self.COPY_TOKEN]
        self.reduce_token_ids = {
            tok_id
            for token_str, tok_id in self.action_token_ids.items()
            if token_str not in (Shift().to_token(), self.COPY_TOKEN)
        }

        # Small action head: same ordering scheme as seq2seq_sr.
        eos_id = int(self.tokenizer.eos_token_id)
        self.full_id_for_head_idx: list[int] = [
            self.copy_token_id,
            self.shift_token_id,
            *sorted(self.reduce_token_ids),
            eos_id,
        ]
        self.head_idx_for_full_id: dict[int, int] = {fid: i for i, fid in enumerate(self.full_id_for_head_idx)}
        self.head_vocab_size = len(self.full_id_for_head_idx)
        self.copy_head_idx = self.head_idx_for_full_id[self.copy_token_id]
        self.shift_head_idx = self.head_idx_for_full_id[self.shift_token_id]
        self.eos_head_idx = self.head_idx_for_full_id[eos_id]
        self.reduce_head_indices = {self.head_idx_for_full_id[fid] for fid in self.reduce_token_ids}

        structural_head_ids = sorted(self.reduce_head_indices | {self.shift_head_idx})
        self.register_buffer(
            "_structural_token_ids_buf",
            torch.tensor(structural_head_ids, dtype=torch.long),
            persistent=False,
        )
        max_full_id = max(self.full_id_for_head_idx) + 1
        lookup = torch.full((max_full_id,), -100, dtype=torch.long)
        for fid, hi in self.head_idx_for_full_id.items():
            lookup[fid] = hi
        self.register_buffer("_label_to_head_lookup", lookup, persistent=False)
        self.register_buffer(
            "_reduce_head_ids_buf",
            torch.tensor(sorted(self.reduce_head_indices), dtype=torch.long),
            persistent=False,
        )

    def _install_action_head(self) -> None:
        """Replace the model's lm_head (tied to embed_tokens for Gemma)
        with a small fresh `Linear(hidden -> head_vocab_size)`. Done AFTER
        PEFT wrap so any LoRA adapter PEFT attached to lm_head gets
        discarded (we want this small head fully trainable, not LoRA).

        Warm-init copies each head row from `embed_tokens[full_id]`. This
        is only meaningfully informative for rows whose `full_id` was
        already in the pretrained vocab (here that's EOS, every other
        head row is for a token added via `resize_token_embeddings` and
        therefore has a freshly-random embedding). For those new rows the
        warm-init is effectively just re-randomization, but it's harmless
        and the EOS case is worth keeping."""
        base = self._underlying_model()

        def _warm_init(new_linear: nn.Linear) -> None:
            with torch.no_grad():
                embed_weight = self._underlying_model().get_input_embeddings().weight
                if embed_weight.shape[-1] != new_linear.weight.shape[-1]:
                    # Asymmetric encoder/decoder backbones (e.g. t5gemma-9b-2b:
                    # encoder embeddings are 3584-wide but the decoder lm_head
                    # projects from 2304) make the tied input embeddings the
                    # wrong width to copy into the head. Fall back to the same
                    # N(0, 0.02) init the head would otherwise get for fresh
                    # rows, rather than crashing on the dim mismatch.
                    new_linear.weight.normal_(mean=0.0, std=0.02)
                    return
                for hi, full_id in enumerate(self.full_id_for_head_idx):
                    src = embed_weight[full_id].to(dtype=new_linear.weight.dtype, device=new_linear.weight.device)
                    new_linear.weight[hi].copy_(src)

        if not hasattr(base, "lm_head"):
            raise RuntimeError(
                f"Don't know how to replace lm_head on {type(base).__name__}. "
                f"Expected an `lm_head` attribute (Linear or PEFT-wrapped Linear)."
            )
        old = base.lm_head
        weight = getattr(old, "weight", None)
        if weight is None and hasattr(old, "base_layer"):
            # PEFT lora.Linear stores the underlying weight on .base_layer.
            weight = old.base_layer.weight
        if weight is None:
            raise RuntimeError(f"lm_head on {type(base).__name__} has no `.weight` and no `.base_layer.weight`.")
        hidden = weight.shape[1]
        new = nn.Linear(hidden, self.head_vocab_size, bias=False).to(dtype=weight.dtype, device=weight.device)
        _warm_init(new)
        base.lm_head = new
        logger.info(f"Replaced lm_head with fresh Linear(hidden={hidden}, head_vocab_size={self.head_vocab_size}).")

    def _underlying_model(self):
        """Walk PEFT wrappers to the underlying causal LM class.

        Returns the class that owns `lm_head` (e.g. `Gemma3ForCausalLM`),
        regardless of whether PEFT is enabled. The discriminator is `PEFT`-
        module-origin rather than attribute presence, because HF's
        `PreTrainedModel.base_model` shortcut ALSO returns the inner
        transformer (`Gemma3TextModel`), so attribute-only walking would
        descend past the LM head on no-PEFT setups."""
        m = self.model
        if type(m).__module__.startswith("peft"):
            m = m.base_model
            if hasattr(m, "model") and not isinstance(m, nn.ModuleList):
                m = m.model
        return m

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def segmenter(self):
        # Truthy → predict_cli._require_segmenter accepts text input. This
        # parser always segments via the model's own output.
        return self

    # -----------------------------------------------------------------
    # from_pretrained
    # -----------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        *,
        device: str | torch.device | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        compile_encoder: bool = False,
    ) -> "DecoderOnlySRParser":
        from iudex.rst.parsers.hfhub import load_parser_from_pretrained

        dev = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return load_parser_from_pretrained(
            repo_or_path,
            parser_cls=cls,
            config_cls=DecoderOnlySRConfig,
            device=dev,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            compile_encoder=compile_encoder,
        )

    # -----------------------------------------------------------------
    # Training forward
    # -----------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """Causal-LM training pass with manual loss on the action positions.

        `batch` carries `input_ids [B, L]`, `attention_mask [B, L]`, and
        `labels [B, L]` with the prefix `[BOS source SEP]` already masked
        to -100. Labels in the action region are full-vocab IDs. We map
        them to head indices here so the small replacement `lm_head` can
        score them.

        Returns the same dict shape as `Seq2SeqSRParser.forward` (`loss`,
        plus the optional `action_loss` / `copy_loss` / `n_action_tokens`
        diagnostic split when there's a mix of structural and copy targets).
        """
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            use_cache=False,
        )
        logits = out.logits  # [B, L, head_vocab_size]

        # Causal-LM internal shift: prediction at position i targets token i+1.
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = batch["labels"][..., 1:].contiguous()

        labels_flat = shifted_labels.reshape(-1)
        max_id = self._label_to_head_lookup.size(0) - 1
        in_range = (labels_flat >= 0) & (labels_flat <= max_id)
        clamped = labels_flat.clamp(min=0, max=max_id)
        head_labels_flat = torch.where(
            in_range,
            self._label_to_head_lookup[clamped],
            torch.full_like(labels_flat, -100),
        )

        base_loss = F.cross_entropy(
            shifted_logits.reshape(-1, self.head_vocab_size).float(),
            head_labels_flat,
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )
        metrics: dict[str, torch.Tensor] = {"loss": base_loss}

        if self._structural_token_ids_buf.numel() == 0:
            return metrics

        valid_mask = head_labels_flat != -100
        is_structural = torch.isin(head_labels_flat, self._structural_token_ids_buf) & valid_mask
        n_total = int(valid_mask.sum().item())
        n_structural = int(is_structural.sum().item())
        n_copy = n_total - n_structural
        if n_structural == 0 or n_copy == 0:
            return metrics

        structural_idx = is_structural.nonzero(as_tuple=True)[0]
        logits_flat = shifted_logits.reshape(-1, self.head_vocab_size)
        structural_logits = logits_flat.index_select(0, structural_idx).float()
        structural_labels = head_labels_flat.index_select(0, structural_idx)
        action_loss = F.cross_entropy(structural_logits, structural_labels, label_smoothing=self.config.label_smoothing)

        with torch.no_grad():
            copy_loss = (base_loss.detach() * n_total - action_loss.detach() * n_structural) / max(n_copy, 1)

        metrics["action_loss"] = action_loss.detach()
        metrics["copy_loss"] = copy_loss
        metrics["n_action_tokens"] = torch.tensor(n_structural, dtype=torch.long)

        w = self.config.action_loss_weight
        if w != 1.0 and n_total > 0:
            alpha = (w - 1.0) * n_structural / n_total
            metrics["loss"] = base_loss + alpha * action_loss

        return metrics

    # -----------------------------------------------------------------
    # Tokenization
    # -----------------------------------------------------------------

    def _tokenize_source(self, text: str) -> list[int]:
        """Tokenize document text into the source-subword stream that the
        action stream's COPY positions will reference. No specials added
        here. Truncates to `max_input_length` with a warning."""
        full_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        enc = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=self.config.max_input_length)
        if full_len > self.config.max_input_length:
            warn(
                f"Source truncated: {full_len} -> {self.config.max_input_length} subwords. "
                f"Bump max_input_length or the doc's tail is invisible to the model."
            )
        return enc["input_ids"]

    def encode_target(self, tree: RstTree) -> tuple[list[int], list[int]] | None:
        """Build a single-stream `(input_ids, labels)` for causal training.

        Layout:
          input_ids = [BOS] + source_ids + [SEP] + seen_actions
          labels    = [-100] * (1 + len(source_ids) + 1) + label_actions

        `seen_actions` substitutes the real source subword at every COPY
        position (so the model sees the actual surface form), while
        `label_actions` keeps `<copy>` as the prediction target. Both end
        with EOS. Lengths agree by construction.

        Returns None when either side overflows its configured cap, or
        when their sum is too large to fit in one stream.
        """
        if self.shift_token_id is None or self.sep_token_id is None:
            raise RuntimeError(
                "encode_target called before action vocab was installed. Did you forget to set cfg.relation_types?"
            )

        text = reconstruct_text(tree)
        source_ids, spans = align_edus_to_tokens(self.tokenizer, text, tree.edus)

        if len(source_ids) > self.config.max_input_length:
            warn(
                f"Source side overflowed: {len(source_ids)} > max_input_length={self.config.max_input_length} "
                f"for a {len(tree.edus)}-EDU tree. Tree DROPPED from this epoch."
            )
            return None

        edu_subword_ids: list[list[int]] = [list(source_ids[s:e]) for s, e in spans]

        actions = tree.to_shift_reduce(include_text=False)
        seen_actions: list[int] = []
        label_actions: list[int] = []
        edu_idx = 0
        for action in actions:
            if isinstance(action, Shift):
                for src_id in edu_subword_ids[edu_idx]:
                    label_actions.append(self.copy_token_id)
                    seen_actions.append(src_id)
                label_actions.append(self.shift_token_id)
                seen_actions.append(self.shift_token_id)
                edu_idx += 1
            elif isinstance(action, Reduce):
                token_str = action.to_token()
                if token_str not in self.action_token_ids:
                    raise ValueError(
                        f"encode_target: Reduce {action!r} produced token {token_str!r} "
                        f"not in this parser's action vocabulary. Did `cfg.relation_types` miss this pair?"
                    )
                tok = self.action_token_ids[token_str]
                label_actions.append(tok)
                seen_actions.append(tok)
        eos_id = int(self.tokenizer.eos_token_id)
        label_actions.append(eos_id)
        seen_actions.append(eos_id)

        if len(label_actions) > self.config.max_output_length:
            warn(
                f"Target truncated: {len(label_actions)} > max_output_length={self.config.max_output_length} "
                f"for a {len(tree.edus)}-EDU tree. Tree DROPPED from this epoch."
            )
            return None

        bos_id = int(self.tokenizer.bos_token_id) if self.tokenizer.bos_token_id is not None else eos_id
        input_ids = [bos_id, *source_ids, self.sep_token_id, *seen_actions]
        labels = [-100] * (1 + len(source_ids) + 1) + label_actions
        assert len(input_ids) == len(labels), (len(input_ids), len(labels))

        # The realized single stream is [BOS] + source + [SEP] + actions, so per-side
        # caps don't bound it. Drop trees whose combined length overflows the budget
        # (the model's positional limit if cheaply known, else the sum of per-side caps).
        sum_cap = self.config.max_input_length + self.config.max_output_length + 2
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        combined_cap = min(sum_cap, max_positions) if isinstance(max_positions, int) and max_positions > 0 else sum_cap
        if len(input_ids) > combined_cap:
            warn(
                f"Combined stream overflowed: {len(input_ids)} > combined cap {combined_cap} "
                f"for a {len(tree.edus)}-EDU tree. Tree DROPPED from this epoch."
            )
            return None
        return input_ids, labels

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_from_text(self, text: str, *, num_beams: int | None = None) -> RstTree:
        return self.predict_batch_from_texts([text], num_beams=num_beams)[0]

    @torch.no_grad()
    def predict_batch_from_texts(
        self,
        texts: list[str],
        *,
        num_beams: int | None = None,
    ) -> list[RstTree]:
        if not texts:
            return []
        effective_beams = int(num_beams if num_beams is not None else self.config.num_beams)
        if effective_beams <= 1:
            return self._predict_batch_greedy(texts)
        results: list[RstTree] = []
        for text in texts:
            results.append(self._predict_one_beam(text, effective_beams))
        return results

    def _build_prefix_ids(self, source_ids: list[int]) -> list[int]:
        bos_id = (
            int(self.tokenizer.bos_token_id)
            if self.tokenizer.bos_token_id is not None
            else int(self.tokenizer.eos_token_id)
        )
        return [bos_id, *source_ids, self.sep_token_id]

    @torch.no_grad()
    def _predict_batch_greedy(self, texts: list[str]) -> list[RstTree]:
        """Per-document greedy decoding. Loops row-by-row. The single-stream
        layout means each row's prefix length is different, and the padding
        + position-id bookkeeping to truly batch this isn't worth the
        speedup at smoke-test scale. Public API matches seq2seq_sr."""
        self.eval()
        return [self._predict_one_greedy(t) for t in texts]

    @torch.no_grad()
    def _predict_one_greedy(self, text: str) -> RstTree:
        self.eval()
        device = self.device
        eos_id = int(self.tokenizer.eos_token_id)
        min_edu_len = max(1, int(self.config.min_edu_length))

        source_ids = self._tokenize_source(text)
        if not source_ids:
            return empty_tree(self.config.relation_types)
        prefix_ids = self._build_prefix_ids(source_ids)

        with self._inference_mode():
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[0, -1, :]

            st = ShiftReduceDecodeState(source_len=len(source_ids), min_edu_length=min_edu_len)
            action_seq: list[int] = []
            hit_max_len = False

            # The total number of model steps is bounded by the action stream
            # we'd ever emit, not the prefix length. max_output_length applies
            # here directly (same semantic as seq2seq_sr's decode budget).
            for step in range(self.config.max_output_length):
                if self.config.use_validity_constraints:
                    masked = torch.full_like(logits, float("-inf"))
                    if st.copy_ok:
                        masked[self.copy_head_idx] = logits[self.copy_head_idx]
                    if st.shift_ok:
                        masked[self.shift_head_idx] = logits[self.shift_head_idx]
                    if st.reduce_ok:
                        masked[self._reduce_head_ids_buf] = logits[self._reduce_head_ids_buf]
                    if st.eos_ok:
                        masked[self.eos_head_idx] = logits[self.eos_head_idx]
                    step_logits = masked
                else:
                    step_logits = logits

                head_idx = int(step_logits.argmax(-1).item())
                full_id = self.full_id_for_head_idx[head_idx]
                action_seq.append(full_id)

                if full_id == eos_id:
                    st.step_eos()
                    break
                if full_id == self.copy_token_id:
                    if not st.step_copy():
                        break
                    next_input = source_ids[st.cursor - 1]
                elif full_id == self.shift_token_id:
                    st.step_shift()
                    next_input = full_id
                elif full_id in self.reduce_token_ids:
                    st.step_reduce()
                    next_input = full_id
                else:
                    st.done = True
                    break

                step_input = torch.tensor([[next_input]], dtype=torch.long, device=device)
                out = self.model(
                    input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                logits = out.logits[0, -1, :]
            else:
                hit_max_len = not st.done

        if hit_max_len:
            warn(
                f"Output truncated at inference (greedy): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS. "
                f"Tree closed by best-effort repair."
            )
        if st.cursor > st.edu_start:
            st.pred_edu_ranges.append((st.edu_start, st.cursor))
        tree = self._tree_from_action_sequence(action_seq, source_ids)
        tree._pred_edu_source_ranges = st.pred_edu_ranges  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]
        return tree

    @torch.no_grad()
    def _predict_one_beam(self, text: str, num_beams: int) -> RstTree:
        """Per-document beam search with K parallel beams. Replicates the
        prefix across the batch dim so the prefix forward seeds K identical
        KV caches. Only beam 0 is alive at step 0 so beams diverge after the
        first decoding step (same trick as seq2seq_sr.`_predict_one_beam`).
        Reorders the KV cache through `reorder_past_key_values` when beams
        switch parents."""
        self.eval()
        device = self.device
        K = int(num_beams)
        eos_id = int(self.tokenizer.eos_token_id)
        pad_id = int(self.tokenizer.pad_token_id)
        head_V = self.head_vocab_size
        min_edu_len = max(1, int(self.config.min_edu_length))

        source_ids = self._tokenize_source(text)
        if not source_ids:
            return empty_tree(self.config.relation_types)
        prefix_ids = self._build_prefix_ids(source_ids)

        with self._inference_mode():
            prefix_t = torch.tensor([prefix_ids], dtype=torch.long, device=device).expand(K, -1).contiguous()
            attention_mask = torch.ones_like(prefix_t)
            out = self.model(
                input_ids=prefix_t,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :]  # [K, head_V]

            # Per-beam state. One ShiftReduceDecodeState per beam, cloned from
            # the chosen parent before each beam's transition is applied.
            states = [ShiftReduceDecodeState(source_len=len(source_ids), min_edu_length=min_edu_len) for _ in range(K)]
            action_seqs: list[list[int]] = [[] for _ in range(K)]
            finished_beams: list[dict] = []

            beam_scores = torch.full((K,), float("-inf"), device=device)
            beam_scores[0] = 0.0

            for step in range(self.config.max_output_length):
                if all(st.done for st in states):
                    break

                masked = torch.full_like(logits, float("-inf"))
                for j, st in enumerate(states):
                    if st.done:
                        continue
                    if st.copy_ok:
                        masked[j, self.copy_head_idx] = logits[j, self.copy_head_idx]
                    if st.shift_ok:
                        masked[j, self.shift_head_idx] = logits[j, self.shift_head_idx]
                    if st.reduce_ok:
                        masked[j, self._reduce_head_ids_buf] = logits[j, self._reduce_head_ids_buf]
                    if st.eos_ok:
                        masked[j, self.eos_head_idx] = logits[j, self.eos_head_idx]
                log_probs = F.log_softmax(masked.float(), dim=-1)
                cum = beam_scores.unsqueeze(1) + log_probs
                # Dead beams (score=-inf, all-masked row) produce -inf + NaN = NaN
                # rows. topk ranks NaN above any finite negative, so without this
                # the dead beam's children would crowd out live beams.
                cum = torch.where(torch.isnan(cum), torch.full_like(cum, float("-inf")), cum)
                # `cum` is [K, head_V]; flattening to [K*head_V] and decoding the
                # flat index splits it back into (parent_beam, action) via
                # flat = parent_beam * head_V + action.
                top_scores, top_idx = cum.view(-1).topk(K)
                parent_of_new = (top_idx // head_V).tolist()
                action_of_new = (top_idx % head_V).tolist()

                parent_tensor = torch.tensor(parent_of_new, device=device, dtype=torch.long)
                # Skip the reorder when it's provably a no-op: step 0
                # has `parent_of_new == [0]*K` because only beam 0 was
                # alive, and the prefix forward already replicated K
                # identical cache rows. Later steps where parents form
                # the identity permutation are also no-ops.
                is_step0_uniform = step == 0 and all(p == 0 for p in parent_of_new)
                is_identity = parent_of_new == list(range(K))
                needs_reorder = past_key_values is not None and not (is_step0_uniform or is_identity)
                if needs_reorder:
                    past_key_values = reorder_past_key_values(past_key_values, parent_tensor, self._underlying_model())

                # Carry per-beam state from parents. Each child gets its OWN
                # cloned state so sibling beams expanded from the same parent
                # don't share (and mutate) one object.
                new_states = [states[p].clone() for p in parent_of_new]
                new_action_seqs = [list(action_seqs[p]) for p in parent_of_new]

                next_inputs = [pad_id] * K
                for j in range(K):
                    st = new_states[j]
                    if st.done:
                        continue
                    head_idx = action_of_new[j]
                    full_id = self.full_id_for_head_idx[head_idx]
                    new_action_seqs[j].append(full_id)
                    if full_id == eos_id:
                        st.step_eos()
                    elif full_id == self.copy_token_id:
                        if st.step_copy():
                            next_inputs[j] = source_ids[st.cursor - 1]
                        # else st.done already set; next_inputs stays pad
                    elif full_id == self.shift_token_id:
                        st.step_shift()
                        next_inputs[j] = full_id
                    elif full_id in self.reduce_token_ids:
                        st.step_reduce()
                        next_inputs[j] = full_id
                    else:
                        st.done = True

                states = new_states
                action_seqs = new_action_seqs
                beam_scores = top_scores

                for j, st in enumerate(states):
                    if st.done and torch.isfinite(beam_scores[j]):
                        finished_beams.append(
                            {
                                "action_seq": list(action_seqs[j]),
                                "pred_edu_ranges": list(st.pred_edu_ranges),
                                "score": float(beam_scores[j].item()),
                                "length": len(action_seqs[j]),
                                "finished": True,
                            }
                        )
                        beam_scores[j] = float("-inf")

                if all(st.done for st in states):
                    break

                step_input = torch.tensor(next_inputs, dtype=torch.long, device=device).unsqueeze(1)
                out = self.model(
                    input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]

        candidates: list[dict] = list(finished_beams)
        for j, st in enumerate(states):
            if not st.done and torch.isfinite(beam_scores[j]):
                ranges = list(st.pred_edu_ranges)
                if st.cursor > st.edu_start:
                    ranges.append((st.edu_start, st.cursor))
                candidates.append(
                    {
                        "action_seq": list(action_seqs[j]),
                        "pred_edu_ranges": ranges,
                        "score": float(beam_scores[j].item()),
                        "length": len(action_seqs[j]),
                        "finished": False,
                    }
                )

        if not candidates:
            return empty_tree(self.config.relation_types)

        best = max(
            candidates,
            key=lambda c: c["score"] / max(c["length"], 1) ** BEAM_LENGTH_PENALTY_ALPHA,
        )
        if not best.get("finished", False):
            warn(
                f"Output truncated at inference (beam): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS for any beam. "
                f"Tree closed by best-effort repair."
            )
        tree = self._tree_from_action_sequence(best["action_seq"], source_ids)
        tree._pred_edu_source_ranges = best["pred_edu_ranges"]  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]
        return tree

    @torch.no_grad()
    def predict_with_gold_edus(self, tree: RstTree) -> RstTree:
        """Greedy decode with gold EDU boundaries forced at the copy/shift
        positions. Reduces stay model-driven, only segmentation is supplied.
        Used by the trainer's final-eval path (gold-EDU eval is gated by the
        `eval_gold_edu` parameter of `_evaluate_on_dev`, not a config field)."""
        return self._predict_one_gold_edu(tree)

    @torch.no_grad()
    def _predict_one_gold_edu(self, tree: RstTree) -> RstTree:
        self.eval()
        device = self.device
        eos_id = int(self.tokenizer.eos_token_id)

        text = reconstruct_text(tree)
        gold_ranges = gold_edu_source_ranges(self.tokenizer, tree)
        source_ids = self._tokenize_source(text)
        if not source_ids:
            return empty_tree(self.config.relation_types)
        source_len = len(source_ids)
        prefix_ids = self._build_prefix_ids(source_ids)

        clamped_ranges: list[tuple[int, int]] = []
        for s, e in gold_ranges:
            if s >= source_len:
                break
            clamped_ranges.append((s, min(e, source_len)))
        if not clamped_ranges:
            return empty_tree(self.config.relation_types)
        edu_ends = [end for _, end in clamped_ranges]
        n_edus = len(edu_ends)

        with self._inference_mode():
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            logits = out.logits[0, -1, :]

            min_edu_len = max(1, int(self.config.min_edu_length))
            # Gold-EDU decode forces the COPY/SHIFT positions, so the mask is
            # driven by `edu_idx` over the gold boundaries rather than the
            # state's validity predicates. The state still tracks cursor/stack/
            # EDU ranges and runs the shared transitions. `st.edu_length` is the
            # copies-since-last-shift buffer.
            st = ShiftReduceDecodeState(source_len=source_len, min_edu_length=min_edu_len)
            edu_idx = 0
            action_seq: list[int] = []
            hit_max_len = False

            for step in range(self.config.max_output_length):
                masked = torch.full_like(logits, float("-inf"))
                more_edus = edu_idx < n_edus
                current_end = edu_ends[edu_idx] if more_edus else source_len

                if more_edus and st.cursor < current_end:
                    masked[self.copy_head_idx] = logits[self.copy_head_idx]
                elif more_edus and st.cursor == current_end and st.edu_length == 0:
                    # Empty-span gold EDU (shorter than a subword): commit it
                    # immediately so edu_idx advances instead of drifting COPY
                    # across the boundary.
                    masked[self.shift_head_idx] = logits[self.shift_head_idx]
                elif more_edus and st.cursor == current_end and st.edu_length > 0:
                    masked[self.shift_head_idx] = logits[self.shift_head_idx]
                else:
                    if st.stack_size >= 2:
                        masked[self._reduce_head_ids_buf] = logits[self._reduce_head_ids_buf]
                    if more_edus:
                        masked[self.copy_head_idx] = logits[self.copy_head_idx]
                    else:
                        if st.stack_size == 1:
                            masked[self.eos_head_idx] = logits[self.eos_head_idx]

                head_idx = int(masked.argmax(-1).item())
                full_id = self.full_id_for_head_idx[head_idx]
                action_seq.append(full_id)

                if full_id == eos_id:
                    st.step_eos()
                    break
                if full_id == self.copy_token_id:
                    if not st.step_copy():
                        break
                    next_input = source_ids[st.cursor - 1]
                elif full_id == self.shift_token_id:
                    st.step_shift()
                    edu_idx += 1
                    next_input = full_id
                elif full_id in self.reduce_token_ids:
                    st.step_reduce()
                    next_input = full_id
                else:
                    st.done = True
                    break

                step_input = torch.tensor([[next_input]], dtype=torch.long, device=device)
                out = self.model(
                    input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                logits = out.logits[0, -1, :]
            else:
                hit_max_len = not st.done

        if hit_max_len:
            warn(
                f"Output truncated at inference (gold-edu): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS. "
                f"Tree closed by best-effort repair."
            )
        if st.cursor > st.edu_start:
            st.pred_edu_ranges.append((st.edu_start, st.cursor))
        tree_out = self._tree_from_action_sequence(action_seq, source_ids)
        tree_out._pred_edu_source_ranges = st.pred_edu_ranges  # type: ignore[attr-defined]
        tree_out._source_ids = source_ids  # type: ignore[attr-defined]
        return tree_out

    @torch.no_grad()
    def predict(self, tree: RstTree, *, num_beams: int | None = None) -> RstTree:
        text = reconstruct_text(tree)
        return self.predict_from_text(text, num_beams=num_beams)

    @torch.no_grad()
    def predict_batch(self, trees: list[RstTree], *, num_beams: int | None = None) -> list[RstTree]:
        texts = [reconstruct_text(t) for t in trees]
        return self.predict_batch_from_texts(texts, num_beams=num_beams)

    def _tree_from_action_sequence(self, action_ids: list[int], source_ids: list[int]) -> RstTree:
        strings: list[str] = []
        source_buffer: list[int] = []
        cursor = 0
        eos_id = int(self.tokenizer.eos_token_id)

        def flush_source():
            if source_buffer:
                decoded = self.tokenizer.decode(source_buffer, skip_special_tokens=False)
                strings.extend(decoded.split())
                source_buffer.clear()

        for tok in action_ids:
            if tok == eos_id:
                flush_source()
                break
            if tok == self.copy_token_id:
                if cursor < len(source_ids):
                    source_buffer.append(source_ids[cursor])
                    cursor += 1
            elif tok == self.shift_token_id:
                flush_source()
                strings.append(Shift().to_token())
            elif tok in self.reduce_token_ids:
                flush_source()
                strings.append(self.tokenizer.convert_ids_to_tokens(tok))
        flush_source()

        actions, malformed_reason = repair_actions(strings, self.reduce_token_map)
        if malformed_reason is not None:
            warn(f"Malformed decoder output ({malformed_reason}). Falling back to single-EDU tree.")
            full_text = " ".join(s for s in strings if not (s == "<shift>" or s in self.reduce_token_map))
            return empty_tree(self.config.relation_types, text=full_text)
        try:
            return RstTree.from_shift_reduce(actions, relation_types=self.config.relation_types)
        except Exception as e:
            # A balanced (repaired) action sequence can still build a
            # pathologically deep tree from an undertrained model's over-
            # segmented output, blowing Python's recursion limit in
            # binarize_tree. Real trees are shallow (GUM maxes ~235 EDUs); only
            # untrusted model output hits this, so degrade on ANY failure,
            # matching the sexp parsers' _tree_from_emitted.
            warn(f"Unbuildable shift-reduce tree ({type(e).__name__}: {e}). Falling back to single-EDU tree.")
            full_text = " ".join(s for s in strings if not (s == "<shift>" or s in self.reduce_token_map))
            return empty_tree(self.config.relation_types, text=full_text)
