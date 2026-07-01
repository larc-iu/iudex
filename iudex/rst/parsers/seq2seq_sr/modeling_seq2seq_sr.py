"""End-to-end RST parser via a fine-tuned encoder-decoder LM that emits
a linearized bottom-up shift-reduce action sequence with source tokens
interleaved verbatim. Recovers both EDU segmentation (from SHIFT
positions) and the labeled tree (from REDUCE actions).

Differs from the other iudex parsers in two ways worth flagging:
  * `forward(batch)` takes a batched dict, not a per-tree forward. The
    seq2seq fine-tuning loop runs proper batches. `train_seq2seq_sr.py`
    knows about this. `predict` and `predict_from_text` stay per-document
    so the shared predict CLI works unchanged.
  * `segmenter` is a truthy property (returns self) so the shared CLI's
    `_require_segmenter` accepts text input. This parser always segments
    by construction, there's no separate segmenter head.
"""

import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from iudex.common.log import warn
from iudex.rst.data.tree import (
    Reduce,
    RstTree,
    Shift,
)
from iudex.rst.parsers.common.seqgen import (
    ShiftReduceDecodeState,
    align_edus_to_tokens,
    beam_reorder_needed,
    beam_topk_step,
    empty_tree,
    gold_edu_source_ranges,
    mask_old_embedding_gradients,
    reconstruct_text,
    reorder_past_key_values,
    repair_actions,
    select_best_beam,
    warm_init_head,
)
from iudex.rst.parsers.seq2seq_sr.configuration_seq2seq_sr import Seq2SeqSRConfig

logger = logging.getLogger(__name__)


class Seq2SeqSRParser(nn.Module):
    def __init__(self, config: Seq2SeqSRConfig, *, compile_encoder: bool = False):
        super().__init__()
        self.config = config
        # compile_encoder is accepted for parser-CLI uniformity but has no
        # effect here. The HF seq2seq model has its own compilation story.
        del compile_encoder

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # Load weights in bf16 when the user opted into mixed precision. T5Gemma2
        # / mT5 release in bf16 anyway. HF's default upcasts to fp32 (doubling
        # weight+grad memory). bf16 here halves them at no quality cost, and the
        # optimizer states (when they inherit param dtype, e.g. torch.AdamW) get
        # halved too. A 2B-param model's optimizer footprint goes from ~32 GB
        # fp32 AdamW down to ~16 GB bf16 AdamW, or ~50 MB with Adafactor.
        # Acknowledged leak: `amp` here doubles as the load-dtype selector but is
        # in iudex.rst.HASH_EXCLUDE, so flipping it keeps the same run hash. A
        # bf16 checkpoint resuming into an fp32 model (or vice versa) under an
        # existing run dir is unsupported. Start a fresh run dir if you change amp.
        # bf16 master weights are fine under LoRA (adapters train on top) but
        # AdamW full-FT on bf16 weights degenerates (sub-resolution updates
        # round away). Full FT loads fp32, the autocast in training stays bf16.
        model_dtype = torch.bfloat16 if (config.amp and config.peft is not None) else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, dtype=model_dtype)

        # Built only after relation inference. Predict-time loads from a
        # checkpoint that has cfg.relation_types populated, so the action
        # vocab is always installed by the time load_state_dict fires.
        self.action_token_ids: dict[str, int] = {}
        self.shift_token_id: int | None = None
        self.reduce_token_ids: set[int] = set()
        self.reduce_token_map: dict[str, Tuple[str, str]] = {}
        if config.relation_types is not None:
            self._install_action_vocab()

        if config.peft is not None:
            self._install_peft(config.peft)

        # Replace the model's giant tied lm_head (Linear(hidden -> 262K))
        # with a small fresh head projecting only to the action vocab + EOS.
        # Done AFTER PEFT wrap so the replacement discards any LoRA adapter
        # that PEFT applied to the old out_proj. Massively shrinks logits
        # memory and decouples input (full vocab) from output (action vocab).
        if config.relation_types is not None:
            self._install_action_head()
            # Train only the newly-added action-token embedding rows via the
            # shared gradient-mask helper (keeps the full embedding trainable,
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
            # Required for gradient checkpointing to actually save memory under
            # AutoModelForSeq2SeqLM: cache must be off so the backward pass
            # recomputes activations instead of recovering them from KV cache.
            self.model.config.use_cache = False
            # PEFT wraps in PeftModel which masks `.gradient_checkpointing_enable()`
            # under some versions, ensure use_cache is also off on the base.
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config"):
                self.model.base_model.config.use_cache = False

    # -----------------------------------------------------------------
    # Action vocabulary installation
    # -----------------------------------------------------------------

    # Sentinel emitted at every source-copy position. The decoder predicts
    # this in place of the actual source subword ID. Tree reconstruction
    # walks a cursor over the input subwords and appends `source_ids[cursor]`
    # for each emitted COPY. This dramatically simplifies the decoder's
    # decision space, every prediction is now over a ~100-action vocab
    # instead of a 262K-vocab softmax dominated by trivially-copyable
    # source tokens.
    COPY_TOKEN: str = "<copy>"

    def _build_action_vocab(self) -> List[str]:
        """Derive the full action-token list from `cfg.relation_types`. Each
        `(rel, kind)` pair contributes either one NN reduce (for multinuc) or
        two reduces (NS, SN for rst-kind). Plus SHIFT and the COPY sentinel.
        """
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
        """Wrap `self.model` in a PeftModel with LoRA adapters. The input
        embedding is NOT handed to PEFT `modules_to_save`: PEFT would keep a
        frozen original copy plus a full trainable copy of the vocab x hidden
        matrix (~600 MB each at 1B scale) and de-tie the encoder/decoder
        embeddings. We instead keep the single tied embedding trainable and
        zero pretrained-row gradients (`seqgen.mask_old_embedding_gradients`).
        The lm_head is likewise out of `modules_to_save` (replaced wholesale by
        a small fresh head)."""
        from peft import LoraConfig, TaskType, get_peft_model

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=peft_cfg.r,
            lora_alpha=peft_cfg.alpha,
            lora_dropout=peft_cfg.dropout,
            target_modules=peft_cfg.target_modules,
            bias=peft_cfg.bias,
            use_dora=peft_cfg.dora,
        )
        self.model = get_peft_model(self.model, lora_cfg)

    def _set_grad_checkpointing(self, enabled: bool) -> None:
        """Toggle gradient checkpointing on the underlying base model. PEFT
        forwards these methods through but be defensive."""
        method = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        fn = getattr(self.model, method, None)
        if callable(fn):
            fn()
            return
        # Fallback: walk submodules and flip the flag directly.
        for mod in self.model.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = enabled

    def _resolve_decoder_start_token_id(self) -> int:
        """Find the decoder start token in a model-family-portable way.
        T5/mT5 expose `decoder_start_token_id` on `config`. T5Gemma 2 only
        exposes it via `model.prepare_decoder_input_ids_from_labels` (which
        prepends `bos_token_id`). Probe the standard paths in order, then
        invoke the canonical HF helper on a stub label to extract whatever
        token gets prepended."""
        for src in (
            getattr(self.model, "generation_config", None),
            getattr(self.model, "config", None),
        ):
            if src is None:
                continue
            tok = getattr(src, "decoder_start_token_id", None)
            if tok is not None:
                return int(tok)
        prepare = getattr(self.model, "prepare_decoder_input_ids_from_labels", None)
        if prepare is not None:
            stub = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            shifted = prepare(labels=stub) if "labels" in prepare.__code__.co_varnames else prepare(stub)
            return int(shifted[0, 0].item())
        for src in (
            getattr(self.model, "generation_config", None),
            getattr(self.model, "config", None),
        ):
            bos = getattr(src, "bos_token_id", None) if src is not None else None
            if bos is not None:
                return int(bos)
        return int(self.tokenizer.pad_token_id)

    def _install_action_vocab(self) -> None:
        action_vocab = self._build_action_vocab()
        # Snapshot original vocab size BEFORE adding new tokens, used by
        # the embedding gradient mask to identify "old" (pretrained) rows
        # that should stay frozen.
        self._original_vocab_size = len(self.tokenizer)
        # Skip tokens that already exist (re-loading a checkpoint with
        # action tokens already in the tokenizer through some flow).
        existing = set(self.tokenizer.get_vocab().keys())
        new_tokens = [t for t in action_vocab if t not in existing]
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.action_token_ids = {t: self.tokenizer.convert_tokens_to_ids(t) for t in action_vocab}
        self.shift_token_id = self.action_token_ids[Shift().to_token()]
        self.copy_token_id = self.action_token_ids[self.COPY_TOKEN]
        self.decoder_start_token_id = self._resolve_decoder_start_token_id()
        self.reduce_token_ids = {
            tok_id
            for token_str, tok_id in self.action_token_ids.items()
            if token_str not in (Shift().to_token(), self.COPY_TOKEN)
        }

        # Build the small action-head vocabulary used by the replacement
        # `lm_head`. Order: COPY, SHIFT, sorted REDUCE-*, EOS. The head's
        # output dim is `len(self.full_id_for_head_idx)`. Labels (full
        # vocab IDs) get mapped to head indices via `_label_to_head_lookup`
        # at training time. Head argmax indices map back to full IDs via
        # `self.full_id_for_head_idx` at inference time.
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

        # `structural` (for the loss split + action_loss_weight upweight)
        # = shift + all reduces, in HEAD-VOCAB index space.
        structural_head_ids = sorted(self.reduce_head_indices | {self.shift_head_idx})
        self.register_buffer(
            "_structural_token_ids_buf",
            torch.tensor(structural_head_ids, dtype=torch.long),
            persistent=False,
        )
        # Lookup table: full-vocab ID -> head index, else -100 (ignore).
        # Used to map training labels (full IDs) to head indices.
        max_full_id = max(self.full_id_for_head_idx) + 1
        lookup = torch.full((max_full_id,), -100, dtype=torch.long)
        for fid, hi in self.head_idx_for_full_id.items():
            lookup[fid] = hi
        self.register_buffer("_label_to_head_lookup", lookup, persistent=False)
        # Reduce indices as a buffer for fast logits masking in the predict loop.
        self.register_buffer(
            "_reduce_head_ids_buf",
            torch.tensor(sorted(self.reduce_head_indices), dtype=torch.long),
            persistent=False,
        )

    def _install_action_head(self) -> None:
        """Replace the model's tied lm_head (a Linear(hidden -> ~262K) sharing
        its weight with embed_tokens) with a fresh `Linear(hidden -> head_vocab_size)`
        projecting only to {COPY, SHIFT, REDUCE-*, EOS}. Removes the head's
        role as a 262K-wide unembedding (the model only ever predicts these
        ~100 actions, and the [B, T, 262K] logits tensor at training was
        ~8 GB at moderate sequence lengths). Also discards any LoRA adapter
        PEFT applied to the original out_proj. We want this small head
        fully trainable, not low-rank-deltaed."""
        base = self._underlying_model()

        # Locate the existing output projection. T5Gemma 2 wraps it inside a
        # `T5Gemma2LMHead` module with an `out_proj` Linear. mT5 / T5 expose
        # `lm_head` directly as the Linear.
        if (
            hasattr(base, "lm_head")
            and hasattr(base.lm_head, "out_proj")
            and isinstance(base.lm_head.out_proj, nn.Linear)
        ):
            old = base.lm_head.out_proj
            hidden = old.in_features
            new = nn.Linear(hidden, self.head_vocab_size, bias=False).to(
                dtype=old.weight.dtype, device=old.weight.device
            )
            warm_init_head(new, self._underlying_model().get_input_embeddings().weight, self.full_id_for_head_idx)
            base.lm_head.out_proj = new
        elif hasattr(base, "lm_head") and isinstance(base.lm_head, nn.Linear):
            old = base.lm_head
            hidden = old.in_features
            new = nn.Linear(hidden, self.head_vocab_size, bias=False).to(
                dtype=old.weight.dtype, device=old.weight.device
            )
            warm_init_head(new, self._underlying_model().get_input_embeddings().weight, self.full_id_for_head_idx)
            base.lm_head = new
        else:
            raise RuntimeError(
                f"Don't know how to replace lm_head on {type(base).__name__}. Expected "
                f"`lm_head` as Linear or `lm_head.out_proj` as Linear."
            )
        logger.info(
            f"Replaced lm_head with fresh Linear(hidden={hidden}, head_vocab_size={self.head_vocab_size}). "
            f"Logits tensor at training shrinks ~{old.out_features / self.head_vocab_size:.0f}× per token."
        )

    def _underlying_model(self):
        """Walk PEFT wrappers to reach the original HF model (the one that owns
        the embeddings + lm_head). Plain attribute-walking is safe on the
        encoder-decoder backbone: unlike the causal sibling, its `base_model`
        shortcut does not over-descend past the LM head, so no PEFT-module-origin
        gate is needed (contrast `decoder_only_sr._underlying_model`)."""
        m = self.model
        if hasattr(m, "base_model"):
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
    ) -> "Seq2SeqSRParser":
        from iudex.rst.parsers.hfhub import load_parser_from_pretrained

        dev = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return load_parser_from_pretrained(
            repo_or_path,
            parser_cls=cls,
            config_cls=Seq2SeqSRConfig,
            device=dev,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            compile_encoder=compile_encoder,
        )

    # -----------------------------------------------------------------
    # Training forward (batched)
    # -----------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """Standard seq2seq teacher-forced cross-entropy. Batched: `batch`
        carries `input_ids`, `attention_mask`, `labels` on the model device.

        Returns the scalar `loss` used for backward plus diagnostic split:
          * `action_loss`: mean CE over labels whose target is an action
            token (`<shift>` or `<reduce_*>`)
          * `copy_loss`: mean CE over the remaining (source-copy) labels

        When `cfg.action_loss_weight != 1.0`, the action-position
        contribution is upweighted in the loss used for backward (copy
        positions dominate the target by ~10:1, so the default token-uniform
        average buries the parsing signal in the trivially-easy copy task).

        When `cfg.width_band_loss` is set, `batch` additionally carries
        `loss_weights` (built in `encode_target`, padded with 1.0) and the
        base CE becomes a weighted mean over positions. With the knob unset
        the batch has no `loss_weights` key and this path is byte-identical
        to before the knob existed.
        """
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            return_dict=True,
        )
        logits = out.logits  # [B, T, head_vocab_size]

        # Map labels (full vocab IDs) → head indices via the lookup buffer.
        # Labels are full-vocab IDs in {copy, shift, reduce_*, eos} plus
        # -100 for padding. The lookup buffer maps those to head indices
        # 0..head_vocab_size-1 (or -100 for anything not in the action
        # vocab, defensive, shouldn't normally happen).
        labels = batch["labels"]
        labels_flat = labels.reshape(-1)
        max_id = self._label_to_head_lookup.size(0) - 1
        in_range = (labels_flat >= 0) & (labels_flat <= max_id)
        clamped = labels_flat.clamp(min=0, max=max_id)
        head_labels_flat = torch.where(
            in_range,
            self._label_to_head_lookup[clamped],
            torch.full_like(labels_flat, -100),
        )

        loss_weights = batch.get("loss_weights")
        if loss_weights is None:
            base_loss = F.cross_entropy(
                logits.reshape(-1, self.head_vocab_size).float(),
                head_labels_flat,
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )
        else:
            per_pos = F.cross_entropy(
                logits.reshape(-1, self.head_vocab_size).float(),
                head_labels_flat,
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
                reduction="none",
            )
            valid = head_labels_flat != -100
            w = loss_weights.reshape(-1)
            # Weighted mean keeps the loss scale comparable to the unweighted
            # path (all-1.0 weights reduce to exactly the mean above).
            base_loss = (per_pos * w)[valid].sum() / w[valid].sum()
        metrics: dict[str, torch.Tensor] = {"loss": base_loss}

        if self._structural_token_ids_buf.numel() == 0:
            return metrics

        # Split: structural (shift + reduces) CE vs copy/eos CE. All in
        # head-vocab index space.
        valid_mask = head_labels_flat != -100
        is_structural = torch.isin(head_labels_flat, self._structural_token_ids_buf) & valid_mask
        n_total = int(valid_mask.sum().item())
        n_structural = int(is_structural.sum().item())
        n_copy = n_total - n_structural
        if n_structural == 0 or n_copy == 0:
            return metrics

        structural_idx = is_structural.nonzero(as_tuple=True)[0]
        logits_flat = logits.reshape(-1, self.head_vocab_size)
        structural_logits = logits_flat.index_select(0, structural_idx).float()
        structural_labels = head_labels_flat.index_select(0, structural_idx)
        action_loss = F.cross_entropy(structural_logits, structural_labels, label_smoothing=self.config.label_smoothing)

        # Copy CE derived from the sum identity (no gradient: the copy
        # gradient already contributes through base_loss).
        with torch.no_grad():
            copy_loss = (base_loss.detach() * n_total - action_loss.detach() * n_structural) / max(n_copy, 1)

        metrics["action_loss"] = action_loss.detach()
        metrics["copy_loss"] = copy_loss
        metrics["n_action_tokens"] = torch.tensor(n_structural, dtype=torch.long)

        # Structural-action upweight: add an extra `alpha * action_loss` term
        # to the backward loss. With alpha = (w - 1) * n_structural / n_total
        # the gradient becomes proportional to `sum_copy_ce + w * sum_struct_ce`,
        # i.e. a w-weighted CE up to a batch-composition-dependent scalar.
        w = self.config.action_loss_weight
        if w != 1.0 and n_total > 0:
            alpha = (w - 1.0) * n_structural / n_total
            metrics["loss"] = base_loss + alpha * action_loss

        return metrics

    # -----------------------------------------------------------------
    # Tokenization
    # -----------------------------------------------------------------

    def encode_input(self, text: str) -> dict[str, list[int]]:
        """Encode the raw document for the encoder side. Shrieks if the
        tokenizer truncates. Silent input truncation corrupts the training
        signal (the target still references EDUs whose source tokens the
        encoder never saw) and silently degrades inference (the model can't
        parse anything past the cut)."""
        # Untruncated length is our truncation detector. Two tokenizer calls
        # is the simplest reliable signal. The second is bounded by
        # max_input_length so the cost is negligible vs. the forward pass.
        full_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
        )
        if full_len > self.config.max_input_length - 1:  # -1 for the trailing EOS
            warn(
                f"Input truncated: {full_len} -> {self.config.max_input_length} subwords. "
                f"Bump max_input_length (model supports up to ~32K for T5Gemma 2, "
                f"~1K-2K for mT5) or this doc's tail is invisible to the model."
            )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def encode_target(self, tree: RstTree) -> tuple[list[int], list[int], list[float] | None] | None:
        """Build aligned target streams (plus optional per-position weights):

          * `labels`: the prediction targets, `<copy>` at source-copy
            positions, `<shift>` and `<reduce_*>` at structural positions,
            `<eos>` at the end. The decoder is trained to PREDICT these.

          * `decoder_input_ids`: what the decoder ACTUALLY SEES in its
            self-attention history, where `<copy>` is replaced by the actual
            source subword ID, so the decoder's input distribution at
            training matches inference (where we substitute before
            appending). Length-aligned with `labels`: position i is
            shift-right of label i (with decoder_start at position 0).

        Length is identical to the previous "actual source subword IDs"
        formulation, but the model's prediction vocabulary at every position
        is now just the ~100 action tokens instead of the 262K full vocab.
        The deterministic copy task no longer competes with structural
        decisions for gradient.

        The third element is `loss_weights`, aligned with `labels`: None
        when `cfg.width_band_loss` is unset, else 1.0 everywhere except
        reduce positions whose merged constituent width the band covers
        (those get `band.weight`). Widths come from replaying the oracle
        with a width stack, so they are exact.

        Returns None when the target overflows `cfg.max_output_length`."""
        if self.shift_token_id is None:
            raise RuntimeError(
                "encode_target called before action vocab was installed. Did you forget to set cfg.relation_types?"
            )
        actions = tree.to_shift_reduce(include_text=False)
        # The per-EDU subword id slices must TILE the whole-doc tokenization so
        # the training-time COPY substitutions match the inference-time
        # `source_ids` stream exactly. `align_edus_to_tokens` guarantees that
        # tiling. The same helper drives `gold_edu_source_ranges` and the
        # trainer's `_gold_edu_token_mapping`.
        text = reconstruct_text(tree)
        full_input_ids, spans = align_edus_to_tokens(self.tokenizer, text, tree.edus)
        edu_subword_ids = [full_input_ids[s:e] for (s, e) in spans]

        band = self.config.width_band_loss
        label_ids: list[int] = []
        seen_ids: list[int] = []  # what the decoder sees in its history (substituted)
        loss_weights: list[float] | None = [] if band is not None else None
        width_stack: list[int] = []  # EDU width of each open stack item
        edu_idx = 0
        for action in actions:
            if isinstance(action, Shift):
                # COPY label + actual source subword in the seen stream, per
                # subword in this EDU. Then SHIFT (same in both streams).
                for src_id in edu_subword_ids[edu_idx]:
                    label_ids.append(self.copy_token_id)
                    seen_ids.append(src_id)
                label_ids.append(self.shift_token_id)
                seen_ids.append(self.shift_token_id)
                edu_idx += 1
                if band is not None:
                    loss_weights.extend([1.0] * (len(label_ids) - len(loss_weights)))
                    width_stack.append(1)
            elif isinstance(action, Reduce):
                token_str = action.to_token()
                if token_str not in self.action_token_ids:
                    raise ValueError(
                        f"encode_target: Reduce {action!r} produced token {token_str!r} "
                        f"not in this parser's action vocabulary. Did `cfg.relation_types` "
                        f"miss this (rel, nuc)?"
                    )
                tok = self.action_token_ids[token_str]
                label_ids.append(tok)
                seen_ids.append(tok)
                if band is not None:
                    merged = width_stack.pop() + width_stack.pop()
                    width_stack.append(merged)
                    loss_weights.append(band.weight if band.covers(merged) else 1.0)
        label_ids.append(self.tokenizer.eos_token_id)
        seen_ids.append(self.tokenizer.eos_token_id)
        if band is not None:
            loss_weights.append(1.0)

        # decoder_input_ids = shift_right(seen_ids): decoder_start at pos 0,
        # then seen_ids[:-1]. Same length as labels.
        decoder_input_ids = [self.decoder_start_token_id] + seen_ids[:-1]

        if len(label_ids) > self.config.max_output_length:
            warn(
                f"Target truncated: {len(label_ids)} > max_output_length="
                f"{self.config.max_output_length} for a {len(tree.edus)}-EDU tree. "
                f"Tree cannot be encoded (training raises on this). Bump max_output_length (model supports "
                f"up to 32K for T5Gemma 2)."
            )
            return None
        return label_ids, decoder_input_ids, loss_weights

    # -----------------------------------------------------------------
    # Greedy decoding
    # -----------------------------------------------------------------

    def _strip_specials(self, ids: list[int]) -> list[int]:
        """Strip a leading BOS and trailing EOS / pad from an encoded id list."""
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        bos_id = self.tokenizer.bos_token_id
        while ids and ids[-1] == pad_id:
            ids.pop()
        if ids and ids[-1] == eos_id:
            ids.pop()
        if bos_id is not None and ids and ids[0] == bos_id:
            ids = ids[1:]
        return ids

    @torch.no_grad()
    def predict_from_text(self, text: str, *, num_beams: int | None = None) -> RstTree:
        """Constrained beam search → tree. `num_beams` overrides
        `cfg.num_beams` for this call (used by dev eval to force greedy
        when `cfg.eval_decode_greedy` is True)."""
        return self.predict_batch_from_texts([text], num_beams=num_beams)[0]

    @torch.no_grad()
    def predict_batch_from_texts(
        self,
        texts: list[str],
        *,
        num_beams: int | None = None,
    ) -> list[RstTree]:
        """Generate trees for a batch of documents. Dispatches to a batched
        greedy path (num_beams <= 1) or a per-example beam search path
        (num_beams > 1). Greedy is used for per-epoch dev eval (via
        `cfg.eval_decode_greedy`). Beam is used for the final eval (via
        `cfg.num_beams`)."""
        if not texts:
            return []
        effective_beams = int(num_beams if num_beams is not None else self.config.num_beams)
        if effective_beams <= 1:
            return self._predict_batch_greedy(texts)
        results: list[RstTree] = []
        for text in texts:
            results.append(self._predict_one_beam(text, effective_beams))
        return results

    @torch.no_grad()
    def _predict_batch_greedy(self, texts: list[str]) -> list[RstTree]:
        """Batched greedy decoding. One decoder forward per step covers all
        rows in parallel. The validity mask + COPY substitution work per-row."""
        self.eval()
        device = self.device

        enc = self.tokenizer(
            texts,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        # Per-row source IDs (the subwords the decoder will paste in at
        # COPY positions). Strip leading BOS and trailing EOS / pad so the
        # cursor tracks ONLY the content subwords, matching what
        # `encode_target` substitutes at training (per-EDU tokenization with
        # `add_special_tokens=False`). T5/mT5 add EOS at the tail. T5Gemma 2
        # prepends BOS at the head. Either or both can be present.
        per_row_source_ids: list[list[int]] = []
        empty_rows: list[int] = []
        for i in range(enc["input_ids"].shape[0]):
            ids = self._strip_specials(enc["input_ids"][i].tolist())
            full_len = len(self.tokenizer(texts[i], add_special_tokens=False).input_ids)
            if full_len > self.config.max_input_length - 1:  # -1 for the trailing EOS
                warn(
                    f"Input truncated for batch row {i}: {full_len} -> ~"
                    f"{self.config.max_input_length} subwords. Bump max_input_length "
                    f"(T5Gemma 2 supports 32K, mT5 effectively 1K-2K)."
                )
            if not ids:
                empty_rows.append(i)
            per_row_source_ids.append(ids)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)

        try:
            # Encoder pass (once, reused via encoder_outputs).
            encoder = self.model.get_encoder()
            enc_out = encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                return_dict=True,
            )

            B = enc["input_ids"].shape[0]
            decoder_start = self.decoder_start_token_id

            min_edu_len = max(1, int(self.config.min_edu_length))
            # Per-row decode state. `st.pred_edu_ranges` live in source_ids
            # token space (the same space as the encoder's tokenization), so
            # eval metrics can use them without a lossy decode→split→re-tokenize
            # round trip. Empty rows start done so they're skipped throughout.
            states = [
                ShiftReduceDecodeState(source_len=len(per_row_source_ids[i]), min_edu_length=min_edu_len)
                for i in range(B)
            ]
            for i in empty_rows:
                states[i].done = True
            action_seqs: list[list[int]] = [[] for _ in range(B)]
            hit_max_len = [False] * B

            # Decoder input + KV cache
            decoder_input_ids = torch.full((B, 1), decoder_start, device=device, dtype=torch.long)
            past_key_values = None

            for step in range(self.config.max_output_length):
                if all(st.done for st in states):
                    break

                # Feed only the last token after step 0 (cache holds the rest).
                step_input = decoder_input_ids[:, -1:] if past_key_values is not None else decoder_input_ids
                out = self.model(
                    encoder_outputs=enc_out,
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                # NOTE: `logits[:, -1, :]` is now [B, head_vocab_size], not
                # [B, 262K]. The replaced lm_head emits only action logits.
                logits = out.logits[:, -1, :]  # [B, head_vocab_size]

                # Per-row validity mask: only legal HEAD INDICES survive.
                if self.config.use_validity_constraints:
                    masked = torch.full_like(logits, float("-inf"))
                    for i, st in enumerate(states):
                        if st.done:
                            continue
                        if st.copy_ok:
                            masked[i, self.copy_head_idx] = logits[i, self.copy_head_idx]
                        if st.shift_ok:
                            masked[i, self.shift_head_idx] = logits[i, self.shift_head_idx]
                        if st.reduce_ok:
                            masked[i, self._reduce_head_ids_buf] = logits[i, self._reduce_head_ids_buf]
                        if st.eos_ok:
                            masked[i, self.eos_head_idx] = logits[i, self.eos_head_idx]
                    logits = masked

                # Greedy in head-vocab space.
                next_head_indices = logits.argmax(-1).tolist()  # [B]

                # Update state per row + compute next decoder-input token.
                # The decoder INPUT is in full vocab (source-subword IDs get
                # substituted at COPY), so we map head index → full ID here.
                next_inputs = [pad_id] * B
                for i, head_idx in enumerate(next_head_indices):
                    st = states[i]
                    if st.done:
                        continue
                    full_id = self.full_id_for_head_idx[head_idx]
                    action_seqs[i].append(full_id)
                    if full_id == eos_id:
                        st.step_eos()
                    elif full_id == self.copy_token_id:
                        if st.step_copy():
                            next_inputs[i] = per_row_source_ids[i][st.cursor - 1]
                        # else st.done already set; next_inputs stays pad
                    elif full_id == self.shift_token_id:
                        st.step_shift()
                        next_inputs[i] = full_id
                    elif full_id in self.reduce_token_ids:
                        st.step_reduce()
                        next_inputs[i] = full_id
                    else:
                        # Shouldn't happen under valid mask.
                        st.done = True

                if all(st.done for st in states):
                    break

                new_step = torch.tensor(next_inputs, device=device, dtype=torch.long).unsqueeze(1)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
            else:
                # for-loop exhausted without `break`: max_output_length hit.
                for i, st in enumerate(states):
                    if not st.done:
                        hit_max_len[i] = True
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        # Tree reconstruction per row.
        results: list[RstTree] = []
        empty_set = set(empty_rows)
        for i in range(B):
            if i in empty_set:
                results.append(empty_tree(self.config.relation_types))
                continue
            if hit_max_len[i]:
                warn(
                    f"Output truncated at inference (batch row {i}): generation hit "
                    f"max_output_length={self.config.max_output_length} without EOS. "
                    f"Tree closed by best-effort repair."
                )
            st = states[i]
            # If decoding stopped mid-EDU (uncommitted COPYs since the last
            # SHIFT), record the in-flight (start, cursor) range so seg eval
            # sees the truncated EDU. `repair_actions` independently appends
            # a synthetic <shift> for trailing source tokens, so the action
            # sequence and pred_edu_ranges stay consistent.
            if st.cursor > st.edu_start:
                st.pred_edu_ranges.append((st.edu_start, st.cursor))
            tree = self._finalize_tree(action_seqs[i], per_row_source_ids[i], st.pred_edu_ranges)
            results.append(tree)
        return results

    # -----------------------------------------------------------------
    # Beam search
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _predict_one_beam(self, text: str, num_beams: int) -> RstTree:
        """Beam search for a single document with `num_beams` parallel beams.
        Each beam carries its own (cursor, stack, edu_length, action_seq,
        pred_edu_ranges, edu_start) state. The encoder runs once, the
        decoder forward at each step processes all K beams in parallel
        (one batch of size K). KV cache is reordered when beams change
        parents via `self.model._reorder_cache`.

        Done per-document rather than batched across documents so memory
        stays bounded by K beams regardless of dev_batch_size. Wall-time
        roughly K× single-doc greedy."""
        self.eval()
        device = self.device
        K = int(num_beams)
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        decoder_start = self.decoder_start_token_id
        min_edu_len = max(1, int(self.config.min_edu_length))

        # Encode the single doc.
        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        # Strip leading BOS + trailing pad / EOS to get the cursor stream.
        ids = self._strip_specials(enc["input_ids"][0].tolist())
        full_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        if full_len > self.config.max_input_length - 1:  # -1 for the trailing EOS
            warn(
                f"Input truncated (beam): {full_len} -> ~{self.config.max_input_length} subwords. "
                f"Bump max_input_length."
            )
        source_ids = ids
        if not source_ids:
            return empty_tree(self.config.relation_types)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)
        try:
            # Encoder pass, once for the single doc, then expand to K beams.
            encoder = self.model.get_encoder()
            enc_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)
            expanded_hidden = enc_out.last_hidden_state.expand(K, -1, -1).contiguous()
            expanded_attn = enc["attention_mask"].expand(K, -1).contiguous()
            enc_out_K = BaseModelOutput(last_hidden_state=expanded_hidden)

            # Per-beam state. One ShiftReduceDecodeState per beam, cloned from
            # the chosen parent before each beam's transition is applied.
            states = [ShiftReduceDecodeState(source_len=len(source_ids), min_edu_length=min_edu_len) for _ in range(K)]
            action_seqs: list[list[int]] = [[] for _ in range(K)]
            errored = [False] * K  # done via a garbage action, not EOS
            # Finished beams (EOS-terminated) are pulled out of the active
            # set so they don't crowd the top-K. Each entry is a snapshot at
            # the moment EOS fired.
            finished_beams: list[dict] = []

            # Initial decoder input + KV cache.
            decoder_input_ids = torch.full((K, 1), decoder_start, device=device, dtype=torch.long)
            past_key_values = None
            # Only beam 0 is "alive" at step 0 (all beams have identical history,
            # so without this they'd all pick the same continuation).
            beam_scores = torch.full((K,), float("-inf"), device=device)
            beam_scores[0] = 0.0

            for step in range(self.config.max_output_length):
                if all(st.done for st in states):
                    break
                step_input = decoder_input_ids[:, -1:] if past_key_values is not None else decoder_input_ids
                out = self.model(
                    encoder_outputs=enc_out_K,
                    attention_mask=expanded_attn,
                    decoder_input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]  # [K, head_V]

                # Per-beam validity mask.
                legal = torch.zeros_like(logits, dtype=torch.bool)
                for j, st in enumerate(states):
                    if st.done:
                        # Done beams are frozen in `finished_beams`. Their
                        # row stays all False so they can never re-enter the
                        # top-K and crowd out active beams.
                        continue
                    if st.copy_ok:
                        legal[j, self.copy_head_idx] = True
                    if st.shift_ok:
                        legal[j, self.shift_head_idx] = True
                    if st.reduce_ok:
                        legal[j, self._reduce_head_ids_buf] = True
                    if st.eos_ok:
                        legal[j, self.eos_head_idx] = True
                top_scores, parent_of_new, action_of_new = beam_topk_step(beam_scores, logits, legal, K)

                parent_tensor = torch.tensor(parent_of_new, device=device, dtype=torch.long)
                if beam_reorder_needed(step, parent_of_new, K, past_key_values):
                    past_key_values = reorder_past_key_values(past_key_values, parent_tensor, self._underlying_model())
                decoder_input_ids = decoder_input_ids[parent_tensor]

                # Carry per-beam state from parents. Each child gets its OWN
                # cloned state so sibling beams expanded from the same parent
                # don't share (and mutate) one object.
                new_states = [states[p].clone() for p in parent_of_new]
                new_action_seqs = [list(action_seqs[p]) for p in parent_of_new]
                new_errored = [errored[p] for p in parent_of_new]

                # Apply each beam's chosen action.
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
                        else:
                            # st.done already set; next_inputs stays pad.
                            new_errored[j] = True
                    elif full_id == self.shift_token_id:
                        st.step_shift()
                        next_inputs[j] = full_id
                    elif full_id in self.reduce_token_ids:
                        st.step_reduce()
                        next_inputs[j] = full_id
                    else:
                        st.done = True
                        new_errored[j] = True

                states = new_states
                action_seqs = new_action_seqs
                errored = new_errored
                beam_scores = top_scores

                # Snapshot any newly-finished beams into `finished_beams`,
                # then freeze their active score to -inf so they can't win
                # top-K again. Active beams keep their score and state.
                for j, st in enumerate(states):
                    if st.done and torch.isfinite(beam_scores[j]):
                        if errored[j]:
                            # A finite-score beam done via a garbage action
                            # (unknown token / copy past end) means the
                            # legality mask and the state machine disagree (a
                            # bug, not normal dead-beam topk backfill, which is
                            # always -inf). Drop the beam rather than record a
                            # broken prefix as a finished candidate.
                            warn(
                                "Beam took a mask-legal but automaton-illegal "
                                "action (mask/state mismatch). Dropping the beam."
                            )
                            beam_scores[j] = float("-inf")
                            continue
                        finished_beams.append(
                            {
                                "action_seq": list(action_seqs[j]),
                                "pred_edu_ranges": list(st.pred_edu_ranges),
                                "score": float(beam_scores[j].item()),
                                "length": len(action_seqs[j]),
                            }
                        )
                        beam_scores[j] = float("-inf")

                new_step = torch.tensor(next_inputs, device=device, dtype=torch.long).unsqueeze(1)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        # Build the candidate pool: finished beams + still-active beams.
        # Each candidate carries its own action_seq, pred_edu_ranges,
        # score, and length.
        for fb in finished_beams:
            fb["finished"] = True
        candidates: list[dict] = list(finished_beams)
        for j, st in enumerate(states):
            if not st.done and torch.isfinite(beam_scores[j]):
                # Active beam: pred_edu_ranges may be missing an in-flight
                # EDU if max_output_length cut us off mid-EDU. Mirror the
                # greedy-path fix so seg eval sees the truncated EDU. The
                # action sequence gets a synthetic <shift> via
                # `repair_actions` downstream. Appending here keeps the
                # two data structures consistent.
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

        best = select_best_beam(candidates)
        if not best.get("finished", False):
            warn(
                f"Output truncated at inference (beam): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS for any beam. "
                f"Tree closed by best-effort repair."
            )
        return self._finalize_tree(best["action_seq"], source_ids, best["pred_edu_ranges"])

    # -----------------------------------------------------------------
    # Gold-EDU forced decode
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_with_gold_edus(self, tree: RstTree) -> RstTree:
        """Greedy decode with gold EDU boundaries forced at the copy/shift
        positions. The model still freely chooses every `<reduce_*>`, so
        binarization + labeling stay model-driven, only segmentation is
        supplied. Used by the trainer's final-eval path (gold-EDU eval is gated
        by the `eval_gold_edu` parameter of `_evaluate_on_dev`, not a config
        field)."""
        return self._predict_one_gold_edu(tree)

    @torch.no_grad()
    def _predict_one_gold_edu(self, tree: RstTree) -> RstTree:
        self.eval()
        device = self.device

        text = reconstruct_text(tree)
        # Re-derive the gold EDU spans in source-id space from the same
        # whole-doc tokenization the encoder will see. Duplicates the small
        # helper in train_seq2seq_sr.py rather than importing across files
        # (and keeps this path runnable without the trainer being loaded).
        gold_ranges = gold_edu_source_ranges(self.tokenizer, tree)

        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        source_ids = self._strip_specials(enc["input_ids"][0].tolist())
        source_len = len(source_ids)
        if not source_ids:
            return empty_tree(self.config.relation_types)

        # Clamp gold ranges to the (possibly truncated) source length. An EDU
        # whose start fell beyond truncation is dropped. One straddling the
        # boundary gets shortened.
        clamped_ranges: list[tuple[int, int]] = []
        for s, e in gold_ranges:
            if s >= source_len:
                break
            clamped_ranges.append((s, min(e, source_len)))
        if not clamped_ranges:
            return empty_tree(self.config.relation_types)
        edu_ends = [end for _, end in clamped_ranges]  # exclusive ends
        n_edus = len(edu_ends)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)
        try:
            encoder = self.model.get_encoder()
            enc_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)

            decoder_start = self.decoder_start_token_id
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

            decoder_input_ids = torch.full((1, 1), decoder_start, device=device, dtype=torch.long)
            past_key_values = None

            for step in range(self.config.max_output_length):
                if st.done:
                    break
                step_input = decoder_input_ids[:, -1:] if past_key_values is not None else decoder_input_ids
                out = self.model(
                    encoder_outputs=enc_out,
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=step_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                logits = out.logits[0, -1, :]  # [head_vocab_size]

                masked = torch.full_like(logits, float("-inf"))
                more_edus = edu_idx < n_edus
                current_end = edu_ends[edu_idx] if more_edus else source_len

                if more_edus and st.edu_length > 0 and st.cursor < current_end:
                    # Mid-EDU: must keep copying this EDU's subwords until its boundary.
                    masked[self.copy_head_idx] = logits[self.copy_head_idx]
                elif more_edus and st.cursor >= current_end:
                    # Boundary reached (content buffered, or an empty-span EDU): commit via SHIFT.
                    masked[self.shift_head_idx] = logits[self.shift_head_idx]
                else:
                    # Buffer empty (EDU start, after a shift/reduce) or all EDUs
                    # exhausted: the model freely chooses REDUCE vs. proceeding,
                    # so reductions interleave with segmentation instead of all
                    # deferring to the end.
                    if st.stack_size >= 2:
                        masked[self._reduce_head_ids_buf] = logits[self._reduce_head_ids_buf]
                    if more_edus:
                        masked[self.copy_head_idx] = logits[self.copy_head_idx]
                    elif st.stack_size == 1:
                        masked[self.eos_head_idx] = logits[self.eos_head_idx]

                head_idx = int(masked.argmax(-1).item())
                full_id = self.full_id_for_head_idx[head_idx]
                action_seq.append(full_id)

                if full_id == eos_id:
                    st.step_eos()
                    next_input = pad_id
                elif full_id == self.copy_token_id:
                    if st.step_copy():
                        next_input = source_ids[st.cursor - 1]
                    else:
                        next_input = pad_id
                elif full_id == self.shift_token_id:
                    st.step_shift()
                    edu_idx += 1
                    next_input = full_id
                elif full_id in self.reduce_token_ids:
                    st.step_reduce()
                    next_input = full_id
                else:
                    st.done = True
                    next_input = pad_id

                if st.done:
                    break
                new_step = torch.tensor([[next_input]], device=device, dtype=torch.long)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
            else:
                # for-else: the step loop ran to max_output_length without an
                # EOS break, so we finalize whatever decoded so far.
                hit_max_len = not st.done
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        if hit_max_len:
            warn(
                f"Output truncated at inference (gold-edu): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS. "
                f"Tree closed by best-effort repair."
            )
        if st.cursor > st.edu_start:
            st.pred_edu_ranges.append((st.edu_start, st.cursor))
        return self._finalize_tree(action_seq, source_ids, st.pred_edu_ranges)

    @torch.no_grad()
    def predict(self, tree: RstTree, *, num_beams: int | None = None) -> RstTree:
        """Reconstruct document text from the gold tree's EDUs, then parse
        end-to-end. The parser does not consume gold EDU boundaries."""
        text = reconstruct_text(tree)
        return self.predict_from_text(text, num_beams=num_beams)

    @torch.no_grad()
    def predict_batch(
        self,
        trees: list[RstTree],
        *,
        num_beams: int | None = None,
    ) -> list[RstTree]:
        """Batched analogue of `predict(tree)`. Reconstructs document text per
        tree, then dispatches through `predict_batch_from_texts` (batched greedy
        masked-argmax loop, or per-example beam search when num_beams > 1)."""
        texts = [reconstruct_text(t) for t in trees]
        return self.predict_batch_from_texts(texts, num_beams=num_beams)

    # -----------------------------------------------------------------
    # Tree reconstruction
    # -----------------------------------------------------------------

    def _finalize_tree(
        self, action_ids: list[int], source_ids: list[int], pred_edu_ranges: list[tuple[int, int]]
    ) -> RstTree:
        """Build the tree from the emitted action sequence and stash the source
        meta the dev eval reads off the tree: per-EDU source-position ranges
        (`_pred_edu_source_ranges`, in the encoder's source-id token space, so
        eval avoids a re-tokenize that would drift from the encoder's whole-doc
        tokenization) and the raw `_source_ids`. Side-channel attributes because
        `RstTree` doesn't natively carry them; the greedy, beam, and gold-EDU
        decoders all funnel through here."""
        tree = self._tree_from_action_sequence(action_ids, source_ids)
        tree._pred_edu_source_ranges = pred_edu_ranges  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]
        return tree

    def _tree_from_action_sequence(self, action_ids: list[int], source_ids: list[int]) -> RstTree:
        """Turn the model's emitted action sequence into an `RstTree`,
        expanding each `<copy>` action into the actual source subword at the
        running cursor position. `source_ids` is the per-row source-subword
        sequence (no specials). The action sequence comes straight from
        `predict_batch_from_texts` (only `<copy>`, `<shift>`, `<reduce_*>`,
        and `<eos>` tokens, in valid order modulo malformed-output repair)."""
        strings: list[str] = []
        source_buffer: list[int] = []
        cursor = 0
        eos_id = self.tokenizer.eos_token_id

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
            # A non-None reason also covers SUCCESSFUL repairs (closing
            # shift/reduces appended after a max_output_length cutoff). Use the
            # repaired actions when there are any: a salvaged partial tree
            # scores far better than the single-EDU fallback on exactly the
            # long documents that hit truncation. Fall back only when the
            # sequence was unrecoverable (empty actions), and let the
            # try/except below catch repaired-but-unbuildable sequences.
            if not actions:
                warn(
                    f"Unrecoverable decoder output ({malformed_reason}), falling back to "
                    f"single-EDU tree. Likely an undertrained model or max_output_length too low."
                )
                full_text = " ".join(s for s in strings if not (s == "<shift>" or s in self.reduce_token_map))
                return empty_tree(self.config.relation_types, text=full_text)
            warn(f"Decoder output repaired ({malformed_reason}).")
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
