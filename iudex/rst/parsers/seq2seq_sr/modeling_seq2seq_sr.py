"""End-to-end RST parser via a fine-tuned encoder-decoder LM that emits
a linearized bottom-up shift-reduce action sequence with source tokens
interleaved verbatim. Recovers both EDU segmentation (from SHIFT
positions) and the labeled tree (from REDUCE actions).

Differs from the other iudex parsers in two ways worth flagging:
  * `forward(batch)` takes a batched dict, not a per-tree forward — the
    seq2seq fine-tuning loop runs proper batches. `train_seq2seq_sr.py`
    knows about this; `predict` and `predict_from_text` stay per-document
    so the shared predict CLI works unchanged.
  * `segmenter` is a truthy property (returns self) so the shared CLI's
    `_require_segmenter` accepts text input. This parser always segments
    by construction; there's no separate segmenter head.
"""

import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from iudex.common.log import warn
from iudex.rst.data.tree import (
    Reduce,
    RstTree,
    Shift,
    ShiftReduceAction,
    strings_to_actions,
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
        # / mT5 release in bf16 anyway; HF's default upcasts to fp32 (doubling
        # weight+grad memory). bf16 here halves them at no quality cost, and the
        # optimizer states (when they inherit param dtype, e.g. torch.AdamW) get
        # halved too — a 2B-param model's optimizer footprint goes from ~32 GB
        # fp32 AdamW down to ~16 GB bf16 AdamW, or ~50 MB with Adafactor.
        torch_dtype = torch.bfloat16 if config.amp else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, torch_dtype=torch_dtype)

        # Built only after relation inference. Predict-time loads from a
        # checkpoint that has cfg.relation_types populated, so the action
        # vocab is always installed by the time load_state_dict fires.
        self.action_token_ids: dict[str, int] = {}
        self.shift_token_id: int | None = None
        self.reduce_token_ids: set[int] = set()
        self.reduce_token_map: dict[str, Tuple[str, str]] = {}
        if config.relation_types is not None:
            self._install_action_vocab()

        # PEFT wrapping happens AFTER action-vocab install so the newly-resized
        # embedding/lm_head rows are part of the base model when peft sees it,
        # and `modules_to_save` can mark them for full training. Wrapping
        # before would leave new-token embeddings frozen at their random init.
        if config.peft is not None:
            self._install_peft(config.peft)

        # Replace the model's giant tied lm_head (Linear(hidden -> 262K))
        # with a small fresh head projecting only to the action vocab + EOS.
        # Done AFTER PEFT wrap so the replacement discards any LoRA adapter
        # that PEFT applied to the old out_proj. Massively shrinks logits
        # memory and decouples input (full vocab) from output (action vocab).
        if config.relation_types is not None:
            self._install_action_head()
            # With old embedding rows no longer doing double duty as the
            # output projection, freezing them is the simple regularization
            # play — only the new action-token rows of embed_tokens train.
            if config.peft is not None and getattr(config.peft, "train_only_new_embedding_rows", True):
                self._mask_old_embedding_gradients()

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            # Required for gradient checkpointing to actually save memory under
            # AutoModelForSeq2SeqLM: cache must be off so the backward pass
            # recomputes activations instead of recovering them from KV cache.
            self.model.config.use_cache = False
            # PEFT wraps in PeftModel which masks `.gradient_checkpointing_enable()`
            # under some versions; ensure use_cache is also off on the base.
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config"):
                self.model.base_model.config.use_cache = False

    # -----------------------------------------------------------------
    # Action vocabulary installation
    # -----------------------------------------------------------------

    # Sentinel emitted at every source-copy position. The decoder predicts
    # this in place of the actual source subword ID; tree reconstruction
    # walks a cursor over the input subwords and appends `source_ids[cursor]`
    # for each emitted COPY. This dramatically simplifies the decoder's
    # decision space — every prediction is now over a ~100-action vocab
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
        """Wrap `self.model` in a PeftModel with LoRA adapters + full-training
        of `peft_cfg.modules_to_save` (the embedding and lm_head's out_proj,
        by default). Verifies the requested `modules_to_save` names actually
        exist on the underlying model so a typo fails loud at init, not
        silently at train.

        Also restores tied-weight semantics that PEFT's modules_to_save breaks
        when wrapping tied modules — see `_retie_modules_to_save` for details.
        """
        from peft import LoraConfig, TaskType, get_peft_model

        existing_module_names = {name.rsplit(".", 1)[-1] for name, _ in self.model.named_modules()}
        missing = [m for m in peft_cfg.modules_to_save if m not in existing_module_names]
        if missing:
            raise ValueError(
                f"peft.modules_to_save references modules not found on {self.config.model_name!r}: "
                f"{missing}. Inspect `model.named_modules()` and update the config. Available leaf "
                f"names include embedding/projection candidates like: "
                f"{sorted(n for n in existing_module_names if 'embed' in n.lower() or 'head' in n.lower() or 'shared' in n.lower() or 'proj' in n.lower())}"
            )

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=peft_cfg.r,
            lora_alpha=peft_cfg.alpha,
            lora_dropout=peft_cfg.dropout,
            target_modules=peft_cfg.target_modules,
            bias=peft_cfg.bias,
            use_dora=peft_cfg.dora,
            modules_to_save=peft_cfg.modules_to_save,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self._retie_modules_to_save()

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

    def _retie_modules_to_save(self) -> None:
        """Restore tied-weight semantics broken by PEFT's modules_to_save.

        PEFT wraps each `modules_to_save` target in a `ModulesToSaveWrapper`
        that keeps the original (frozen) module at `.original_module` and an
        independent trainable copy at `.modules_to_save["default"]`. For
        modules whose originals shared storage — e.g. T5Gemma 2 ties
        `encoder.text_model.embed_tokens`, `decoder.embed_tokens`, and
        `lm_head.out_proj` to one weight tensor under
        `tie_word_embeddings=True` — PEFT silently de-ties them, creating
        N independent trainable copies of what's logically one tensor. This
        wastes ~(N-1) × tensor_size of weight + grad memory AND breaks the
        inductive bias the model was pretrained with (input/output embeddings
        decouple during fine-tuning).

        Here we group wrappers by their original `.weight.data_ptr()` (which
        reflects the pre-PEFT tying), then rebind every trainable copy in a
        group to share one canonical `nn.Parameter`. Backward accumulates
        gradients from all attachment points into that single tensor and
        the optimizer updates it once.
        """
        by_ptr: dict[int, list[tuple[str, nn.Module]]] = {}
        for name, mod in self.model.named_modules():
            if type(mod).__name__ != "ModulesToSaveWrapper":
                continue
            original = getattr(mod, "original_module", None)
            trainable_dict = getattr(mod, "modules_to_save", None)
            if original is None or trainable_dict is None or "default" not in trainable_dict:
                continue
            trainable = trainable_dict["default"]
            ow = getattr(original, "weight", None)
            tw = getattr(trainable, "weight", None)
            if not (isinstance(ow, nn.Parameter) and isinstance(tw, nn.Parameter)):
                continue
            by_ptr.setdefault(ow.data_ptr(), []).append((name, trainable))

        retied_groups: list[list[str]] = []
        for group in by_ptr.values():
            if len(group) < 2:
                continue
            _, canonical = group[0]
            for _, trainable in group[1:]:
                # Reassign the .weight Parameter so all trainable copies share
                # one tensor. nn.Module.__setattr__ detects nn.Parameter and
                # re-registers it under the same name.
                trainable.weight = canonical.weight
            retied_groups.append([name for name, _ in group])

        if retied_groups:
            logger.info(
                f"Re-tied trainable weight Parameters across modules_to_save "
                f"groups (preserving the backbone's tie_word_embeddings): {retied_groups}"
            )

    def _resolve_decoder_start_token_id(self) -> int:
        """Find the decoder start token in a model-family-portable way.
        T5/mT5 expose `decoder_start_token_id` on `config`; T5Gemma 2 only
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
            stub = torch.zeros((1, 1), dtype=torch.long, device=next(self.model.parameters()).device)
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
        # Snapshot original vocab size BEFORE adding new tokens — used by
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
        # at training time; head argmax indices map back to full IDs via
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
        PEFT applied to the original out_proj — we want this small head
        fully trainable, not low-rank-deltaed."""
        base = self._underlying_model()

        # Locate the existing output projection. T5Gemma 2 wraps it inside a
        # `T5Gemma2LMHead` module with an `out_proj` Linear. mT5 / T5 expose
        # `lm_head` directly as the Linear.
        # Warm-init each head row from the matching embed_tokens row. The
        # original lm_head was tied to embed_tokens, so row `full_id` of
        # embed_tokens is the "right" unembedding direction for token
        # `full_id`. Copying those rows into our small head means the model
        # starts already knowing which hidden direction maps to which token,
        # skipping the first chunk of training that would otherwise just
        # relearn that alignment. For action tokens whose embed row was
        # freshly created by `resize_token_embeddings`, the row is itself
        # randomly initialized, so this is no worse than the previous
        # N(0, 0.02) init for those entries (and strictly better for
        # pre-existing tokens like EOS).
        def _warm_init(new_linear: nn.Linear) -> None:
            with torch.no_grad():
                embed_weight = self._underlying_model().get_input_embeddings().weight
                for hi, full_id in enumerate(self.full_id_for_head_idx):
                    src = embed_weight[full_id].to(dtype=new_linear.weight.dtype, device=new_linear.weight.device)
                    new_linear.weight[hi].copy_(src)

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
            _warm_init(new)
            base.lm_head.out_proj = new
        elif hasattr(base, "lm_head") and isinstance(base.lm_head, nn.Linear):
            old = base.lm_head
            hidden = old.in_features
            new = nn.Linear(hidden, self.head_vocab_size, bias=False).to(
                dtype=old.weight.dtype, device=old.weight.device
            )
            _warm_init(new)
            base.lm_head = new
        else:
            raise RuntimeError(
                f"Don't know how to replace lm_head on {type(base).__name__}; expected "
                f"`lm_head` as Linear or `lm_head.out_proj` as Linear."
            )
        logger.info(
            f"Replaced lm_head with fresh Linear(hidden={hidden}, head_vocab_size={self.head_vocab_size}). "
            f"Logits tensor at training shrinks ~{old.out_features / self.head_vocab_size:.0f}× per token."
        )

    def _underlying_model(self):
        """Walk PEFT wrappers to reach the original HF model."""
        m = self.model
        if hasattr(m, "base_model"):
            m = m.base_model
        if hasattr(m, "model") and not isinstance(m, nn.ModuleList):
            m = m.model
        return m

    def _mask_old_embedding_gradients(self) -> None:
        """Restrict gradient updates on the trainable embed_tokens to ONLY
        the newly-added action-token rows. Old rows (the pretrained 262K
        vocabulary) stay frozen at their pretrained values; only the ~100
        new rows accumulate gradient. Drops the model's overfit surface
        area by ~600 MB of trainable parameter space at no quality cost
        (the old embeddings were already well-pretrained on trillions of
        tokens; fine-tuning them on ~150 GUM docs is mostly noise).

        Implementation: register a backward hook on the canonical trainable
        embedding Parameter (the one `_retie_modules_to_save` shared across
        the encoder/decoder embed wrappers). The hook zeros the gradient
        rows for indices < `self._original_vocab_size`. The optimizer
        therefore sees zero grad on those rows and doesn't update them.
        """
        canonical = self._find_canonical_embed_weight()
        if canonical is None:
            logger.info("_mask_old_embedding_gradients: no trainable embed_tokens Parameter found; nothing to mask.")
            return
        n_old = int(self._original_vocab_size)

        def _zero_old_rows(grad: torch.Tensor) -> torch.Tensor:
            out = grad.clone()
            out[:n_old].zero_()
            return out

        canonical.register_hook(_zero_old_rows)
        logger.info(
            f"Embed gradient mask: rows [0, {n_old}) frozen; only rows "
            f"[{n_old}, {canonical.shape[0]}) (the {canonical.shape[0] - n_old} action-token rows) update."
        )

        # Sanity check: every trainable embed Parameter in every
        # ModulesToSaveWrapper must share storage with `canonical`. If retie
        # missed a wrapper (e.g. early-continue path for a singleton group),
        # the hook on `canonical` would NOT cover that other Parameter and
        # old rows could leak gradient through it. Fail loudly instead of
        # silently mis-training.
        canonical_ptr = canonical.data_ptr()
        leaky_paths: list[str] = []
        for name, mod in self.model.named_modules():
            if type(mod).__name__ != "ModulesToSaveWrapper":
                continue
            trainable_dict = getattr(mod, "modules_to_save", None)
            if trainable_dict is None or "default" not in trainable_dict:
                continue
            trainable = trainable_dict["default"]
            if not isinstance(trainable, nn.Embedding):
                continue
            if trainable.weight.data_ptr() != canonical_ptr:
                leaky_paths.append(name)
        if leaky_paths:
            raise RuntimeError(
                "retie didn't catch all embed wrappers — gradient mask is "
                f"leaky on these paths: {leaky_paths}. Canonical embed at "
                f"data_ptr={canonical_ptr:x} is not shared by the above "
                "ModulesToSaveWrapper trainable copies."
            )

    def _verify_embedding_mask(self) -> None:
        """One-time sanity check: run a forward+backward on a tiny dummy
        batch, then assert that the gradient on the old-row indices of the
        canonical embed Parameter is exactly zero. Not meant for routine
        training; call manually (e.g. via `python -c`) to confirm the hook
        plumbed through correctly after init."""
        canonical = self._find_canonical_embed_weight()
        if canonical is None:
            logger.info("_verify_embedding_mask: no trainable embed Parameter; nothing to verify.")
            return
        n_old = int(self._original_vocab_size)
        device = canonical.device
        # Tiny dummy batch: feed a single short sequence of action tokens
        # (so they touch both old and new rows of the embed).
        input_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
        # Decoder input: a single new-vocab row (so the gradient on at least
        # one new row is non-zero, proving the hook isn't just zeroing
        # everything).
        new_row_id = n_old  # first new row
        decoder_input_ids = torch.tensor([[new_row_id]], dtype=torch.long, device=device)
        labels = torch.tensor([[0]], dtype=torch.long, device=device)
        self.zero_grad(set_to_none=True)
        out = self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        loss = out.loss if hasattr(out, "loss") and out.loss is not None else out.logits.sum()
        loss.backward()
        grad = canonical.grad
        if grad is None:
            raise RuntimeError("_verify_embedding_mask: canonical embed received no gradient at all.")
        old_grad_norm = grad[:n_old].abs().sum().item()
        new_grad_norm = grad[n_old:].abs().sum().item()
        if old_grad_norm != 0.0:
            raise RuntimeError(
                f"_verify_embedding_mask: old-row gradient is nonzero "
                f"(|grad[:{n_old}]|.sum() = {old_grad_norm}). The hook didn't fire on the right Parameter."
            )
        logger.info(
            f"_verify_embedding_mask OK: |grad[:{n_old}]|.sum() = 0, "
            f"|grad[{n_old}:]|.sum() = {new_grad_norm:.4e} (nonzero, as expected)."
        )
        self.zero_grad(set_to_none=True)

    def _find_canonical_embed_weight(self) -> torch.nn.Parameter | None:
        """Find the shared trainable embedding Parameter set up by
        `_retie_modules_to_save`. Returns None if no embed module is in
        `modules_to_save` (i.e., the user disabled the full-FT path)."""
        for name, mod in self.model.named_modules():
            if type(mod).__name__ != "ModulesToSaveWrapper":
                continue
            trainable_dict = getattr(mod, "modules_to_save", None)
            if trainable_dict is None or "default" not in trainable_dict:
                continue
            trainable = trainable_dict["default"]
            if isinstance(trainable, nn.Embedding):
                return trainable.weight
        return None

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
        # -100 for padding; the lookup buffer maps those to head indices
        # 0..head_vocab_size-1 (or -100 for anything not in the action
        # vocab, defensive — shouldn't normally happen).
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

        base_loss = F.cross_entropy(
            logits.reshape(-1, self.head_vocab_size).float(),
            head_labels_flat,
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )
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
    # Action-aware tokenization (used by both training and inference)
    # -----------------------------------------------------------------

    def encode_input(self, text: str) -> dict[str, list[int]]:
        """Encode the raw document for the encoder side. Shrieks if the
        tokenizer truncates — silent input truncation corrupts the training
        signal (the target still references EDUs whose source tokens the
        encoder never saw) and silently degrades inference (the model can't
        parse anything past the cut)."""
        # Untruncated length is our truncation detector. Two tokenizer calls
        # is the simplest reliable signal; the second is bounded by
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
                f"Bump max_input_length (model supports up to ~32K for T5Gemma 2; "
                f"~1K-2K for mT5) or this doc's tail is invisible to the model."
            )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def encode_target(self, tree: RstTree) -> tuple[list[int], list[int]] | None:
        """Build two aligned target streams:

          * `labels`: the prediction targets — `<copy>` at source-copy
            positions, `<shift>` and `<reduce_*>` at structural positions,
            `<eos>` at the end. The decoder is trained to PREDICT these.

          * `decoder_input_ids`: what the decoder ACTUALLY SEES in its
            self-attention history — `<copy>` is replaced by the actual
            source subword ID, so the decoder's input distribution at
            training matches inference (where we substitute before
            appending). Length-aligned with `labels`: position i is
            shift-right of label i (with decoder_start at position 0).

        Length is identical to the previous "actual source subword IDs"
        formulation, but the model's prediction vocabulary at every position
        is now just the ~100 action tokens instead of the 262K full vocab —
        the deterministic copy task no longer competes with structural
        decisions for gradient.

        Returns None when the target overflows `cfg.max_output_length`."""
        if self.shift_token_id is None:
            raise RuntimeError(
                "encode_target called before action vocab was installed; did you forget to set cfg.relation_types?"
            )
        actions = tree.to_shift_reduce(include_text=False)
        # Tokenize the whole document once with offsets, then slice each EDU's
        # subword IDs by its character range. Per-EDU tokenization drifts from
        # whole-doc tokenization on SentencePiece-family tokenizers (whitespace-
        # sensitive), which desyncs the training-time COPY substitutions from
        # the inference-time `source_ids` stream. Mirrors `_gold_edu_token_mapping`
        # in train_seq2seq_sr.py so training and inference see the same token
        # space the encoder sees at predict time.
        text = _reconstruct_text(tree)
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        full_input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        edu_subword_ids: list[list[int]] = []
        char_cursor = 0
        for i, edu in enumerate(tree.edus):
            if i > 0:
                prefix = edu.prefix if edu.prefix is not None else " "
                char_cursor += len(prefix)
            char_start = char_cursor
            char_cursor += len(edu.text)
            char_end = char_cursor
            first: int | None = None
            last: int | None = None
            for j, (tok_cs, tok_ce) in enumerate(offsets):
                if tok_cs < char_end and tok_ce > char_start:
                    if first is None:
                        first = j
                    last = j
            if first is None or last is None:
                edu_subword_ids.append([])
            else:
                edu_subword_ids.append(list(full_input_ids[first : last + 1]))

        label_ids: list[int] = []
        seen_ids: list[int] = []  # what the decoder sees in its history (substituted)
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
        label_ids.append(self.tokenizer.eos_token_id)
        seen_ids.append(self.tokenizer.eos_token_id)

        # decoder_input_ids = shift_right(seen_ids): decoder_start at pos 0,
        # then seen_ids[:-1]. Same length as labels.
        decoder_input_ids = [self.decoder_start_token_id] + seen_ids[:-1]

        if len(label_ids) > self.config.max_output_length:
            warn(
                f"Target truncated: {len(label_ids)} > max_output_length="
                f"{self.config.max_output_length} for a {len(tree.edus)}-EDU tree. "
                f"Tree DROPPED from this epoch. Bump max_output_length (model supports "
                f"up to 32K for T5Gemma 2)."
            )
            return None
        return label_ids, decoder_input_ids

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

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
        `cfg.eval_decode_greedy`); beam is used for the final eval (via
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
        rows in parallel; the validity mask + COPY substitution work per-row."""
        self.eval()
        device = next(self.parameters()).device

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
        # cursor tracks ONLY the content subwords — matching what
        # `encode_target` substitutes at training (per-EDU tokenization with
        # `add_special_tokens=False`). T5/mT5 add EOS at the tail; T5Gemma 2
        # prepends BOS at the head. Either or both can be present.
        bos_id = self.tokenizer.bos_token_id
        per_row_source_ids: list[list[int]] = []
        empty_rows: list[int] = []
        for i in range(enc["input_ids"].shape[0]):
            ids = enc["input_ids"][i].tolist()
            while ids and ids[-1] == pad_id:
                ids.pop()
            if ids and ids[-1] == eos_id:
                ids.pop()
            if bos_id is not None and ids and ids[0] == bos_id:
                ids = ids[1:]
            full_len = len(self.tokenizer(texts[i], add_special_tokens=False).input_ids)
            if full_len > self.config.max_input_length - 1:
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

            # Per-row state
            cursors = [0] * B
            stacks = [0] * B
            # Number of COPYs in the currently-building EDU. Replaces the
            # earlier `edu_has_content` flag so we can enforce
            # `min_edu_length` (each shift requires the EDU to have at
            # least that many copies, except at end-of-source).
            edu_lengths = [0] * B
            done = [i in empty_rows for i in range(B)]
            action_seqs: list[list[int]] = [[] for _ in range(B)]
            hit_max_len = [False] * B
            # Per-row pred EDU ranges in source_ids token space (the same
            # space as the encoder's tokenization). Tracked directly off
            # the cursor here so eval metrics can use these without going
            # through a lossy decode→split→re-tokenize round trip.
            pred_edu_ranges: list[list[tuple[int, int]]] = [[] for _ in range(B)]
            edu_starts = [0] * B
            min_edu_len = max(1, int(self.config.min_edu_length))

            # Decoder input + KV cache
            decoder_input_ids = torch.full((B, 1), decoder_start, device=device, dtype=torch.long)
            past_key_values = None

            for step in range(self.config.max_output_length):
                if all(done):
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
                # [B, 262K] — the replaced lm_head emits only action logits.
                logits = out.logits[:, -1, :]  # [B, head_vocab_size]

                # Per-row validity mask: only legal HEAD INDICES survive.
                if self.config.use_validity_constraints:
                    masked = torch.full_like(logits, float("-inf"))
                    for i in range(B):
                        if done[i]:
                            continue
                        source_len = len(per_row_source_ids[i])
                        at_end = cursors[i] >= source_len
                        # COPY legal iff cursor < source_len
                        if not at_end:
                            masked[i, self.copy_head_idx] = logits[i, self.copy_head_idx]
                        # SHIFT legal iff the current EDU has at least
                        # `min_edu_length` COPYs — OR we're at end-of-source
                        # with any content (need to commit the final EDU).
                        shift_ok = edu_lengths[i] >= min_edu_len or (at_end and edu_lengths[i] >= 1)
                        if shift_ok:
                            masked[i, self.shift_head_idx] = logits[i, self.shift_head_idx]
                        # REDUCE-* legal iff stack >= 2
                        if stacks[i] >= 2:
                            masked[i, self._reduce_head_ids_buf] = logits[i, self._reduce_head_ids_buf]
                        # EOS legal iff cursor at end, stack singleton, no pending content
                        if at_end and stacks[i] == 1 and edu_lengths[i] == 0:
                            masked[i, self.eos_head_idx] = logits[i, self.eos_head_idx]
                    logits = masked

                # Greedy in head-vocab space.
                next_head_indices = logits.argmax(-1).tolist()  # [B]

                # Update state per row + compute next decoder-input token.
                # The decoder INPUT is in full vocab (source-subword IDs get
                # substituted at COPY), so we map head index → full ID here.
                next_inputs = [pad_id] * B
                for i, head_idx in enumerate(next_head_indices):
                    if done[i]:
                        continue
                    full_id = self.full_id_for_head_idx[head_idx]
                    action_seqs[i].append(full_id)
                    if full_id == eos_id:
                        done[i] = True
                    elif full_id == self.copy_token_id:
                        if cursors[i] < len(per_row_source_ids[i]):
                            next_inputs[i] = per_row_source_ids[i][cursors[i]]
                            cursors[i] += 1
                            edu_lengths[i] += 1
                        else:
                            # Constraint should have prevented this; bail.
                            done[i] = True
                    elif full_id == self.shift_token_id:
                        stacks[i] += 1
                        edu_lengths[i] = 0
                        # Record this EDU's source-position range.
                        pred_edu_ranges[i].append((edu_starts[i], cursors[i]))
                        edu_starts[i] = cursors[i]
                        next_inputs[i] = full_id
                    elif full_id in self.reduce_token_ids:
                        stacks[i] -= 1
                        next_inputs[i] = full_id
                    else:
                        # Shouldn't happen under valid mask.
                        done[i] = True

                if all(done):
                    break

                new_step = torch.tensor(next_inputs, device=device, dtype=torch.long).unsqueeze(1)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
            else:
                # for-loop exhausted without `break`: max_output_length hit.
                for i in range(B):
                    if not done[i]:
                        hit_max_len[i] = True
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        # Tree reconstruction per row.
        results: list[RstTree] = []
        empty_set = set(empty_rows)
        for i in range(B):
            if i in empty_set:
                results.append(_empty_tree(self.config.relation_types))
                continue
            if hit_max_len[i]:
                warn(
                    f"Output truncated at inference (batch row {i}): generation hit "
                    f"max_output_length={self.config.max_output_length} without EOS. "
                    f"Tree closed by best-effort repair."
                )
            # If decoding stopped mid-EDU (uncommitted COPYs since the last
            # SHIFT), record the in-flight (start, cursor) range so seg eval
            # sees the truncated EDU. `_repair_actions` independently appends
            # a synthetic <shift> for trailing source tokens, so the action
            # sequence and pred_edu_ranges stay consistent.
            if cursors[i] > edu_starts[i]:
                pred_edu_ranges[i].append((edu_starts[i], cursors[i]))
            tree = self._tree_from_action_sequence(action_seqs[i], per_row_source_ids[i])
            # Stash per-EDU source-position ranges on the tree so eval can
            # use them without re-tokenizing (which drifts vs the encoder's
            # whole-doc tokenization). Side-channel via a `_meta` attribute
            # since RstTree doesn't natively carry this.
            tree._pred_edu_source_ranges = pred_edu_ranges[i]  # type: ignore[attr-defined]
            tree._source_ids = per_row_source_ids[i]  # type: ignore[attr-defined]
            results.append(tree)
        return results

    @torch.no_grad()
    def _predict_one_beam(self, text: str, num_beams: int) -> RstTree:
        """Beam search for a single document with `num_beams` parallel beams.
        Each beam carries its own (cursor, stack, edu_length, action_seq,
        pred_edu_ranges, edu_start) state. The encoder runs once; the
        decoder forward at each step processes all K beams in parallel
        (one batch of size K). KV cache is reordered when beams change
        parents via `self.model._reorder_cache`.

        Done per-document rather than batched across documents so memory
        stays bounded by K beams regardless of dev_batch_size. Wall-time
        roughly K× single-doc greedy."""
        self.eval()
        device = next(self.parameters()).device
        K = int(num_beams)
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        bos_id = self.tokenizer.bos_token_id
        decoder_start = self.decoder_start_token_id
        head_V = self.head_vocab_size
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
        ids = enc["input_ids"][0].tolist()
        while ids and ids[-1] == pad_id:
            ids.pop()
        if ids and ids[-1] == eos_id:
            ids.pop()
        if bos_id is not None and ids and ids[0] == bos_id:
            ids = ids[1:]
        full_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        if full_len > self.config.max_input_length - 1:
            warn(
                f"Input truncated (beam): {full_len} -> ~{self.config.max_input_length} subwords. "
                f"Bump max_input_length."
            )
        source_ids = ids
        if not source_ids:
            return _empty_tree(self.config.relation_types)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)
        try:
            # Encoder pass — once for the single doc, then expand to K beams.
            encoder = self.model.get_encoder()
            enc_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)
            from transformers.modeling_outputs import BaseModelOutput

            expanded_hidden = enc_out.last_hidden_state.expand(K, -1, -1).contiguous()
            expanded_attn = enc["attention_mask"].expand(K, -1).contiguous()
            enc_out_K = BaseModelOutput(last_hidden_state=expanded_hidden)

            # Per-beam state.
            cursors = [0] * K
            stacks = [0] * K
            edu_lengths = [0] * K
            done = [False] * K
            action_seqs: list[list[int]] = [[] for _ in range(K)]
            pred_edu_ranges: list[list[tuple[int, int]]] = [[] for _ in range(K)]
            edu_starts = [0] * K
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
            source_len = len(source_ids)

            for step in range(self.config.max_output_length):
                if all(done):
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
                masked = torch.full_like(logits, float("-inf"))
                for j in range(K):
                    if done[j]:
                        # Done beams are frozen in `finished_beams`; their
                        # row stays all -inf so they can never re-enter the
                        # top-K and crowd out active beams.
                        continue
                    at_end = cursors[j] >= source_len
                    if not at_end:
                        masked[j, self.copy_head_idx] = logits[j, self.copy_head_idx]
                    shift_ok = edu_lengths[j] >= min_edu_len or (at_end and edu_lengths[j] >= 1)
                    if shift_ok:
                        masked[j, self.shift_head_idx] = logits[j, self.shift_head_idx]
                    if stacks[j] >= 2:
                        masked[j, self._reduce_head_ids_buf] = logits[j, self._reduce_head_ids_buf]
                    if at_end and stacks[j] == 1 and edu_lengths[j] == 0:
                        masked[j, self.eos_head_idx] = logits[j, self.eos_head_idx]
                log_probs = F.log_softmax(masked.float(), dim=-1)
                # Add cumulative scores: [K, head_V]
                cum = beam_scores.unsqueeze(1) + log_probs
                # Top-K continuations from K beams × head_V actions.
                top_scores, top_idx = cum.view(-1).topk(K)
                parent_of_new = (top_idx // head_V).tolist()
                action_of_new = (top_idx % head_V).tolist()

                # Reorder KV cache by parent_of_new. HF model layouts vary:
                # (a) `_reorder_cache` on the underlying model (T5/T5Gemma2),
                # (b) a DynamicCache with `reorder_cache` method (newer HF),
                # (c) tuple-of-tuple of Tensors with possible None entries
                #     (cross-attn slots that aren't populated yet).
                parent_tensor = torch.tensor(parent_of_new, device=device, dtype=torch.long)
                if past_key_values is not None:
                    past_key_values = _reorder_pkv(past_key_values, parent_tensor, self._underlying_model())
                # Reorder decoder_input_ids by parent.
                decoder_input_ids = decoder_input_ids[parent_tensor]

                # Carry per-beam state from parents.
                new_cursors = [cursors[p] for p in parent_of_new]
                new_stacks = [stacks[p] for p in parent_of_new]
                new_edu_lengths = [edu_lengths[p] for p in parent_of_new]
                new_done = [done[p] for p in parent_of_new]
                new_action_seqs = [list(action_seqs[p]) for p in parent_of_new]
                new_pred_edu_ranges = [list(pred_edu_ranges[p]) for p in parent_of_new]
                new_edu_starts = [edu_starts[p] for p in parent_of_new]

                # Apply each beam's chosen action.
                next_inputs = [pad_id] * K
                for j in range(K):
                    if new_done[j]:
                        continue
                    head_idx = action_of_new[j]
                    full_id = self.full_id_for_head_idx[head_idx]
                    new_action_seqs[j].append(full_id)
                    if full_id == eos_id:
                        new_done[j] = True
                    elif full_id == self.copy_token_id:
                        if new_cursors[j] < source_len:
                            next_inputs[j] = source_ids[new_cursors[j]]
                            new_cursors[j] += 1
                            new_edu_lengths[j] += 1
                        else:
                            new_done[j] = True
                    elif full_id == self.shift_token_id:
                        new_stacks[j] += 1
                        new_edu_lengths[j] = 0
                        new_pred_edu_ranges[j].append((new_edu_starts[j], new_cursors[j]))
                        new_edu_starts[j] = new_cursors[j]
                        next_inputs[j] = full_id
                    elif full_id in self.reduce_token_ids:
                        new_stacks[j] -= 1
                        next_inputs[j] = full_id
                    else:
                        new_done[j] = True

                cursors, stacks, edu_lengths = new_cursors, new_stacks, new_edu_lengths
                done = new_done
                action_seqs = new_action_seqs
                pred_edu_ranges = new_pred_edu_ranges
                edu_starts = new_edu_starts
                beam_scores = top_scores

                # Snapshot any newly-finished beams into `finished_beams`,
                # then freeze their active score to -inf so they can't win
                # top-K again. Active beams keep their score and state.
                for j in range(K):
                    if done[j] and torch.isfinite(beam_scores[j]):
                        finished_beams.append(
                            {
                                "action_seq": list(action_seqs[j]),
                                "pred_edu_ranges": list(pred_edu_ranges[j]),
                                "score": float(beam_scores[j].item()),
                                "length": len(action_seqs[j]),
                                "cursor": cursors[j],
                                "edu_start": edu_starts[j],
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
        for j in range(K):
            if not done[j] and torch.isfinite(beam_scores[j]):
                # Active beam: pred_edu_ranges may be missing an in-flight
                # EDU if max_output_length cut us off mid-EDU. Mirror the
                # greedy-path fix so seg eval sees the truncated EDU. The
                # action sequence gets a synthetic <shift> via
                # `_repair_actions` downstream; appending here keeps the
                # two data structures consistent.
                ranges = list(pred_edu_ranges[j])
                if cursors[j] > edu_starts[j]:
                    ranges.append((edu_starts[j], cursors[j]))
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
            return _empty_tree(self.config.relation_types)

        # Length-normalized scoring: dividing cumulative sum log-prob by
        # length**alpha mitigates the systematic bias toward shorter
        # beams (every emitted token has log-prob <= 0, so sum-log-prob
        # monotonically favors fewer-token = fewer-EDU trajectories).
        # alpha=0.6 is the GNMT default (alpha=1.0 is mean log-prob, also
        # defensible; 0.6 is the standard mitigation).
        length_penalty_alpha = 0.6
        best = max(
            candidates,
            key=lambda c: c["score"] / max(c["length"], 1) ** length_penalty_alpha,
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
        positions. The model still freely chooses every `<reduce_*>`, so
        binarization + labeling stay model-driven; only segmentation is
        supplied. Used by training-time eval when `cfg.eval_gold_edu`."""
        return self._predict_one_gold_edu(tree)

    @torch.no_grad()
    def _predict_one_gold_edu(self, tree: RstTree) -> RstTree:
        self.eval()
        device = next(self.parameters()).device

        text = _reconstruct_text(tree)
        # Re-derive the gold EDU spans in source-id space from the same
        # whole-doc tokenization the encoder will see. Duplicates the small
        # helper in train_seq2seq_sr.py rather than importing across files
        # (and keeps this path runnable without the trainer being loaded).
        gold_ranges = _gold_edu_source_ranges(self.tokenizer, tree)

        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        bos_id = self.tokenizer.bos_token_id

        ids = enc["input_ids"][0].tolist()
        while ids and ids[-1] == pad_id:
            ids.pop()
        if ids and ids[-1] == eos_id:
            ids.pop()
        if bos_id is not None and ids and ids[0] == bos_id:
            ids = ids[1:]
        source_ids = ids
        source_len = len(source_ids)
        if not source_ids:
            return _empty_tree(self.config.relation_types)

        # Clamp gold ranges to the (possibly truncated) source length. An EDU
        # whose start fell beyond truncation is dropped; one straddling the
        # boundary gets shortened.
        clamped_ranges: list[tuple[int, int]] = []
        for s, e in gold_ranges:
            if s >= source_len:
                break
            clamped_ranges.append((s, min(e, source_len)))
        if not clamped_ranges:
            return _empty_tree(self.config.relation_types)
        edu_ends = [end for _, end in clamped_ranges]  # exclusive ends
        n_edus = len(edu_ends)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)
        try:
            encoder = self.model.get_encoder()
            enc_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)

            decoder_start = self.decoder_start_token_id
            cursor = 0
            stack = 0
            edu_idx = 0
            in_edu_buffer = 0  # copies since last shift (or start)
            action_seq: list[int] = []
            pred_edu_ranges: list[tuple[int, int]] = []
            edu_start = 0
            done = False
            hit_max_len = False

            decoder_input_ids = torch.full((1, 1), decoder_start, device=device, dtype=torch.long)
            past_key_values = None

            for step in range(self.config.max_output_length):
                if done:
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

                if more_edus and cursor < current_end:
                    # Inside the current gold EDU: force COPY.
                    masked[self.copy_head_idx] = logits[self.copy_head_idx]
                elif more_edus and cursor == current_end and in_edu_buffer > 0:
                    # Reached the boundary with content buffered: force SHIFT.
                    masked[self.shift_head_idx] = logits[self.shift_head_idx]
                else:
                    # Between EDUs (just shifted, buffer empty) or all EDUs
                    # exhausted. The model freely chooses REDUCE vs the next
                    # structural step.
                    if stack >= 2:
                        masked[self._reduce_head_ids_buf] = logits[self._reduce_head_ids_buf]
                    if more_edus:
                        # Start of the next EDU: only COPY moves forward.
                        masked[self.copy_head_idx] = logits[self.copy_head_idx]
                    else:
                        if stack == 1:
                            masked[self.eos_head_idx] = logits[self.eos_head_idx]

                head_idx = int(masked.argmax(-1).item())
                full_id = self.full_id_for_head_idx[head_idx]
                action_seq.append(full_id)

                if full_id == eos_id:
                    done = True
                    next_input = pad_id
                elif full_id == self.copy_token_id:
                    if cursor < source_len:
                        next_input = source_ids[cursor]
                        cursor += 1
                        in_edu_buffer += 1
                    else:
                        done = True
                        next_input = pad_id
                elif full_id == self.shift_token_id:
                    stack += 1
                    pred_edu_ranges.append((edu_start, cursor))
                    edu_start = cursor
                    in_edu_buffer = 0
                    edu_idx += 1
                    next_input = full_id
                elif full_id in self.reduce_token_ids:
                    stack -= 1
                    next_input = full_id
                else:
                    done = True
                    next_input = pad_id

                if done:
                    break
                new_step = torch.tensor([[next_input]], device=device, dtype=torch.long)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
            else:
                hit_max_len = not done
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        if hit_max_len:
            warn(
                f"Output truncated at inference (gold-edu): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS. "
                f"Tree closed by best-effort repair."
            )
        if cursor > edu_start:
            pred_edu_ranges.append((edu_start, cursor))
        tree_out = self._tree_from_action_sequence(action_seq, source_ids)
        tree_out._pred_edu_source_ranges = pred_edu_ranges  # type: ignore[attr-defined]
        tree_out._source_ids = source_ids  # type: ignore[attr-defined]
        return tree_out

    @torch.no_grad()
    def predict(self, tree: RstTree, *, num_beams: int | None = None) -> RstTree:
        """Reconstruct document text from the gold tree's EDUs, then parse
        end-to-end. The parser does not consume gold EDU boundaries."""
        text = _reconstruct_text(tree)
        return self.predict_from_text(text, num_beams=num_beams)

    @torch.no_grad()
    def predict_batch(
        self,
        trees: list[RstTree],
        *,
        num_beams: int | None = None,
    ) -> list[RstTree]:
        """Batched analogue of `predict(tree)`. Reconstructs document text
        per tree, then runs a single batched `generate()`."""
        texts = [_reconstruct_text(t) for t in trees]
        return self.predict_batch_from_texts(texts, num_beams=num_beams)

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

        actions, malformed_reason = self._repair_actions(strings)
        if malformed_reason is not None:
            warn(
                f"Malformed decoder output ({malformed_reason}); falling back to "
                f"single-EDU tree. Likely an undertrained model or max_output_length too low."
            )
            full_text = " ".join(s for s in strings if not (s == "<shift>" or s in self.reduce_token_map))
            return _empty_tree(self.config.relation_types, text=full_text)
        return RstTree.from_shift_reduce(actions, relation_types=self.config.relation_types)

    def _repair_actions(self, strings: list[str]) -> tuple[list[ShiftReduceAction], str | None]:
        """Try `strings_to_actions` on the raw string list. If trailing
        source tokens are present, append a closing `<shift>` and the right
        number of fallback reduces to drain the stack. Returns the action
        list plus a reason if the sequence had to be repaired, None if it
        parsed cleanly."""
        try:
            actions = strings_to_actions(strings, self.reduce_token_map)
        except ValueError:
            # Trailing source tokens: append a closing <shift>, then we'll
            # check stack-size against the resulting Shift count and add
            # reduces below.
            repaired = list(strings) + [Shift().to_token()]
            try:
                actions = strings_to_actions(repaired, self.reduce_token_map)
            except ValueError as e:
                return [], str(e)
            # Need to drain the stack. Add NS-elaboration-ish fallback reduces.
            n_shifts = sum(1 for a in actions if isinstance(a, Shift))
            n_reduces = sum(1 for a in actions if isinstance(a, Reduce))
            needed = (n_shifts - 1) - n_reduces
            if needed > 0:
                fallback = self._fallback_reduce()
                if fallback is None:
                    return actions, "no fallback reduce token available"
                actions = list(actions) + [fallback] * needed
            return actions, "max_length hit mid-EDU; appended closing shift/reduces"
        # No exception: still need to verify shift/reduce balance.
        n_shifts = sum(1 for a in actions if isinstance(a, Shift))
        n_reduces = sum(1 for a in actions if isinstance(a, Reduce))
        if n_shifts == 0:
            return actions, "no shifts in generated sequence"
        if n_reduces != n_shifts - 1:
            needed = (n_shifts - 1) - n_reduces
            if needed < 0:
                return actions, f"too many reduces ({n_reduces}) for {n_shifts} shifts"
            fallback = self._fallback_reduce()
            if fallback is None:
                return actions, "stack underdrained and no fallback reduce available"
            return list(actions) + [fallback] * needed, "stack underdrained; appended closing reduces"
        return actions, None

    def _fallback_reduce(self) -> "Reduce | None":
        """A Reduce action we can use to close an unfinished tree. Prefers
        NS-elaboration if available; falls back to the first reduce in the
        vocabulary."""
        for token_str, (nuc, rel) in self.reduce_token_map.items():
            if (nuc, rel) == ("NS", "elaboration"):
                return Reduce(nuc=nuc, rel=rel)
        for token_str, (nuc, rel) in self.reduce_token_map.items():
            return Reduce(nuc=nuc, rel=rel)
        return None


def _reconstruct_text(tree: RstTree) -> str:
    """Reverse the storage convention: join EDU strings with spaces (or
    each EDU's `prefix` field if populated, for detokenized corpora)."""
    parts: list[str] = []
    for i, edu in enumerate(tree.edus):
        if i == 0:
            parts.append(edu.text)
            continue
        prefix = edu.prefix if edu.prefix is not None else " "
        parts.append(prefix + edu.text)
    return "".join(parts)


def _gold_edu_source_ranges(tokenizer, tree: RstTree) -> list[tuple[int, int]]:
    """Per-EDU `(start, end_exclusive)` token-position ranges in the encoder's
    whole-doc tokenization space. Mirrors `_gold_edu_token_mapping` in
    train_seq2seq_sr.py — duplicated so the predict path doesn't pull the
    trainer into module-load."""
    text = _reconstruct_text(tree)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    mapping: list[tuple[int, int]] = []
    char_cursor = 0
    for i, edu in enumerate(tree.edus):
        if i > 0:
            prefix = edu.prefix if edu.prefix is not None else " "
            char_cursor += len(prefix)
        char_start = char_cursor
        char_cursor += len(edu.text)
        char_end = char_cursor
        first: int | None = None
        last: int | None = None
        for j, (tok_cs, tok_ce) in enumerate(offsets):
            if tok_cs < char_end and tok_ce > char_start:
                if first is None:
                    first = j
                last = j
        if first is None or last is None:
            anchor = first if first is not None else max(0, len(offsets) - 1)
            mapping.append((anchor, anchor))
            continue
        mapping.append((first, last + 1))
    return mapping


def _empty_tree(relation_types, text: str = "") -> RstTree:
    # Single-EDU fallback for empty / unrecoverable input. The text payload
    # becomes one EDU so downstream callers (to_rs4_string, eval) work.
    actions: list[ShiftReduceAction] = [Shift(edu_text=text or "")]
    return RstTree.from_shift_reduce(actions, relation_types=relation_types)


def _reorder_pkv(past_key_values, beam_idx: torch.Tensor, underlying_model):
    """Reorder a HF past_key_values cache along the beam dimension. Handles
    three layouts:
      1. Underlying model exposes `_reorder_cache(pkv, beam_idx)` (T5/T5Gemma2
         and most HF seq2seq models).
      2. `past_key_values` is a `DynamicCache`-like object with its own
         `reorder_cache` method (newer transformers).
      3. Tuple-of-tuple of Tensors (older HF), possibly with `None` entries
         for unfilled cross-attention slots.
    """
    # Path 1: canonical HF helper on the base model. T5Gemma 2's inherited
    # `_reorder_cache` assumes the legacy tuple-of-tuple layout; newer HF
    # versions may hand us a DynamicCache instead, which makes that call
    # blow up. Catch and fall through to the next path on type/attribute
    # mismatches.
    reorder = getattr(underlying_model, "_reorder_cache", None)
    if callable(reorder):
        try:
            return reorder(past_key_values, beam_idx)
        except (TypeError, AttributeError) as e:
            import warnings

            warnings.warn(
                f"{type(underlying_model).__name__}._reorder_cache failed on "
                f"{type(past_key_values).__name__} ({type(e).__name__}: {e}); "
                "falling back to object/tuple cache reordering.",
                stacklevel=2,
            )
    # Path 2: DynamicCache or similar object-style cache.
    if hasattr(past_key_values, "reorder_cache"):
        return past_key_values.reorder_cache(beam_idx)
    # Path 3: manual tuple walk; handle Nones gracefully.
    return tuple(
        tuple(t.index_select(0, beam_idx) if isinstance(t, torch.Tensor) else t for t in layer)
        for layer in past_key_values
    )
