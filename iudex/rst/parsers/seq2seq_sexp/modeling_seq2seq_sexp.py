"""End-to-end RST parser via a fine-tuned encoder-decoder LM that emits
a nested s-expression serialization of the binary RST tree, with source
tokens substituted via `<copy>` (use_copy=True) or interleaved verbatim
(use_copy=False). Sibling of `seq2seq_sr`, sharing its training/decoding
recipe but swapping flat shift-reduce for nested s-expressions.

Action vocabulary (use_copy=True):
    {<sexp_open>, <sexp_close>, <eos>} U {<reduce_ns_*>, <reduce_sn_*>,
    <reduce_nn_*>} U {<copy>}

In use_copy=False mode the `<copy>` token is dropped and the source
subwords appear in-stream as their actual tokenizer IDs. We keep
`use_copy=True` as the canonical / well-tested path (small action head,
installed by `_install_action_head`). The `use_copy=False` branch is
supported in `encode_target` and decoding but keeps the full backbone
lm_head (no head replacement) so the source vocab is naturally
available for source-token prediction. This trades the memory benefit
of the small head for keeping source-token prediction working without
enumerating the source vocab at init.
"""

import dataclasses
import logging
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from iudex.common.log import warn
from iudex.rst.data.tree import Reduce, RstTree
from iudex.rst.parsers.common.seqgen import (
    align_edus_to_tokens,
    beam_reorder_needed,
    beam_topk_step,
    empty_tree,
    gold_edu_source_ranges,
    mask_old_embedding_gradients,
    reconstruct_text,
    reorder_past_key_values,
    select_best_beam,
    warm_init_head,
)
from iudex.rst.parsers.common.sexp_constraints import (
    FORCE_CONTENT,
    GoldEduForcer,
    SexpDecodingState,
)
from iudex.rst.parsers.seq2seq_sexp.configuration_seq2seq_sexp import Seq2SeqSexpConfig

logger = logging.getLogger(__name__)


class Seq2SeqSexpParser(nn.Module):
    # Structural special tokens added to the tokenizer. New, not SentencePiece
    # `(` / `)`, to avoid the family's whitespace/escape quirks around bare
    # punctuation.
    OPEN_TOKEN: str = "<sexp_open>"
    CLOSE_TOKEN: str = "<sexp_close>"
    COPY_TOKEN: str = "<copy>"

    def __init__(self, config: Seq2SeqSexpConfig, *, compile_encoder: bool = False):
        super().__init__()
        self.config = config
        # compile_encoder is accepted for parser-CLI uniformity but has no
        # effect here. The HF seq2seq model has its own compilation story.
        del compile_encoder

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # bf16 master weights are fine under LoRA (adapters train on top) but
        # AdamW full-FT on bf16 weights degenerates (sub-resolution updates
        # round away). Full FT loads fp32, the autocast in training stays bf16.
        model_dtype = torch.bfloat16 if (config.amp and config.peft is not None) else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, dtype=model_dtype)

        self.label_token_ids: dict[str, int] = {}
        self.label_id_set: set[int] = set()
        self.label_token_map: dict[str, Tuple[str, str]] = {}
        if config.relation_types is not None:
            self._install_label_vocab()

        if config.peft is not None:
            self._install_peft(config.peft)

        if config.relation_types is not None and config.use_copy:
            self._install_action_head()
            # use_copy=True: small action head, so the input embedding keeps the
            # full matrix trainable while the shared gradient-mask helper zeroes
            # pretrained-row gradients (only the new structural/label rows
            # update). It never overrides the embedding forward, so backbone-
            # specific behavior like Gemma scaling is preserved. use_copy=False
            # keeps the full tied lm_head, which must train all embedding rows to
            # score source ids, so that path stays on the PEFT modules_to_save
            # scheme. See `mask_old_embedding_gradients`.
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

    def _build_label_vocab(self) -> List[str]:
        """Specials + per-(rel, kind) reduces (same `<reduce_*>` tokens as
        seq2seq_sr) + optional copy token."""
        assert self.config.relation_types is not None
        reduces: list[str] = []
        self.label_token_map = {}
        for rel, kind in self.config.relation_types:
            nucs = ("NN",) if kind == "multinuc" else ("NS", "SN")
            for nuc in nucs:
                token = Reduce(nuc=nuc, rel=rel).to_token()
                reduces.append(token)
                self.label_token_map[token] = (nuc, rel)
        tokens = [self.OPEN_TOKEN, self.CLOSE_TOKEN] + reduces
        if self.config.use_copy:
            tokens.append(self.COPY_TOKEN)
        return tokens

    def _install_peft(self, peft_cfg) -> None:
        """Wrap in LoRA adapters. Under `use_copy=True` the input embedding is
        kept OUT of PEFT modules_to_save (the single embedding stays trainable
        with pretrained-row gradients zeroed via `mask_old_embedding_gradients`).
        Under `use_copy=False` the full tied lm_head must learn source ids, so
        the embedding is wrapped in modules_to_save and re-tied like before."""
        from peft import LoraConfig, TaskType, get_peft_model

        use_modules_to_save = not self.config.use_copy
        modules_to_save = peft_cfg.modules_to_save if use_modules_to_save else None
        if use_modules_to_save:
            existing_module_names = {name.rsplit(".", 1)[-1] for name, _ in self.model.named_modules()}
            missing = [m for m in peft_cfg.modules_to_save if m not in existing_module_names]
            if missing:
                raise ValueError(
                    f"peft.modules_to_save references modules not found on {self.config.model_name!r}: "
                    f"{missing}. Available leaf names include: "
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
            modules_to_save=modules_to_save,
        )
        self.model = get_peft_model(self.model, lora_cfg)
        if use_modules_to_save:
            self._retie_modules_to_save()

    def _set_grad_checkpointing(self, enabled: bool) -> None:
        method = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        fn = getattr(self.model, method, None)
        if callable(fn):
            fn()
            return
        for mod in self.model.modules():
            if hasattr(mod, "gradient_checkpointing"):
                mod.gradient_checkpointing = enabled

    def _retie_modules_to_save(self) -> None:
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
                trainable.weight = canonical.weight
            retied_groups.append([name for name, _ in group])
        if retied_groups:
            logger.info(f"Re-tied trainable weight Parameters across modules_to_save groups: {retied_groups}")

    def _resolve_decoder_start_token_id(self) -> int:
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

    def _install_label_vocab(self) -> None:
        action_vocab = self._build_label_vocab()
        self._original_vocab_size = len(self.tokenizer)
        existing = set(self.tokenizer.get_vocab().keys())
        new_tokens = [t for t in action_vocab if t not in existing]
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.label_token_ids = {t: self.tokenizer.convert_tokens_to_ids(t) for t in action_vocab}
        self.open_token_id = self.label_token_ids[self.OPEN_TOKEN]
        self.close_token_id = self.label_token_ids[self.CLOSE_TOKEN]
        self.copy_token_id = self.label_token_ids[self.COPY_TOKEN] if self.config.use_copy else None
        self.decoder_start_token_id = self._resolve_decoder_start_token_id()
        self.label_id_set = {
            tok_id
            for token_str, tok_id in self.label_token_ids.items()
            if token_str not in (self.OPEN_TOKEN, self.CLOSE_TOKEN, self.COPY_TOKEN)
        }

        eos_id = int(self.tokenizer.eos_token_id)
        if self.config.use_copy:
            # Small action head ordering:
            #   [<sexp_open>, <sexp_close>, sorted REDUCE-*, eos, <copy>]
            assert self.copy_token_id is not None
            head_ids: list[int] = [
                self.open_token_id,
                self.close_token_id,
                *sorted(self.label_id_set),
                eos_id,
                self.copy_token_id,
            ]
            self.full_id_for_head_idx: list[int] = head_ids
            self.head_idx_for_full_id: dict[int, int] = {fid: i for i, fid in enumerate(self.full_id_for_head_idx)}
            self.head_vocab_size = len(self.full_id_for_head_idx)
            self.open_head_idx = self.head_idx_for_full_id[self.open_token_id]
            self.close_head_idx = self.head_idx_for_full_id[self.close_token_id]
            self.eos_head_idx = self.head_idx_for_full_id[eos_id]
            self.copy_head_idx = self.head_idx_for_full_id[self.copy_token_id]
            self.label_head_indices = {self.head_idx_for_full_id[fid] for fid in self.label_id_set}

            structural_head_ids = sorted(self.label_head_indices | {self.open_head_idx, self.close_head_idx})
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
        else:
            # use_copy=False: scoring vocab IS the full pretrained vocab. No
            # head replacement, no full_id<->head_idx mapping. Structural
            # full-vocab ids are tracked for action_loss_weight rebalance and
            # for the decoding masks.
            self.copy_head_idx = None
            self.full_id_for_head_idx = []
            self.head_idx_for_full_id = {}
            self.head_vocab_size = int(len(self.tokenizer))
            self.label_head_indices = set()
            structural_full_ids = sorted(self.label_id_set | {self.open_token_id, self.close_token_id})
            self.register_buffer(
                "_structural_full_ids_buf",
                torch.tensor(structural_full_ids, dtype=torch.long),
                persistent=False,
            )

    def _install_action_head(self) -> None:
        base = self._underlying_model()

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
                f"Don't know how to replace lm_head on {type(base).__name__}, expected "
                f"`lm_head` as Linear or `lm_head.out_proj` as Linear."
            )
        logger.info(f"Replaced lm_head with fresh Linear(hidden={hidden}, head_vocab_size={self.head_vocab_size}).")

    def _underlying_model(self):
        """Walk PEFT wrappers to reach the original HF model (the one that owns
        the embeddings + lm_head). Plain attribute-walking is safe on the
        encoder-decoder backbone: unlike the causal sibling, its `base_model`
        shortcut does not over-descend past the LM head, so no PEFT-module-origin
        gate is needed (contrast `decoder_only_sexp._underlying_model`)."""
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
        # Truthy: this parser always segments by construction.
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
    ) -> "Seq2SeqSexpParser":
        from iudex.rst.parsers.hfhub import load_parser_from_pretrained

        dev = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return load_parser_from_pretrained(
            repo_or_path,
            parser_cls=cls,
            config_cls=Seq2SeqSexpConfig,
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
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            return_dict=True,
        )
        logits = out.logits  # [B, T, V] where V = head_vocab_size or full vocab

        labels = batch["labels"]
        labels_flat = labels.reshape(-1)
        V = logits.size(-1)
        logits_flat = logits.reshape(-1, V)

        if self.config.use_copy:
            max_id = self._label_to_head_lookup.size(0) - 1
            in_range = (labels_flat >= 0) & (labels_flat <= max_id)
            clamped = labels_flat.clamp(min=0, max=max_id)
            scored_labels_flat = torch.where(
                in_range,
                self._label_to_head_lookup[clamped],
                torch.full_like(labels_flat, -100),
            )
            structural_buf = self._structural_token_ids_buf
        else:
            # Identity mapping: labels are already full-vocab ids, only -100
            # is special (the structural buffer is in full-vocab ids).
            scored_labels_flat = labels_flat
            structural_buf = self._structural_full_ids_buf

        base_loss = F.cross_entropy(
            logits_flat.float(),
            scored_labels_flat,
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )
        metrics: dict[str, torch.Tensor] = {"loss": base_loss}

        if structural_buf.numel() == 0:
            return metrics

        valid_mask = scored_labels_flat != -100
        is_structural = torch.isin(scored_labels_flat, structural_buf) & valid_mask
        n_total = int(valid_mask.sum().item())
        n_structural = int(is_structural.sum().item())
        n_copy = n_total - n_structural
        if n_structural == 0 or n_copy == 0:
            return metrics

        structural_idx = is_structural.nonzero(as_tuple=True)[0]
        structural_logits = logits_flat.index_select(0, structural_idx).float()
        structural_labels = scored_labels_flat.index_select(0, structural_idx)
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

    def encode_input(self, text: str) -> dict[str, list[int]]:
        full_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
        )
        if full_len > self.config.max_input_length - 1:  # -1 for the trailing EOS
            warn(f"Input truncated: {full_len} -> {self.config.max_input_length} subwords. Bump max_input_length.")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def _edu_subword_ids(self, tree: RstTree) -> tuple[str, list[list[int]]]:
        """Per-EDU subword IDs in the encoder's whole-doc tokenization space."""
        text = reconstruct_text(tree)
        full_input_ids, spans = align_edus_to_tokens(self.tokenizer, text, tree.edus)
        edu_subword_ids = [list(full_input_ids[s:e]) for s, e in spans]
        return text, edu_subword_ids

    def encode_target(self, tree: RstTree) -> tuple[list[int], list[int]] | None:
        """Build `(labels, decoder_input_ids)` by walking the tree's sexp
        serialization. Both streams are length-aligned. `decoder_input_ids`
        is the shift-right of the substituted "seen" stream.

        - `<sexp_open>` / `<sexp_close>` and relation labels are predicted
          tokens AND seen tokens.
        - Each EDU's subwords appear, in encoder order, between the leaf
          `<sexp_open>` and `<sexp_close>`. With `use_copy=True`, the
          label at those positions is `<copy>` and the seen stream
          carries the actual source subword id (mirroring seq2seq_sr's
          training-time substitution). With `use_copy=False`, label and
          seen alike carry the actual source subword id.
        - EOS terminates both streams.
        """
        if not self.label_token_ids:
            raise RuntimeError("encode_target called before action vocab was installed. Set cfg.relation_types first.")
        _text, edu_subwords = self._edu_subword_ids(tree)

        # The target references source subwords from the FULL (untruncated) doc
        # tokenization, but `encode_input` truncates the encoder source to
        # `max_input_length`. If the source overruns that budget the target
        # would reference positions the encoder never saw (and under
        # use_copy=False the labels themselves become unseen source ids). Drop
        # the tree rather than emit a desynced (source, target) pair. The minus
        # one budgets the trailing special token `encode_input` adds.
        source_subword_count = sum(len(ids) for ids in edu_subwords)
        if source_subword_count > self.config.max_input_length - 1:
            warn(
                f"Source too long: {source_subword_count} > max_input_length="
                f"{self.config.max_input_length} subwords for a {len(tree.edus)}-EDU tree. "
                f"Tree cannot be encoded (would desync target from truncated encoder source; training raises on this)."
            )
            return None

        label_ids: list[int] = []
        seen_ids: list[int] = []

        def emit_open():
            label_ids.append(self.open_token_id)
            seen_ids.append(self.open_token_id)

        def emit_close():
            label_ids.append(self.close_token_id)
            seen_ids.append(self.close_token_id)

        def emit_label(nuc: str, rel: str):
            token = Reduce(nuc=nuc, rel=rel).to_token()
            if token not in self.label_token_ids:
                raise ValueError(f"encode_target: label token {token!r} missing from vocab. Check cfg.relation_types.")
            tid = self.label_token_ids[token]
            label_ids.append(tid)
            seen_ids.append(tid)

        def emit_leaf(edu_idx: int):
            emit_open()
            for src_id in edu_subwords[edu_idx]:
                if self.config.use_copy:
                    label_ids.append(self.copy_token_id)
                    seen_ids.append(src_id)
                else:
                    label_ids.append(src_id)
                    seen_ids.append(src_id)
            emit_close()

        if len(tree.edus) == 1:
            emit_leaf(0)
        else:
            # Walk the binarized tree directly. `_build_binary_tree` returns
            # leaves as ("edu", text) and internals as ("node", nuc, rel,
            # left, right) in text order, so no DFS-order replay of
            # `parsing_actions` is needed (and the prior right-first
            # ordering of `parsing_actions` made that replay incorrect on
            # real trees).
            binary = tree._build_binary_tree()
            edu_idx = [0]

            def walk(node):
                if node[0] == "edu":
                    emit_leaf(edu_idx[0])
                    edu_idx[0] += 1
                    return
                _, nuc, rel, left, right = node
                if self.config.traversal_order == "preorder":
                    emit_open()
                    emit_label(nuc, rel)
                    walk(left)
                    walk(right)
                    emit_close()
                else:
                    emit_open()
                    walk(left)
                    walk(right)
                    emit_label(nuc, rel)
                    emit_close()

            walk(binary)

        label_ids.append(self.tokenizer.eos_token_id)
        seen_ids.append(self.tokenizer.eos_token_id)
        decoder_input_ids = [self.decoder_start_token_id] + seen_ids[:-1]

        if len(label_ids) > self.config.max_output_length:
            warn(
                f"Target truncated: {len(label_ids)} > max_output_length={self.config.max_output_length} "
                f"for a {len(tree.edus)}-EDU tree. Tree cannot be encoded (training raises on this)."
            )
            return None
        return label_ids, decoder_input_ids

    # -----------------------------------------------------------------
    # Inference helpers
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

    def _tokenizer_special_ids(self) -> frozenset[int]:
        ids: set[int] = set()
        for attr in ("pad_token_id", "bos_token_id", "unk_token_id", "decoder_start_token_id"):
            v = getattr(self.tokenizer, attr, None)
            if v is not None:
                ids.add(int(v))
        # Catches additional specials (PAD aliases, model-specific markers).
        for v in getattr(self.tokenizer, "all_special_ids", []) or []:
            ids.add(int(v))
        dst = getattr(self, "decoder_start_token_id", None)
        if dst is not None:
            ids.add(int(dst))
        # EOS, structural specials, and copy live in their own slots in
        # `structural_ids()`, so don't add them here either (they'd be a
        # no-op union but it's tidier to keep the sets disjoint).
        return frozenset(ids)

    def _initial_state(self, source_ids: list[int]) -> SexpDecodingState:
        return SexpDecodingState(
            source_len=len(source_ids),
            traversal_order=self.config.traversal_order,
            use_copy=self.config.use_copy,
            open_id=self.open_token_id,
            close_id=self.close_token_id,
            eos_id=int(self.tokenizer.eos_token_id),
            label_ids=frozenset(int(x) for x in self.label_id_set),
            copy_id=self.copy_token_id if self.config.use_copy else None,
            source_ids=tuple(source_ids) if not self.config.use_copy else (),
            min_edu_length=int(self.config.min_edu_length),
            constrain_content=bool(self.config.constrain_content),
            tokenizer_special_ids=self._tokenizer_special_ids(),
        )

    def _mask_logits_for_state(self, state: SexpDecodingState, head_V: int) -> torch.Tensor:
        """Boolean mask over the scoring vocab of legal actions at this state.

        `use_copy=True`: scoring vocab is the small action head, so map legal
            full-vocab ids through `head_idx_for_full_id`.
        `use_copy=False`: scoring vocab IS the full tokenizer vocab, legal
            ids are used directly as positions in the mask. When the state
            says content is wildcarded, admit any non-structural id.
        """
        legal = state.legal_actions()
        mask = torch.zeros(head_V, dtype=torch.bool)
        if self.config.use_copy:
            for full_id in legal:
                hi = self.head_idx_for_full_id.get(int(full_id))
                if hi is not None:
                    mask[hi] = True
            return mask
        for full_id in legal:
            fid = int(full_id)
            if 0 <= fid < head_V:
                mask[fid] = True
        if state.content_is_wildcard():
            mask[:] = True
            for fid in state.structural_ids():
                if 0 <= int(fid) < head_V:
                    mask[int(fid)] = False
            # Re-enable explicitly-legal structural ids (e.g. close paren may
            # be the lone structural legal here, we want it back in the mask).
            for full_id in legal:
                fid = int(full_id)
                if 0 <= fid < head_V:
                    mask[fid] = True
        return mask

    def _narrowed_mask(self, narrowed, state: SexpDecodingState, base_mask: torch.Tensor) -> torch.Tensor:
        """Materialize a GoldEduForcer narrowing onto the base legal mask (see
        `GoldEduForcer.narrowed_legal` for the None | frozenset | FORCE_CONTENT
        contract). None -> base_mask unchanged. FORCE_CONTENT -> content wildcard:
        every scoring id except the structurals (CLOSE included, so the leaf
        can't close before the gold target). Multi-element frozenset -> base_mask
        intersected with those ids (mapped to scoring-vocab indices). A singleton
        frozenset is a hard force handled by the caller before reaching here."""
        V = base_mask.shape[-1]
        if narrowed is FORCE_CONTENT:
            mask = torch.ones(V, dtype=torch.bool, device=base_mask.device)
            for fid in state.structural_ids():
                if 0 <= int(fid) < V:
                    mask[int(fid)] = False
            return mask
        if isinstance(narrowed, frozenset):
            keep = torch.zeros(V, dtype=torch.bool, device=base_mask.device)
            for fid in narrowed:
                hi = self.head_idx_for_full_id.get(int(fid)) if self.config.use_copy else int(fid)
                if hi is not None and 0 <= int(hi) < V:
                    keep[int(hi)] = True
            return base_mask & keep
        return base_mask

    # -----------------------------------------------------------------
    # Greedy decoding
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_from_text(self, text: str, *, num_beams: int | None = None) -> RstTree:
        return self.predict_batch_from_texts([text], num_beams=num_beams)[0]

    @torch.no_grad()
    def predict_batch_from_texts(self, texts: list[str], *, num_beams: int | None = None) -> list[RstTree]:
        if not texts:
            return []
        effective_beams = int(num_beams if num_beams is not None else self.config.num_beams)
        if effective_beams <= 1:
            return [self._predict_one_greedy(t) for t in texts]
        return [self._predict_one_beam(t, effective_beams) for t in texts]

    @torch.no_grad()
    def _predict_one_greedy(self, text: str) -> RstTree:
        self.eval()
        device = self.device

        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        ids = enc["input_ids"][0].tolist()
        source_ids = self._strip_specials(ids)
        if not source_ids:
            return empty_tree(self.config.relation_types)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)
        try:
            encoder = self.model.get_encoder()
            enc_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)

            state = self._initial_state(source_ids)
            action_seq: list[int] = []  # full-vocab IDs in emission order

            decoder_input_ids = torch.full((1, 1), self.decoder_start_token_id, device=device, dtype=torch.long)
            past_key_values = None
            done = False
            hit_max_len = False

            for _step in range(self.config.max_output_length):
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
                logits = out.logits[0, -1, :]  # [head_V]

                V = int(logits.size(-1))
                if self.config.use_validity_constraints:
                    mask = self._mask_logits_for_state(state, V).to(device)
                    masked = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
                else:
                    masked = logits
                head_idx = int(masked.argmax(-1).item())
                full_id = self.full_id_for_head_idx[head_idx] if self.config.use_copy else head_idx
                action_seq.append(full_id)

                try:
                    state = state.step(full_id)
                except ValueError:
                    done = True
                    break

                next_input = full_id
                if self.config.use_copy and full_id == self.copy_token_id:
                    # Replace `<copy>` in decoder input with the actual source
                    # subword the cursor just advanced past.
                    src_pos = state.cursor - 1
                    if 0 <= src_pos < len(source_ids):
                        next_input = source_ids[src_pos]
                if full_id == self.tokenizer.eos_token_id:
                    done = True
                    break

                new_step = torch.tensor([[next_input]], device=device, dtype=torch.long)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
            else:
                # for-else: the step loop ran to max_output_length without an
                # EOS break, so we finalize whatever decoded so far.
                hit_max_len = not done
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        if hit_max_len:
            warn(f"Output truncated at inference: max_output_length={self.config.max_output_length} without EOS.")

        # Recompute edu_ranges from the action sequence to handle any
        # mid-leaf truncation cleanly.
        edu_ranges = _edu_ranges_from_actions(
            action_seq,
            source_ids,
            self,
        )
        return self._finalize_tree(action_seq, source_ids, edu_ranges)

    # -----------------------------------------------------------------
    # Beam search
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _predict_one_beam(self, text: str, num_beams: int) -> RstTree:
        self.eval()
        device = self.device
        K = int(num_beams)
        # In use_copy=False mode the scoring vocab IS the model's full output
        # vocab (the pretrained lm_head). In use_copy=True it's the small
        # replacement head. Either way the size is the precomputed
        # `self.head_vocab_size`.
        head_V = self.head_vocab_size

        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        ids = enc["input_ids"][0].tolist()
        source_ids = self._strip_specials(ids)
        if not source_ids:
            return empty_tree(self.config.relation_types)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)
        try:
            encoder = self.model.get_encoder()
            enc_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)

            from transformers.modeling_outputs import BaseModelOutput

            expanded_hidden = enc_out.last_hidden_state.expand(K, -1, -1).contiguous()
            expanded_attn = enc["attention_mask"].expand(K, -1).contiguous()
            enc_out_K = BaseModelOutput(last_hidden_state=expanded_hidden)

            states: list[SexpDecodingState] = [self._initial_state(source_ids) for _ in range(K)]
            action_seqs: list[list[int]] = [[] for _ in range(K)]
            done = [False] * K
            errored = [False] * K  # done via a PDA-rejected action, not EOS
            beam_scores = torch.full((K,), float("-inf"), device=device)
            beam_scores[0] = 0.0
            decoder_input_ids = torch.full((K, 1), self.decoder_start_token_id, device=device, dtype=torch.long)
            past_key_values = None
            finished: list[dict] = []

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

                legal = torch.zeros_like(logits, dtype=torch.bool)
                for j in range(K):
                    if done[j]:
                        continue
                    legal[j] = self._mask_logits_for_state(states[j], head_V).to(device)
                top_scores, parent_of_new, action_of_new = beam_topk_step(beam_scores, logits, legal, K)

                parent_tensor = torch.tensor(parent_of_new, device=device, dtype=torch.long)
                if beam_reorder_needed(step, parent_of_new, K, past_key_values):
                    past_key_values = reorder_past_key_values(past_key_values, parent_tensor, self._underlying_model())
                decoder_input_ids = decoder_input_ids[parent_tensor]

                # No .clone() needed: SexpDecodingState is immutable, so
                # .step() below returns a fresh state and sibling beams sharing
                # a parent never mutate it (unlike the SR ShiftReduceDecodeState
                # path, which clones because it mutates in place).
                new_states = [states[p] for p in parent_of_new]
                new_action_seqs = [list(action_seqs[p]) for p in parent_of_new]
                new_done = [done[p] for p in parent_of_new]
                new_errored = [errored[p] for p in parent_of_new]
                next_inputs = [self.tokenizer.pad_token_id] * K
                for j in range(K):
                    if new_done[j]:
                        continue
                    hi = action_of_new[j]
                    full_id = self.full_id_for_head_idx[hi] if self.config.use_copy else hi
                    new_action_seqs[j].append(full_id)
                    try:
                        new_states[j] = new_states[j].step(full_id)
                    except ValueError:
                        new_done[j] = True
                        new_errored[j] = True
                        continue
                    if full_id == self.tokenizer.eos_token_id:
                        new_done[j] = True
                        continue
                    if self.config.use_copy and full_id == self.copy_token_id:
                        src_pos = new_states[j].cursor - 1
                        if 0 <= src_pos < len(source_ids):
                            next_inputs[j] = source_ids[src_pos]
                        else:
                            next_inputs[j] = full_id
                    else:
                        next_inputs[j] = full_id

                states = new_states
                action_seqs = new_action_seqs
                done = new_done
                errored = new_errored
                beam_scores = top_scores

                for j in range(K):
                    if done[j] and torch.isfinite(beam_scores[j]):
                        if errored[j]:
                            # A finite-score beam whose action the PDA rejected
                            # means the legality mask and the PDA disagree (a
                            # bug, not normal dead-beam topk backfill, which is
                            # always -inf). Drop the beam rather than record a
                            # broken prefix as a finished candidate.
                            warn(
                                "Beam took a mask-legal but PDA-illegal action "
                                "(mask/PDA mismatch). Dropping the beam."
                            )
                            beam_scores[j] = float("-inf")
                            continue
                        finished.append(
                            {
                                "action_seq": list(action_seqs[j]),
                                "score": float(beam_scores[j].item()),
                                "length": len(action_seqs[j]),
                                "finished": True,
                            }
                        )
                        beam_scores[j] = float("-inf")

                new_step = torch.tensor(next_inputs, device=device, dtype=torch.long).unsqueeze(1)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        candidates: list[dict] = list(finished)
        for j in range(K):
            if not done[j] and torch.isfinite(beam_scores[j]):
                candidates.append(
                    {
                        "action_seq": list(action_seqs[j]),
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
                f"Output truncated at inference (beam): no beam reached EOS within "
                f"max_output_length={self.config.max_output_length}."
            )
        edu_ranges = _edu_ranges_from_actions(best["action_seq"], source_ids, self)
        return self._finalize_tree(best["action_seq"], source_ids, edu_ranges)

    # -----------------------------------------------------------------
    # Gold-EDU forced decode
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_with_gold_edus(self, tree: RstTree) -> RstTree:
        return self._predict_one_gold_edu(tree)

    @torch.no_grad()
    def _predict_one_gold_edu(self, tree: RstTree) -> RstTree:
        """Forced decode with gold EDU boundaries.

        Forcing contract (shared with decoder_only_sexp): segmentation
        matches gold by construction.

        - Outside a leaf with more leaves to open: force OPEN (or content
          for preorder fresh-frames where OPEN isn't legal).
        - Inside a leaf, cursor below the gold end: force content
          (`<copy>` / source id).
        - Inside a leaf, cursor at the gold end: force CLOSE.
        - After all leaves closed and inside internal nodes: force CLOSE
          until depth==0, then force EOS.

        Tree shape (which OPENs hold internal vs leaf children) and labels
        come from the model. Segmentation is gold by construction.
        """
        self.eval()
        device = self.device
        text = reconstruct_text(tree)
        gold_ranges = gold_edu_source_ranges(self.tokenizer, tree)

        enc = self.tokenizer(
            text,
            max_length=self.config.max_input_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)
        ids = enc["input_ids"][0].tolist()
        source_ids = self._strip_specials(ids)
        source_len = len(source_ids)
        if not source_ids:
            return empty_tree(self.config.relation_types)

        clamped_ranges: list[tuple[int, int]] = []
        for s, e in gold_ranges:
            if s >= source_len:
                break
            clamped_ranges.append((s, min(e, source_len)))
        if not clamped_ranges:
            return empty_tree(self.config.relation_types)
        n_edus = len(clamped_ranges)
        forcer = GoldEduForcer(n_edus, clamped_ranges)

        gc_active = self.config.gradient_checkpointing
        if gc_active:
            self._set_grad_checkpointing(False)
        try:
            encoder = self.model.get_encoder()
            enc_out = encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True)

            state = dataclasses.replace(self._initial_state(source_ids), min_edu_length=1)
            action_seq: list[int] = []
            decoder_input_ids = torch.full((1, 1), self.decoder_start_token_id, device=device, dtype=torch.long)
            past_key_values = None
            done = False
            hit_max_len = False

            for _step in range(self.config.max_output_length):
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
                logits = out.logits[0, -1, :]

                narrowed = forcer.narrowed_legal(state)
                if isinstance(narrowed, frozenset) and len(narrowed) == 1:
                    full_id = int(next(iter(narrowed)))
                else:
                    base_mask = self._mask_logits_for_state(state, int(logits.size(-1))).to(logits.device)
                    mask = self._narrowed_mask(narrowed, state, base_mask)
                    masked = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
                    head_idx = int(masked.argmax(-1).item())
                    full_id = self.full_id_for_head_idx[head_idx] if self.config.use_copy else head_idx

                action_seq.append(full_id)
                before_state = state
                try:
                    state = state.step(full_id)
                except ValueError:
                    done = True
                    break
                forcer.observe(before_state, state, full_id)

                if full_id == self.tokenizer.eos_token_id:
                    done = True
                    break

                next_input = full_id
                if self.config.use_copy and full_id == self.copy_token_id:
                    src_pos = state.cursor - 1
                    if 0 <= src_pos < len(source_ids):
                        next_input = source_ids[src_pos]
                new_step = torch.tensor([[next_input]], device=device, dtype=torch.long)
                decoder_input_ids = torch.cat([decoder_input_ids, new_step], dim=1)
            else:
                # for-else: the step loop ran to max_output_length without an
                # EOS break, so we finalize whatever decoded so far.
                hit_max_len = not done
        finally:
            if gc_active:
                self._set_grad_checkpointing(True)

        if hit_max_len:
            warn(
                f"Output truncated at inference (gold-edu): max_output_length="
                f"{self.config.max_output_length} without EOS."
            )
        edu_ranges = _edu_ranges_from_actions(action_seq, source_ids, self)
        return self._finalize_tree(action_seq, source_ids, edu_ranges)

    @torch.no_grad()
    def predict(self, tree: RstTree, *, num_beams: int | None = None) -> RstTree:
        text = reconstruct_text(tree)
        return self.predict_from_text(text, num_beams=num_beams)

    @torch.no_grad()
    def predict_batch(self, trees: list[RstTree], *, num_beams: int | None = None) -> list[RstTree]:
        texts = [reconstruct_text(t) for t in trees]
        return self.predict_batch_from_texts(texts, num_beams=num_beams)

    # -----------------------------------------------------------------
    # Tree reconstruction
    # -----------------------------------------------------------------

    def _finalize_tree(
        self, emitted_ids: list[int], source_ids: list[int], pred_edu_ranges: list[tuple[int, int]]
    ) -> RstTree:
        """Build the tree from the emitted sexp ids and stash the source meta the
        dev eval reads off the tree: per-EDU source-position ranges
        (`_pred_edu_source_ranges`, in the encoder's source-id token space) and
        the raw `_source_ids`. If `from_sexp` fell back to a degenerate tree
        (`_from_sexp_failed`), the action-derived ranges are meaningless and are
        nulled out. Side-channel attributes; greedy, beam, and gold-EDU all
        funnel through here."""
        tree = self._tree_from_emitted(emitted_ids, source_ids)
        ranges = [] if getattr(tree, "_from_sexp_failed", False) else pred_edu_ranges
        tree._pred_edu_source_ranges = ranges  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]
        return tree

    def _tree_from_emitted(self, action_ids: list[int], source_ids: list[int]) -> RstTree:
        """Turn the emitted action sequence into an `RstTree` by building a
        sexp string and running it through `RstTree.from_sexp`. Falls back
        to an empty tree on any parse failure. Empty-tree fallbacks carry
        `_from_sexp_failed=True` so callers can null out
        `_pred_edu_source_ranges` (otherwise the action-derived ranges
        wouldn't match the single-EDU fallback's edu count)."""
        sexp_str, edu_texts = self._actions_to_sexp_string(action_ids, source_ids)
        try:
            return RstTree.from_sexp(
                sexp_str,
                traversal_order=self.config.traversal_order,
                edus=edu_texts,
                relation_types=self.config.relation_types,
            )
        except Exception as e:
            # Parses untrusted model output, so degrade on ANY parse failure
            # (incl. RecursionError from a pathologically deep predicted tree),
            # matching decoder_only_sexp. The type is logged so a genuine bug
            # surfaces as a flood of warnings rather than a silent swallow.
            warn(f"Malformed decoder sexp output ({type(e).__name__}: {e}). Falling back to single-EDU tree.")
            full_text = " ".join(edu_texts) if edu_texts else ""
            tree = empty_tree(self.config.relation_types, text=full_text)
            tree._from_sexp_failed = True  # type: ignore[attr-defined]
            return tree

    def _actions_to_sexp_string(self, action_ids: list[int], source_ids: list[int]) -> tuple[str, list[str]]:
        """Walk the action sequence, emitting `<edu>` placeholders for
        leaves and returning the EDU surface forms in document order so
        `RstTree.from_sexp(edus=...)` can fill them back in."""
        eos_id = self.tokenizer.eos_token_id
        pieces: list[str] = []
        edu_texts: list[str] = []
        leaf_buffer: list[int] = []  # source subword ids inside the current leaf
        cursor = 0
        depth = 0
        leaf_open = False  # whether the innermost open span is a leaf

        # We track the "innermost-open-span is unresolved/leaf/internal"
        # via a small stack: per-depth boolean `is_leaf`.
        kind_stack: list[str | None] = []

        def flush_leaf():
            nonlocal leaf_buffer
            if leaf_buffer:
                decoded = self.tokenizer.decode(leaf_buffer, skip_special_tokens=False)
                edu_texts.append(decoded)
            else:
                edu_texts.append("")
            leaf_buffer = []

        # We will produce the canonical placeholder sexp: every leaf is
        # `<edu>`, internal nodes are `(<edu> <edu> NS:rel)` (postorder)
        # or `(NS:rel <edu> <edu>)` (preorder). EDU surface text is
        # passed via the `edus=` arg to `from_sexp`.
        for tok in action_ids:
            if tok == eos_id:
                break
            if tok == self.open_token_id:
                pieces.append("(")
                kind_stack.append(None)
                depth += 1
                leaf_open = False
            elif tok == self.close_token_id:
                if kind_stack and kind_stack[-1] == "leaf":
                    # Emit the placeholder for this leaf, drop the open we
                    # previously appended for it.
                    # The open is the most recent "(" in pieces, remove it
                    # and replace with `<edu>`.
                    # Search back for last "(":
                    for k in range(len(pieces) - 1, -1, -1):
                        if pieces[k] == "(":
                            pieces[k] = "<edu>"
                            break
                    flush_leaf()
                    # The interior open is collapsed, no matching close needed.
                else:
                    pieces.append(")")
                if kind_stack:
                    kind_stack.pop()
                depth = max(0, depth - 1)
                leaf_open = bool(kind_stack) and kind_stack[-1] == "leaf"
            elif tok in self.label_id_set:
                token_str = self.tokenizer.convert_ids_to_tokens(tok)
                nuc, rel = self.label_token_map[token_str]
                pieces.append(f"{nuc}:{rel}")
                if kind_stack:
                    kind_stack[-1] = "internal"
            elif self.config.use_copy and self.copy_token_id is not None and tok == self.copy_token_id:
                if cursor < len(source_ids):
                    leaf_buffer.append(source_ids[cursor])
                    cursor += 1
                if kind_stack and kind_stack[-1] is None:
                    kind_stack[-1] = "leaf"
                    leaf_open = True
            else:
                # Source-id token (use_copy=False) or free-content emission.
                # Mirror decoder_only_sexp: buffer the token unconditionally
                # and advance cursor based on constraint mode. Under
                # constrain_content=False the model emits its own subwords,
                # so the cursor advances per emission regardless of match.
                leaf_buffer.append(tok)
                if self.config.constrain_content:
                    if cursor < len(source_ids) and tok == source_ids[cursor]:
                        cursor += 1
                else:
                    cursor += 1
                if kind_stack and kind_stack[-1] is None:
                    kind_stack[-1] = "leaf"
                    leaf_open = True

        # Best-effort close: drain any unclosed leaf/internal.
        while kind_stack:
            if kind_stack[-1] == "leaf":
                for k in range(len(pieces) - 1, -1, -1):
                    if pieces[k] == "(":
                        pieces[k] = "<edu>"
                        break
                flush_leaf()
            else:
                pieces.append(")")
            kind_stack.pop()

        if not edu_texts:
            # No leaves were emitted, produce a degenerate single-EDU sexp.
            edu_texts = [""]
            return "<edu>", edu_texts
        return " ".join(pieces), edu_texts


# ----- module-level helpers (used by predict + tests) -----


def _edu_ranges_from_actions(
    action_ids: list[int],
    source_ids: list[int],
    parser: "Seq2SeqSexpParser",
) -> list[tuple[int, int]]:
    """Replay the action sequence against `SexpDecodingState` to recover
    per-EDU `(start, end_exclusive)` source-position ranges. Mid-leaf
    truncation closes the final EDU at the current cursor. Uses
    min_edu_length=1 since we're tracing a sequence that has already been
    emitted (the decode-time mask already enforced the configured min)."""
    state = dataclasses.replace(parser._initial_state(source_ids), min_edu_length=1)
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    eos_id = parser.tokenizer.eos_token_id
    for tok in action_ids:
        if tok == eos_id:
            break
        pre_in_leaf = state.in_edu_leaf
        pre_cursor = state.cursor
        try:
            state = state.step(tok)
        except ValueError:
            break
        post_in_leaf = state.in_edu_leaf
        if not pre_in_leaf and post_in_leaf:
            start = pre_cursor
        if pre_in_leaf and not post_in_leaf:
            if start is None:
                start = pre_cursor
            ranges.append((start, state.cursor))
            start = None
    if state.in_edu_leaf and start is not None and state.cursor > start:
        ranges.append((start, state.cursor))
    return ranges
