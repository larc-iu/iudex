"""End-to-end RST parser via a fine-tuned decoder-only causal LM that emits
a nested s-expression serialization of the tree. Single-stream sibling of
`seq2seq_sexp` and structural sibling of `decoder_only_sr`.

Input layout (training and inference):

    [BOS] source_subwords [SEP] sexp_tokens [EOS]

where `[SEP]` is a learned `<|start_of_sexp|>` token added through the same
special-token / `modules_to_save` flow as the action tokens. Training loss
masks the prefix `[BOS source SEP]` to -100 and only scores the sexp portion
plus the trailing EOS.

Two action-vocab modes:

  use_copy=True:  action vocab = {<sexp_open>, <sexp_close>, <copy>, <eos>}
                  union relation labels. The lm_head is replaced with a small
                  fresh `Linear(hidden, head_vocab_size)`. At inference COPY
                  triggers substitution of the current source subword into the
                  next-step input (same flow as decoder_only_sr).
  use_copy=False: action vocab = {<sexp_open>, <sexp_close>, <eos>} union
                  relation labels. Source subwords appear in-stream as native
                  tokenizer ids. The full pretrained lm_head is kept (we have
                  to score arbitrary subword ids).
"""

import contextlib
import dataclasses
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from iudex.common.log import warn
from iudex.rst.data.tree import RstTree
from iudex.rst.parsers.common.sexp_constraints import (
    GoldEduForcer,
    SexpDecodingState,
)
from iudex.rst.parsers.decoder_only_sexp.configuration_decoder_only_sexp import (
    DecoderOnlySexpConfig,
)

logger = logging.getLogger(__name__)


class DecoderOnlySexpParser(nn.Module):
    SEP_TOKEN: str = "<|start_of_sexp|>"
    OPEN_TOKEN: str = "<sexp_open>"
    CLOSE_TOKEN: str = "<sexp_close>"
    COPY_TOKEN: str = "<copy>"

    def __init__(self, config: DecoderOnlySexpConfig, *, compile_encoder: bool = False):
        super().__init__()
        self.config = config
        del compile_encoder

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.bfloat16 if config.amp else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch_dtype)

        self.label_token_ids: dict[str, int] = {}
        self.label_token_map: dict[str, tuple[str, str]] = {}
        self.label_id_set: set[int] = set()
        self.sep_token_id: int | None = None
        self.open_token_id: int | None = None
        self.close_token_id: int | None = None
        self.copy_token_id: int | None = None
        if config.relation_types is not None:
            self._install_action_vocab()

        if config.peft is not None:
            self._install_peft(config.peft)

        if config.relation_types is not None and config.use_copy:
            self._install_action_head()
            # use_copy=True: small action head, so the input embedding can
            # freeze its base and train only the new-token rows. use_copy=False
            # keeps the full tied lm_head, which must train all embedding rows
            # to score source ids, so we leave that path on the old PEFT
            # modules_to_save scheme.
            self._carve_new_token_embeddings()

        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "config"):
                self.model.base_model.config.use_cache = False

    # -----------------------------------------------------------------
    # Action vocabulary installation
    # -----------------------------------------------------------------

    def _build_label_vocab(self) -> list[str]:
        assert self.config.relation_types is not None
        labels: list[str] = []
        self.label_token_map = {}
        for rel, kind in self.config.relation_types:
            nucs = ("NN",) if kind == "multinuc" else ("NS", "SN")
            for nuc in nucs:
                token = f"<{nuc}:{rel}>"
                labels.append(token)
                self.label_token_map[token] = (nuc, rel)
        return labels

    def _install_peft(self, peft_cfg) -> None:
        """Wrap in LoRA adapters. Under `use_copy=True` the input embedding is
        kept OUT of PEFT modules_to_save (frozen base + small new-rows
        Parameter via `_carve_new_token_embeddings`). Under `use_copy=False`
        the full tied lm_head must learn source ids, so the embedding is
        wrapped in modules_to_save and re-tied like before."""
        from peft import LoraConfig, TaskType, get_peft_model

        use_modules_to_save = not self.config.use_copy
        modules_to_save = peft_cfg.modules_to_save if use_modules_to_save else None
        if use_modules_to_save:
            existing_module_names = {name.rsplit(".", 1)[-1] for name, _ in self.model.named_modules()}
            missing = [m for m in peft_cfg.modules_to_save if m not in existing_module_names]
            if missing:
                raise ValueError(
                    f"peft.modules_to_save references modules not found on {self.config.model_name!r}: "
                    f"{missing}. Available leaf names include embedding/projection candidates like: "
                    f"{sorted(n for n in existing_module_names if 'embed' in n.lower() or 'head' in n.lower() or 'proj' in n.lower())}"
                )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
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

    @contextlib.contextmanager
    def _inference_mode(self):
        """Disable gradient checkpointing AND force `config.use_cache=True`
        during prediction (mirrors decoder_only_sr). Restores both on exit."""
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

    def _install_action_vocab(self) -> None:
        label_vocab = self._build_label_vocab()
        self._original_vocab_size = len(self.tokenizer)
        existing = set(self.tokenizer.get_vocab().keys())
        specials = [self.SEP_TOKEN, self.OPEN_TOKEN, self.CLOSE_TOKEN]
        if self.config.use_copy:
            specials.append(self.COPY_TOKEN)
        all_new = specials + label_vocab
        new_tokens = [t for t in all_new if t not in existing]
        if new_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.sep_token_id = int(self.tokenizer.convert_tokens_to_ids(self.SEP_TOKEN))
        self.open_token_id = int(self.tokenizer.convert_tokens_to_ids(self.OPEN_TOKEN))
        self.close_token_id = int(self.tokenizer.convert_tokens_to_ids(self.CLOSE_TOKEN))
        if self.config.use_copy:
            self.copy_token_id = int(self.tokenizer.convert_tokens_to_ids(self.COPY_TOKEN))
        self.label_token_ids = {t: int(self.tokenizer.convert_tokens_to_ids(t)) for t in label_vocab}
        self.label_id_set = set(self.label_token_ids.values())
        self.label_id_to_str = {tid: t for t, tid in self.label_token_ids.items()}

        if self.config.use_copy:
            eos_id = int(self.tokenizer.eos_token_id)
            self.full_id_for_head_idx: list[int] = [
                self.copy_token_id,
                self.open_token_id,
                self.close_token_id,
                *sorted(self.label_id_set),
                eos_id,
            ]
            self.head_idx_for_full_id: dict[int, int] = {fid: i for i, fid in enumerate(self.full_id_for_head_idx)}
            self.head_vocab_size = len(self.full_id_for_head_idx)
            self.copy_head_idx = self.head_idx_for_full_id[self.copy_token_id]
            self.open_head_idx = self.head_idx_for_full_id[self.open_token_id]
            self.close_head_idx = self.head_idx_for_full_id[self.close_token_id]
            self.eos_head_idx = self.head_idx_for_full_id[eos_id]
            self.label_head_indices = {self.head_idx_for_full_id[fid] for fid in self.label_id_set}

            max_full_id = max(self.full_id_for_head_idx) + 1
            lookup = torch.full((max_full_id,), -100, dtype=torch.long)
            for fid, hi in self.head_idx_for_full_id.items():
                lookup[fid] = hi
            self.register_buffer("_label_to_head_lookup", lookup, persistent=False)
            structural_head_ids = sorted(self.label_head_indices | {self.open_head_idx, self.close_head_idx})
            self.register_buffer(
                "_structural_token_ids_buf",
                torch.tensor(structural_head_ids, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "_label_head_ids_buf",
                torch.tensor(sorted(self.label_head_indices), dtype=torch.long),
                persistent=False,
            )
        else:
            # use_copy=False: structural action ids in FULL vocab space, used
            # only by the action_loss_weight rebalance in `forward`. Source
            # subword ids at leaf-content positions get loss weight 1.0.
            structural_full_ids = sorted(self.label_id_set | {self.open_token_id, self.close_token_id})
            self.register_buffer(
                "_structural_full_ids_buf",
                torch.tensor(structural_full_ids, dtype=torch.long),
                persistent=False,
            )

    def _install_action_head(self) -> None:
        """Replace the model's lm_head with a small fresh `Linear(hidden ->
        head_vocab_size)`. Only used when `use_copy=True`; `use_copy=False`
        keeps the full vocab head because source subwords are scored too."""
        base = self._underlying_model()

        def _warm_init(new_linear: nn.Linear) -> None:
            with torch.no_grad():
                embed_weight = self._underlying_model().get_input_embeddings().weight
                for hi, full_id in enumerate(self.full_id_for_head_idx):
                    src = embed_weight[full_id].to(dtype=new_linear.weight.dtype, device=new_linear.weight.device)
                    new_linear.weight[hi].copy_(src)

        if not hasattr(base, "lm_head"):
            raise RuntimeError(
                f"Don't know how to replace lm_head on {type(base).__name__}; "
                f"expected an `lm_head` attribute (Linear or PEFT-wrapped Linear)."
            )
        old = base.lm_head
        weight = getattr(old, "weight", None)
        if weight is None and hasattr(old, "base_layer"):
            weight = old.base_layer.weight
        if weight is None:
            raise RuntimeError(f"lm_head on {type(base).__name__} has no `.weight` and no `.base_layer.weight`.")
        hidden = weight.shape[1]
        new = nn.Linear(hidden, self.head_vocab_size, bias=False).to(dtype=weight.dtype, device=weight.device)
        _warm_init(new)
        base.lm_head = new
        logger.info(f"Replaced lm_head with fresh Linear(hidden={hidden}, head_vocab_size={self.head_vocab_size}).")

    def _underlying_model(self):
        m = self.model
        if type(m).__module__.startswith("peft"):
            m = m.base_model
            if hasattr(m, "model") and not isinstance(m, nn.ModuleList):
                m = m.model
        return m

    def _carve_new_token_embeddings(self) -> None:
        """Carve the new (SEP + structural + label) token rows out of the
        resized input embedding into a small trainable
        `self.new_token_embeddings` Parameter, freeze the base matrix, and
        splice the two at lookup time. Only called under `use_copy=True` (the
        small action head makes the full embedding redundant on the output
        side). See `Seq2SeqSRParser._carve_new_token_embeddings` for the
        full rationale. Gradient flows only to `new_token_embeddings`; the
        frozen base weight's `.grad` stays None."""
        n_old = int(self._original_vocab_size)
        embed = self._underlying_model().get_input_embeddings()
        full_weight = embed.weight
        n_total = full_weight.shape[0]
        if n_total <= n_old:
            return

        self.new_token_embeddings = nn.Parameter(full_weight.data[n_old:].clone())
        full_weight.requires_grad_(False)
        self._embed_n_old = n_old

        target_ptr = full_weight.data_ptr()
        new_param = self.new_token_embeddings

        def _spliced_embedding_forward(input_ids: torch.Tensor) -> torch.Tensor:
            base = F.embedding(input_ids.clamp(max=n_old - 1), full_weight)
            new = F.embedding((input_ids - n_old).clamp(min=0), new_param)
            return torch.where((input_ids >= n_old).unsqueeze(-1), new, base)

        patched = 0
        for mod in self.model.modules():
            if isinstance(mod, nn.Embedding) and mod.weight.data_ptr() == target_ptr:
                mod.forward = _spliced_embedding_forward
                patched += 1
        logger.info(
            f"Carved {n_total - n_old} new-token embedding rows into a trainable Parameter "
            f"(base {n_total}x{full_weight.shape[1]} frozen); patched {patched} embedding lookup(s)."
        )

    @property
    def segmenter(self):
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
    ) -> "DecoderOnlySexpParser":
        from iudex.rst.parsers.hfhub import load_parser_from_pretrained

        dev = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return load_parser_from_pretrained(
            repo_or_path,
            parser_cls=cls,
            config_cls=DecoderOnlySexpConfig,
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
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            use_cache=False,
        )
        logits = out.logits

        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_labels = batch["labels"][..., 1:].contiguous()

        if self.config.use_copy:
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
            action_loss = F.cross_entropy(
                structural_logits, structural_labels, label_smoothing=self.config.label_smoothing
            )
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

        # use_copy=False: full-vocab cross-entropy over the original lm_head.
        # Source subwords at leaf-content positions are scored alongside the
        # structural action positions (open/close/labels/eos/sep).
        vocab_size = shifted_logits.size(-1)
        labels_flat = shifted_labels.reshape(-1)
        logits_flat = shifted_logits.reshape(-1, vocab_size)
        base_loss = F.cross_entropy(
            logits_flat.float(),
            labels_flat,
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )
        metrics: dict[str, torch.Tensor] = {"loss": base_loss}

        valid_mask = labels_flat != -100
        is_structural = torch.isin(labels_flat, self._structural_full_ids_buf) & valid_mask
        n_total = int(valid_mask.sum().item())
        n_structural = int(is_structural.sum().item())
        n_copy = n_total - n_structural
        if n_structural == 0 or n_copy == 0:
            return metrics

        structural_idx = is_structural.nonzero(as_tuple=True)[0]
        structural_logits = logits_flat.index_select(0, structural_idx).float()
        structural_labels = labels_flat.index_select(0, structural_idx)
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
        full_len = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        enc = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=self.config.max_input_length)
        if full_len > self.config.max_input_length:
            warn(
                f"Source truncated: {full_len} -> {self.config.max_input_length} subwords. "
                f"Bump max_input_length or the doc's tail is invisible to the model."
            )
        return enc["input_ids"]

    def _edu_subword_ids(self, tree: RstTree) -> tuple[list[int], list[list[int]]]:
        """Tokenize the reconstructed doc text once. Return (source_ids,
        per-EDU subword id lists) using character-offset alignment."""
        text = _reconstruct_text(tree)
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        source_ids = enc["input_ids"]
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
                edu_subword_ids.append(list(source_ids[first : last + 1]))
        return source_ids, edu_subword_ids

    def _build_sexp_tokens(
        self,
        tree: RstTree,
        edu_subword_ids: list[list[int]],
    ) -> tuple[list[int], list[int]]:
        """Produce (seen_ids, label_ids) for the sexp portion of the stream.

        `seen_ids` is what flows into the model's input embedding at the next
        time step (COPY positions are substituted with the source subword
        when `use_copy=True`). `label_ids` is what we score against (kept as
        `<copy>` at COPY positions). For `use_copy=False` the two are
        identical and source subwords appear literally.
        """
        traversal = self.config.traversal_order
        binary = tree._build_binary_tree()
        seen: list[int] = []
        labels: list[int] = []

        def render(node, edu_idx: list[int]) -> None:
            if node[0] == "edu":
                idx = edu_idx[0]
                edu_idx[0] += 1
                subwords = edu_subword_ids[idx]
                seen.append(self.open_token_id)
                labels.append(self.open_token_id)
                for sub in subwords:
                    if self.config.use_copy:
                        seen.append(sub)
                        labels.append(self.copy_token_id)
                    else:
                        seen.append(sub)
                        labels.append(sub)
                seen.append(self.close_token_id)
                labels.append(self.close_token_id)
                return
            _, nuc, rel, left, right = node
            label_str = f"<{nuc}:{rel}>"
            if label_str not in self.label_token_ids:
                raise ValueError(
                    f"_build_sexp_tokens: label {label_str!r} not in this parser's label vocab. "
                    f"Did `cfg.relation_types` miss this pair?"
                )
            label_id = self.label_token_ids[label_str]
            seen.append(self.open_token_id)
            labels.append(self.open_token_id)
            if traversal == "preorder":
                seen.append(label_id)
                labels.append(label_id)
                render(left, edu_idx)
                render(right, edu_idx)
            else:
                render(left, edu_idx)
                render(right, edu_idx)
                seen.append(label_id)
                labels.append(label_id)
            seen.append(self.close_token_id)
            labels.append(self.close_token_id)

        render(binary, [0])
        return seen, labels

    def encode_target(self, tree: RstTree) -> tuple[list[int], list[int]] | None:
        """Build a single-stream `(input_ids, labels)` for causal training.

        Layout:
          input_ids = [BOS] + source_ids + [SEP] + seen_sexp + [EOS]
          labels    = [-100] * (1 + len(source_ids)) + [SEP] + label_sexp + [EOS]

        The label mask covers ONLY the sexp portion (from SEP inclusive,
        through EOS). Returns None when either side overflows its cap.
        """
        if self.open_token_id is None or self.sep_token_id is None:
            raise RuntimeError(
                "encode_target called before action vocab was installed; did you forget to set cfg.relation_types?"
            )

        source_ids, edu_subword_ids = self._edu_subword_ids(tree)
        if len(source_ids) > self.config.max_input_length:
            warn(
                f"Source side overflowed: {len(source_ids)} > max_input_length={self.config.max_input_length} "
                f"for a {len(tree.edus)}-EDU tree. Tree DROPPED from this epoch."
            )
            return None

        seen_sexp, label_sexp = self._build_sexp_tokens(tree, edu_subword_ids)
        eos_id = int(self.tokenizer.eos_token_id)
        seen_sexp.append(eos_id)
        label_sexp.append(eos_id)

        if len(label_sexp) > self.config.max_output_length:
            warn(
                f"Target truncated: {len(label_sexp)} > max_output_length={self.config.max_output_length} "
                f"for a {len(tree.edus)}-EDU tree. Tree DROPPED from this epoch."
            )
            return None

        bos_id = int(self.tokenizer.bos_token_id) if self.tokenizer.bos_token_id is not None else eos_id
        input_ids = [bos_id, *source_ids, self.sep_token_id, *seen_sexp]
        # Score from the SEP position onward: the model first predicts SEP
        # (as the transition out of the source prefix), then the sexp body,
        # then EOS. -100 only on [BOS source].
        labels = [-100] * (1 + len(source_ids)) + [self.sep_token_id] + label_sexp
        assert len(input_ids) == len(labels), (len(input_ids), len(labels))
        return input_ids, labels

    # -----------------------------------------------------------------
    # Constraint state
    # -----------------------------------------------------------------

    def _tokenizer_special_ids(self) -> frozenset[int]:
        ids: set[int] = set()
        for attr in ("pad_token_id", "bos_token_id", "unk_token_id"):
            v = getattr(self.tokenizer, attr, None)
            if v is not None:
                ids.add(int(v))
        for v in getattr(self.tokenizer, "all_special_ids", []) or []:
            ids.add(int(v))
        # SEP is also a special-introduced token. Keep it structural so
        # constrain_content=False can't accidentally emit it as content.
        if getattr(self, "sep_token_id", None) is not None:
            ids.add(int(self.sep_token_id))
        return frozenset(ids)

    def _state_for_source(self, source_ids: list[int]) -> SexpDecodingState:
        return SexpDecodingState(
            source_len=len(source_ids),
            traversal_order=self.config.traversal_order,
            use_copy=self.config.use_copy,
            open_id=self.open_token_id,
            close_id=self.close_token_id,
            eos_id=int(self.tokenizer.eos_token_id),
            label_ids=frozenset(self.label_id_set),
            copy_id=self.copy_token_id if self.config.use_copy else None,
            source_ids=tuple() if self.config.use_copy else tuple(source_ids),
            min_edu_length=int(self.config.min_edu_length),
            constrain_content=bool(self.config.constrain_content),
            tokenizer_special_ids=self._tokenizer_special_ids(),
        )

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
            return [self._predict_one_greedy(t) for t in texts]
        return [self._predict_one_beam(t, effective_beams) for t in texts]

    def _build_prefix_ids(self, source_ids: list[int]) -> list[int]:
        bos_id = (
            int(self.tokenizer.bos_token_id)
            if self.tokenizer.bos_token_id is not None
            else int(self.tokenizer.eos_token_id)
        )
        return [bos_id, *source_ids, self.sep_token_id]

    def _vocab_size(self) -> int:
        return int(self._underlying_model().lm_head.weight.shape[0])

    def _mask_logits_for_state(
        self,
        logits: torch.Tensor,
        state: SexpDecodingState,
    ) -> torch.Tensor:
        """Mask logits to the SexpDecodingState's legal set. `logits` is a 1-D
        tensor in the model's native scoring space: head-vocab when
        `use_copy=True`, full vocab when `use_copy=False`. When the state
        marks content as wildcarded (`use_copy=False, constrain_content=False`
        inside a leaf or fresh frame), admit any non-structural id."""
        legal = state.legal_actions()
        masked = torch.full_like(logits, float("-inf"))
        if self.config.use_copy:
            if not legal:
                return masked
            for full_id in legal:
                hi = self.head_idx_for_full_id.get(int(full_id))
                if hi is not None:
                    masked[hi] = logits[hi]
            return masked
        # use_copy=False path.
        V = int(logits.size(-1))
        if state.content_is_wildcard():
            masked.copy_(logits)
            for fid in state.structural_ids():
                if 0 <= int(fid) < V:
                    masked[int(fid)] = float("-inf")
            for full_id in legal:
                fid = int(full_id)
                if 0 <= fid < V:
                    masked[fid] = logits[fid]
            return masked
        if not legal:
            return masked
        idx = torch.tensor(sorted(int(x) for x in legal), dtype=torch.long, device=logits.device)
        masked[idx] = logits[idx]
        return masked

    def _decode_full_id(self, head_idx: int) -> int:
        if self.config.use_copy:
            return self.full_id_for_head_idx[head_idx]
        return head_idx

    @torch.no_grad()
    def _predict_one_greedy(self, text: str) -> RstTree:
        self.eval()
        device = next(self.parameters()).device
        eos_id = int(self.tokenizer.eos_token_id)

        source_ids = self._tokenize_source(text)
        if not source_ids:
            return _empty_tree(self.config.relation_types)
        prefix_ids = self._build_prefix_ids(source_ids)
        source_len = len(source_ids)
        state = self._state_for_source(source_ids)

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

            emitted_ids: list[int] = []
            pred_edu_ranges: list[tuple[int, int]] = []
            leaf_start: int | None = None
            done = False
            hit_max_len = False

            for step in range(self.config.max_output_length):
                if self.config.use_validity_constraints:
                    step_logits = self._mask_logits_for_state(logits, state)
                else:
                    step_logits = logits

                head_idx = int(step_logits.argmax(-1).item())
                full_id = self._decode_full_id(head_idx)

                # Detect leaf-close BEFORE stepping (so we still see the leaf frame).
                closing_leaf = full_id == self.close_token_id and bool(state.stack) and state.stack[-1].kind == "leaf"
                pre_cursor = state.cursor
                try:
                    state = state.step(full_id)
                except ValueError:
                    done = True
                    break
                emitted_ids.append(full_id)

                if full_id == eos_id:
                    done = True
                    break

                # Leaf bookkeeping: cursor advance happens via content tokens.
                if state.cursor > pre_cursor and leaf_start is None:
                    leaf_start = pre_cursor
                if closing_leaf and leaf_start is not None:
                    pred_edu_ranges.append((leaf_start, state.cursor))
                    leaf_start = None

                # Next model input. For COPY, substitute the source subword.
                if self.config.use_copy and full_id == self.copy_token_id:
                    src_pos = state.cursor - 1
                    if 0 <= src_pos < len(source_ids):
                        next_input = source_ids[src_pos]
                    else:
                        next_input = full_id
                else:
                    next_input = full_id

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
                hit_max_len = not done

        if hit_max_len:
            warn(
                f"Output truncated at inference (greedy): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS. "
                f"Tree closed by best-effort repair."
            )
        tree = self._tree_from_emitted(emitted_ids, source_ids)
        # Dedup adjacent identical ranges in case the loop double-counted.
        clean_ranges: list[tuple[int, int]] = []
        for r in pred_edu_ranges:
            if not clean_ranges or clean_ranges[-1] != r:
                clean_ranges.append(r)
        if getattr(tree, "_from_sexp_failed", False):
            clean_ranges = []
        tree._pred_edu_source_ranges = clean_ranges  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]
        return tree

    @torch.no_grad()
    def _predict_one_beam(self, text: str, num_beams: int) -> RstTree:
        self.eval()
        device = next(self.parameters()).device
        K = int(num_beams)
        eos_id = int(self.tokenizer.eos_token_id)

        source_ids = self._tokenize_source(text)
        if not source_ids:
            return _empty_tree(self.config.relation_types)
        prefix_ids = self._build_prefix_ids(source_ids)
        source_len = len(source_ids)
        head_V = self.head_vocab_size if self.config.use_copy else self._vocab_size()

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
            logits = out.logits[:, -1, :]  # [K, V]

            states: list[SexpDecodingState] = [self._state_for_source(source_ids) for _ in range(K)]
            done = [False] * K
            emitted_seqs: list[list[int]] = [[] for _ in range(K)]
            pred_edu_ranges: list[list[tuple[int, int]]] = [[] for _ in range(K)]
            leaf_starts: list[int | None] = [None] * K
            finished_beams: list[dict] = []

            beam_scores = torch.full((K,), float("-inf"), device=device)
            beam_scores[0] = 0.0

            for step in range(self.config.max_output_length):
                if all(done):
                    break

                masked = torch.full_like(logits, float("-inf"))
                for j in range(K):
                    if done[j]:
                        continue
                    masked[j] = self._mask_logits_for_state(logits[j], states[j])

                log_probs = F.log_softmax(masked.float(), dim=-1)
                cum = beam_scores.unsqueeze(1) + log_probs
                # Dead beams (score=-inf, all-masked row) produce -inf + NaN = NaN
                # rows. topk ranks NaN above any finite negative, so without this
                # the dead beam's children would crowd out live beams.
                cum = torch.where(torch.isnan(cum), torch.full_like(cum, float("-inf")), cum)
                top_scores, top_idx = cum.view(-1).topk(K)
                parent_of_new = (top_idx // head_V).tolist()
                action_of_new = (top_idx % head_V).tolist()

                parent_tensor = torch.tensor(parent_of_new, device=device, dtype=torch.long)
                is_step0_uniform = step == 0 and all(p == 0 for p in parent_of_new)
                is_identity = parent_of_new == list(range(K))
                needs_reorder = past_key_values is not None and not (is_step0_uniform or is_identity)
                if needs_reorder:
                    past_key_values = _reorder_pkv(past_key_values, parent_tensor, self._underlying_model())

                new_states = [states[p] for p in parent_of_new]
                new_done = [done[p] for p in parent_of_new]
                new_emitted = [list(emitted_seqs[p]) for p in parent_of_new]
                new_pred_edu_ranges = [list(pred_edu_ranges[p]) for p in parent_of_new]
                new_leaf_starts: list[int | None] = [leaf_starts[p] for p in parent_of_new]

                pad_id = int(self.tokenizer.pad_token_id)
                next_inputs = [pad_id] * K
                for j in range(K):
                    if new_done[j]:
                        continue
                    head_idx = action_of_new[j]
                    full_id = self._decode_full_id(head_idx)
                    pre_state = new_states[j]
                    closing_leaf = (
                        full_id == self.close_token_id and bool(pre_state.stack) and pre_state.stack[-1].kind == "leaf"
                    )
                    pre_cursor = pre_state.cursor
                    try:
                        new_states[j] = new_states[j].step(full_id)
                    except ValueError:
                        new_done[j] = True
                        continue
                    new_emitted[j].append(full_id)
                    if full_id == eos_id:
                        new_done[j] = True
                        continue

                    if new_states[j].cursor > pre_cursor and new_leaf_starts[j] is None:
                        new_leaf_starts[j] = pre_cursor
                    if closing_leaf and new_leaf_starts[j] is not None:
                        new_pred_edu_ranges[j].append((new_leaf_starts[j], new_states[j].cursor))
                        new_leaf_starts[j] = None

                    if self.config.use_copy and full_id == self.copy_token_id:
                        src_pos = new_states[j].cursor - 1
                        if 0 <= src_pos < len(source_ids):
                            next_inputs[j] = source_ids[src_pos]
                        else:
                            next_inputs[j] = full_id
                    else:
                        next_inputs[j] = full_id

                states = new_states
                done = new_done
                emitted_seqs = new_emitted
                pred_edu_ranges = new_pred_edu_ranges
                leaf_starts = new_leaf_starts
                beam_scores = top_scores

                for j in range(K):
                    if done[j] and torch.isfinite(beam_scores[j]):
                        clean = _dedup_ranges(pred_edu_ranges[j])
                        finished_beams.append(
                            {
                                "emitted": list(emitted_seqs[j]),
                                "pred_edu_ranges": clean,
                                "score": float(beam_scores[j].item()),
                                "length": len(emitted_seqs[j]),
                                "finished": True,
                            }
                        )
                        beam_scores[j] = float("-inf")

                if all(done):
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
        for j in range(K):
            if not done[j] and torch.isfinite(beam_scores[j]):
                clean = _dedup_ranges(pred_edu_ranges[j])
                candidates.append(
                    {
                        "emitted": list(emitted_seqs[j]),
                        "pred_edu_ranges": clean,
                        "score": float(beam_scores[j].item()),
                        "length": len(emitted_seqs[j]),
                        "finished": False,
                    }
                )

        if not candidates:
            return _empty_tree(self.config.relation_types)

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
        tree = self._tree_from_emitted(best["emitted"], source_ids)
        pred_ranges = best["pred_edu_ranges"]
        if getattr(tree, "_from_sexp_failed", False):
            pred_ranges = []
        tree._pred_edu_source_ranges = pred_ranges  # type: ignore[attr-defined]
        tree._source_ids = source_ids  # type: ignore[attr-defined]
        return tree

    @torch.no_grad()
    def predict_with_gold_edus(self, tree: RstTree) -> RstTree:
        return self._predict_one_gold_edu(tree)

    @torch.no_grad()
    def _predict_one_gold_edu(self, tree: RstTree) -> RstTree:
        """Forced decode with gold EDU boundaries.

        Forcing contract (shared with seq2seq_sexp): segmentation matches
        gold by construction.

        - Outside a leaf with more leaves to open: force OPEN (or content
          for preorder fresh-frames where OPEN isn't legal).
        - Inside a leaf, cursor below the gold end: force content
          (`<copy>` / source id).
        - Inside a leaf, cursor at the gold end: force CLOSE.
        - After all leaves closed and inside internal nodes: force CLOSE
          until depth==0, then force EOS.

        Tree shape and labels come from the model.
        """
        self.eval()
        device = next(self.parameters()).device
        eos_id = int(self.tokenizer.eos_token_id)

        text = _reconstruct_text(tree)
        gold_ranges = _gold_edu_source_ranges(self.tokenizer, tree)
        source_ids = self._tokenize_source(text)
        if not source_ids:
            return _empty_tree(self.config.relation_types)
        source_len = len(source_ids)
        prefix_ids = self._build_prefix_ids(source_ids)

        clamped_ranges: list[tuple[int, int]] = []
        for s, e in gold_ranges:
            if s >= source_len:
                break
            clamped_ranges.append((s, min(e, source_len)))
        if not clamped_ranges:
            return _empty_tree(self.config.relation_types)
        n_edus = len(clamped_ranges)
        forcer = GoldEduForcer(n_edus, clamped_ranges)

        state = dataclasses.replace(self._state_for_source(source_ids), min_edu_length=1)

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

            emitted_ids: list[int] = []
            pred_edu_ranges: list[tuple[int, int]] = []
            leaf_start: int | None = None
            done = False
            hit_max_len = False

            for step in range(self.config.max_output_length):
                narrowed = forcer.narrowed_legal(state)
                if narrowed is not None and len(narrowed) == 1:
                    full_id = int(next(iter(narrowed)))
                else:
                    masked = self._mask_logits_for_state(logits, state)
                    if narrowed is not None:
                        V = int(masked.size(-1))
                        keep = torch.full_like(masked, float("-inf"))
                        for fid in narrowed:
                            hi = self.head_idx_for_full_id.get(int(fid)) if self.config.use_copy else int(fid)
                            if hi is not None and 0 <= int(hi) < V:
                                keep[int(hi)] = masked[int(hi)]
                        masked = keep
                    head_idx = int(masked.argmax(-1).item())
                    full_id = self._decode_full_id(head_idx)

                closing_leaf = full_id == self.close_token_id and bool(state.stack) and state.stack[-1].kind == "leaf"
                pre_cursor = state.cursor
                before_state = state
                try:
                    state = state.step(full_id)
                except ValueError:
                    done = True
                    break
                emitted_ids.append(full_id)
                forcer.observe(before_state, state, full_id)

                if full_id == eos_id:
                    done = True
                    break

                if state.cursor > pre_cursor and leaf_start is None:
                    leaf_start = pre_cursor
                if closing_leaf and leaf_start is not None:
                    pred_edu_ranges.append((leaf_start, state.cursor))
                    leaf_start = None

                if self.config.use_copy and full_id == self.copy_token_id:
                    src_pos = state.cursor - 1
                    if 0 <= src_pos < len(source_ids):
                        next_input = source_ids[src_pos]
                    else:
                        next_input = full_id
                else:
                    next_input = full_id

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
                hit_max_len = not done

        if hit_max_len:
            warn(
                f"Output truncated at inference (gold-edu): generation hit "
                f"max_output_length={self.config.max_output_length} without EOS. "
                f"Tree closed by best-effort repair."
            )
        tree_out = self._tree_from_emitted(emitted_ids, source_ids)
        pred_ranges = _dedup_ranges(pred_edu_ranges)
        if getattr(tree_out, "_from_sexp_failed", False):
            pred_ranges = []
        tree_out._pred_edu_source_ranges = pred_ranges  # type: ignore[attr-defined]
        tree_out._source_ids = source_ids  # type: ignore[attr-defined]
        return tree_out

    @torch.no_grad()
    def predict(self, tree: RstTree, *, num_beams: int | None = None) -> RstTree:
        text = _reconstruct_text(tree)
        return self.predict_from_text(text, num_beams=num_beams)

    @torch.no_grad()
    def predict_batch(self, trees: list[RstTree], *, num_beams: int | None = None) -> list[RstTree]:
        texts = [_reconstruct_text(t) for t in trees]
        return self.predict_batch_from_texts(texts, num_beams=num_beams)

    # -----------------------------------------------------------------
    # Tree reconstruction
    # -----------------------------------------------------------------

    def _tree_from_emitted(self, emitted_ids: list[int], source_ids: list[int]) -> RstTree:
        """Stringify the emitted action sequence and call `RstTree.from_sexp`.
        Falls back to a single-EDU tree on malformed output."""
        eos_id = int(self.tokenizer.eos_token_id)
        parts: list[str] = []
        leaf_buf: list[int] = []
        cursor = 0
        in_leaf = False

        def flush_leaf():
            if leaf_buf:
                decoded = self.tokenizer.decode(leaf_buf, skip_special_tokens=False).strip()
                if decoded:
                    parts.append(decoded.replace("(", "-LRB-").replace(")", "-RRB-"))
                leaf_buf.clear()

        for tok in emitted_ids:
            if tok == eos_id:
                break
            if tok == self.open_token_id:
                flush_leaf()
                parts.append("(")
                # Whether this opens an internal node or a leaf is determined
                # by what comes next, but we set in_leaf optimistically and
                # flip it back below if a label or another '(' shows up.
                in_leaf = True
                continue
            if tok == self.close_token_id:
                flush_leaf()
                parts.append(")")
                in_leaf = False
                continue
            if tok in self.label_id_set:
                flush_leaf()
                parts.append(self.label_id_to_str[tok][1:-1])  # strip the angle brackets
                in_leaf = False
                continue
            if self.config.use_copy and tok == self.copy_token_id:
                if cursor < len(source_ids):
                    leaf_buf.append(source_ids[cursor])
                    cursor += 1
                continue
            # use_copy=False source token
            leaf_buf.append(tok)
        flush_leaf()

        sexp_text = " ".join(parts)
        try:
            return RstTree.from_sexp(
                sexp_text,
                traversal_order=self.config.traversal_order,
                relation_types=self.config.relation_types,
            )
        except Exception as e:
            warn(f"Malformed sexp output ({type(e).__name__}: {e}); falling back to single-EDU tree.")
            full_text = self.tokenizer.decode(source_ids, skip_special_tokens=True)
            tree = _empty_tree(self.config.relation_types, text=full_text)
            # Mark the fallback so callers null out `_pred_edu_source_ranges`
            # (otherwise the action-derived ranges wouldn't match the
            # single-EDU fallback's edu count and downstream eval would skip).
            tree._from_sexp_failed = True  # type: ignore[attr-defined]
            return tree


def _reconstruct_text(tree: RstTree) -> str:
    parts: list[str] = []
    for i, edu in enumerate(tree.edus):
        if i == 0:
            parts.append(edu.text)
            continue
        prefix = edu.prefix if edu.prefix is not None else " "
        parts.append(prefix + edu.text)
    return "".join(parts)


def _gold_edu_source_ranges(tokenizer, tree: RstTree) -> list[tuple[int, int]]:
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
    from iudex.rst.data.tree import RstNode

    edu_nodes = [RstNode("1", "terminal", text or "")]
    return RstTree(edu_nodes, [], relation_types=relation_types)


def _dedup_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in ranges:
        if not out or out[-1] != r:
            out.append(r)
    return out


def _reorder_pkv(past_key_values, beam_idx: torch.Tensor, underlying_model):
    """Reorder a HF `past_key_values` cache along the beam dimension.
    Mirrors `decoder_only_sr._reorder_pkv` (DynamicCache.reorder_cache
    mutates in place and returns None — fall back to the mutated object
    in that case)."""
    reorder = getattr(underlying_model, "_reorder_cache", None)
    if callable(reorder):
        try:
            result = reorder(past_key_values, beam_idx)
            return result if result is not None else past_key_values
        except (TypeError, AttributeError) as e:
            import warnings

            warnings.warn(
                f"{type(underlying_model).__name__}._reorder_cache failed on "
                f"{type(past_key_values).__name__} ({type(e).__name__}: {e}); "
                "falling back to object/tuple cache reordering.",
                stacklevel=2,
            )
    if hasattr(past_key_values, "reorder_cache"):
        result = past_key_values.reorder_cache(beam_idx)
        return result if result is not None else past_key_values
    return tuple(
        tuple(t.index_select(0, beam_idx) if isinstance(t, torch.Tensor) else t for t in layer)
        for layer in past_key_values
    )
