"""The Guardian: a sequence classifier that labels text safe (0) or hate (1).

:class:`Guardian` wraps a Hugging Face model and tokenizer behind three calls
(``predict_proba``, ``predict``, ``save``) so the rest of the package does not
depend on transformers directly. :class:`TextClassifier` is the protocol the
Evader and the evaluation code need; tests use small fakes that implement it.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from red_queen.config import AppConfig, GuardianConfig, HubConfig, TinyModelConfig
from red_queen.errors import ArtifactError
from red_queen.logging_setup import fields
from red_queen.retry import with_retries

log = logging.getLogger(__name__)

NUM_LABELS = 2
CONFIG_SNAPSHOT = "red_queen_config.yaml"
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


class TextClassifier(Protocol):
    """What the attacker and the evaluation need from a classifier."""

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """P(hate) for each text."""

    def predict(self, texts: Sequence[str]) -> npt.NDArray[np.int64]:
        """Predicted label (0 safe, 1 hate) for each text."""


class Guardian:
    """DistilBERT-style binary classifier with batched inference."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        max_length: int,
        eval_batch_size: int,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.eval_batch_size = eval_batch_size

    # ── construction ────────────────────────────────────────────────

    @classmethod
    def create(cls, cfg: AppConfig, device: torch.device, corpus: Sequence[str]) -> Guardian:
        """A fresh, untrained Guardian as described by ``cfg.guardian``.

        ``corpus`` is only used to fit the vocabulary of a tiny offline model.
        """
        if cfg.guardian.tiny is not None:
            return cls.tiny(cfg.guardian, cfg.guardian.tiny, corpus, device, seed=cfg.seed)
        return cls.from_pretrained(cfg.guardian.model_name, cfg.guardian, device, cfg.hub)

    @classmethod
    def from_pretrained(
        cls, source: str | Path, cfg: GuardianConfig, device: torch.device, hub: HubConfig
    ) -> Guardian:
        """Load from a local checkpoint directory or a Hub model id, in one step."""
        local = Path(source).is_dir()
        if isinstance(source, Path) and not local:
            raise ArtifactError(f"Guardian checkpoint {source} does not exist")

        def load() -> tuple[Any, Any]:
            tokenizer = AutoTokenizer.from_pretrained(str(source))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(source), num_labels=NUM_LABELS
            )
            return model, tokenizer

        model, tokenizer = load() if local else with_retries(load, f"model {source}", hub)
        log.info("loaded guardian", extra=fields(source=str(source)))
        return cls(model, tokenizer, device, cfg.max_length, cfg.eval_batch_size)

    @classmethod
    def tiny(
        cls,
        cfg: GuardianConfig,
        tiny: TinyModelConfig,
        corpus: Sequence[str],
        device: torch.device,
        seed: int,
    ) -> Guardian:
        """Randomly initialised small DistilBERT with a WordPiece vocab fit on ``corpus``.

        Needs no network access; used by tests and the CPU smoke pipeline.
        """
        tokenizer = build_wordpiece_tokenizer(corpus, tiny.vocab_size)
        config = DistilBertConfig(
            vocab_size=len(tokenizer),
            dim=tiny.dim,
            n_layers=tiny.n_layers,
            n_heads=tiny.n_heads,
            hidden_dim=tiny.hidden_dim,
            max_position_embeddings=max(cfg.max_length, 8),
            pad_token_id=tokenizer.pad_token_id,
        )
        config.num_labels = NUM_LABELS
        torch.manual_seed(seed)
        model = DistilBertForSequenceClassification(config)
        return cls(model, tokenizer, device, cfg.max_length, cfg.eval_batch_size)

    # ── inference ───────────────────────────────────────────────────

    def encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        """Tokenise with padding to the longest text in the batch."""
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")
        }

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        self.model.eval()
        probs: list[npt.NDArray[np.float64]] = []
        for start in range(0, len(texts), self.eval_batch_size):
            batch = texts[start : start + self.eval_batch_size]
            logits = self.model(**self.encode(batch)).logits
            probs.append(
                torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy().astype(np.float64)
            )
        return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float64)

    def predict(self, texts: Sequence[str]) -> npt.NDArray[np.int64]:
        # With two classes, argmax of the logits is P(hate) > 0.5.
        return (self.predict_proba(texts) > 0.5).astype(np.int64)

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: Path, cfg: AppConfig | None = None) -> Path:
        """Save model and tokenizer; optionally the config that produced them."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if cfg is not None:
            cfg.to_yaml(path / CONFIG_SNAPSHOT)
        log.info("saved guardian", extra=fields(path=str(path)))
        return path


def build_wordpiece_tokenizer(corpus: Sequence[str], vocab_size: int) -> PreTrainedTokenizerFast:
    """Lowercase WordPiece tokenizer with BERT-style special tokens, built offline.

    The vocabulary is constructed directly instead of with the tokenizers
    trainer, whose tie-breaking is not stable across runs: special tokens, then
    every character seen (as a word start and as a ``##`` continuation), then
    the most frequent words (ties broken alphabetically) until ``vocab_size``.
    """
    pad, unk, cls_token, sep, mask = SPECIAL_TOKENS
    normalizer = normalizers.BertNormalizer(lowercase=True)
    pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    counts: Counter[str] = Counter()
    for text in corpus:
        words = pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(text))
        counts.update(word for word, _ in words)
    chars = sorted({ch for word in counts for ch in word})
    vocab_tokens = [*SPECIAL_TOKENS, *chars, *(f"##{ch}" for ch in chars)]
    seen = set(vocab_tokens)
    for word, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(vocab_tokens) >= vocab_size:
            break
        if word not in seen:
            vocab_tokens.append(word)
            seen.add(word)
    tok = Tokenizer(models.WordPiece({t: i for i, t in enumerate(vocab_tokens)}, unk_token=unk))
    tok.normalizer = normalizer
    tok.pre_tokenizer = pre_tokenizer
    tok.post_processor = processors.TemplateProcessing(
        single=f"{cls_token} $A {sep}",
        special_tokens=[(cls_token, tok.token_to_id(cls_token)), (sep, tok.token_to_id(sep))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token=unk,
        pad_token=pad,
        cls_token=cls_token,
        sep_token=sep,
        mask_token=mask,
        model_input_names=["input_ids", "attention_mask"],
    )
