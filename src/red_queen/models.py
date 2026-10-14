"""
Guardian model: a DistilBERT-based binary hate speech classifier.
"""

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)


class TextClassificationDataset(Dataset):
    """Simple dataset wrapping tokenized text + labels."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class Guardian:
    """
    Wrapper around DistilBERT for binary hate speech classification.
    Provides train, predict, and predict_proba methods.
    """

    def __init__(self, cfg: dict, device: torch.device):
        gcfg = cfg["guardian"]
        self.device = device
        self.max_length = gcfg["max_length"]

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(gcfg["model_name"])
        self.model = DistilBertForSequenceClassification.from_pretrained(
            gcfg["model_name"],
            num_labels=gcfg["num_labels"],
        ).to(device)

    def make_dataset(self, texts, labels) -> TextClassificationDataset:
        return TextClassificationDataset(list(texts), list(labels), self.tokenizer, self.max_length)

    @torch.no_grad()
    def predict(self, texts: list, batch_size: int = 128) -> list:
        """Return predicted class labels (0 or 1) for a list of texts."""
        self.model.eval()
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
        return all_preds

    @torch.no_grad()
    def predict_proba(self, texts: list, batch_size: int = 128) -> list:
        """Return P(hate) for each text."""
        self.model.eval()
        all_probs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
        return all_probs

    def save(self, path: str):
        """Save model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[INFO] Guardian saved to {path}")

    def load(self, path: str):
        """Load model and tokenizer from a saved directory."""
        self.model = DistilBertForSequenceClassification.from_pretrained(path).to(self.device)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        print(f"[INFO] Guardian loaded from {path}")


class EvaderNetwork(nn.Module):
    """
    Small MLP policy network for the Evader.
    Input: text feature vector (state_dim)
    Output: action logits (num_actions)
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.net(state)

    def get_action_and_log_prob(self, state: torch.Tensor):
        """
        Sample an action from the policy and return (action, log_prob).
        """
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy
