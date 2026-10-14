"""
Evader Environment: a contextual bandit where the Evader picks a perturbation
action for a hateful text sample and gets reward based on whether it fools
the Guardian.

State: text feature vector (from preprocess.extract_text_features)
Action: integer 0..6 (perturbation type)
Reward: +1 if Guardian misclassifies perturbed hateful text as safe, 0 otherwise.
"""

import random
import numpy as np
import torch

from red_queen.preprocess import (
    apply_perturbation,
    extract_text_features,
    ACTION_NAMES,
    NUM_ACTIONS,
)


class EvaderBanditEnv:
    """
    Contextual bandit environment for the Evader.

    Each step:
      1. Sample a batch of hateful texts from the dataset.
      2. Evader picks an action (perturbation) for each.
      3. Apply perturbation.
      4. Query Guardian for prediction on perturbed text.
      5. Reward = 1 if Guardian says "safe" (i.e., attack succeeded), else 0.
    """

    def __init__(self, hate_texts: list, guardian, cfg: dict):
        """
        Args:
            hate_texts: list of strings (all labeled as hateful)
            guardian: Guardian instance with predict() and predict_proba()
            cfg: config dict
        """
        self.texts = hate_texts
        self.guardian = guardian
        self.cfg = cfg
        self.batch_size = cfg["evader"]["batch_size"]

    def sample_batch(self) -> list:
        """Sample a random batch of hateful texts."""
        return random.sample(
            self.texts, min(self.batch_size, len(self.texts))
        )

    def get_states(self, texts: list) -> np.ndarray:
        """
        Compute state vectors for a batch of texts.
        Includes Guardian's predicted hate probability as a feature.
        """
        probs = self.guardian.predict_proba(texts)
        states = np.array([
            extract_text_features(t, p) for t, p in zip(texts, probs)
        ], dtype=np.float32)
        return states

    def step(self, texts: list, actions: list) -> dict:
        """
        Apply perturbations and compute rewards.

        Args:
            texts: list of original hateful texts
            actions: list of int action indices

        Returns:
            dict with keys:
              perturbed_texts, rewards, attack_success,
              original_preds, perturbed_preds, actions
        """
        perturbed = [
            apply_perturbation(t, a, self.cfg)
            for t, a in zip(texts, actions)
        ]

        # Get Guardian predictions on perturbed texts
        perturbed_preds = self.guardian.predict(perturbed)
        # Also get predictions on originals for reference
        original_preds = self.guardian.predict(texts)

        # Reward: 1 if Guardian predicts 'safe' (0) on a hateful text
        rewards = [1.0 if pred == 0 else 0.0 for pred in perturbed_preds]

        return {
            "perturbed_texts": perturbed,
            "rewards": np.array(rewards, dtype=np.float32),
            "attack_success": np.mean(rewards),
            "original_preds": original_preds,
            "perturbed_preds": perturbed_preds,
            "actions": actions,
        }

    def get_action_distribution_stats(self, actions: list) -> dict:
        """Return action frequency distribution for logging."""
        counts = {name: 0 for name in ACTION_NAMES}
        for a in actions:
            counts[ACTION_NAMES[a]] += 1
        total = len(actions)
        return {name: count / total for name, count in counts.items()}
