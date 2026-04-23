"""
Foundation model embeddings for molecular property prediction.

Provides frozen pretrained embeddings from:
  - CheMeleon: chemistry-pretrained Chemprop D-MPNN (2048-dim fingerprint)
  - ChemBERTa: SMILES-based BERT pretrained on 77M molecules (384-dim)

Both classes cache embeddings to disk so repeated calls are free.

Usage:
    from src.models.foundation_embeddings import CheMeleonEmbedder, ChemBERTaEmbedder

    chemeleon = CheMeleonEmbedder(device="cpu", cache_dir="data/embed_cache")
    X = chemeleon.transform(smiles_list)   # (n, 2048) float32

    chemberta = ChemBERTaEmbedder(device="cpu", cache_dir="data/embed_cache")
    X = chemberta.transform(smiles_list)   # (n, 384) float32
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CheMeleon
# ---------------------------------------------------------------------------

CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_FILENAME = "chemeleon_mp.pt"


class CheMeleonEmbedder:
    """Frozen CheMeleon D-MPNN fingerprints (2048-dim).

    Checkpoint is downloaded once to ~/.chemprop/ and cached.
    Embedding arrays are cached to cache_dir keyed by a hash of the SMILES list.
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 256,
        cache_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path.home() / ".chemprop"
        self._model = None
        self._featurizer = None

    def _load(self):
        if self._model is not None:
            return

        import torch
        from chemprop import featurizers, nn as cpnn
        from chemprop.models import MPNN

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / CHEMELEON_FILENAME

        if not ckpt_path.exists():
            logger.info(f"Downloading CheMeleon checkpoint to {ckpt_path} ...")
            urlretrieve(CHEMELEON_URL, ckpt_path)
            logger.info("Download complete.")

        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        mp = cpnn.BondMessagePassing(**checkpoint["hyper_parameters"])
        mp.load_state_dict(checkpoint["state_dict"])
        agg = cpnn.MeanAggregation()
        from chemprop.nn import RegressionFFN
        predictor = RegressionFFN(input_dim=mp.output_dim)
        model = MPNN(message_passing=mp, agg=agg, predictor=predictor)
        model.eval()
        model.to(self.device)

        self._featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        self._model = model
        logger.info(f"CheMeleon loaded on {self.device} (output_dim={mp.output_dim})")

    def _embed_batch(self, smiles_batch: list[str]) -> np.ndarray:
        import torch
        from rdkit import Chem
        from chemprop.data import BatchMolGraph

        mols = [Chem.MolFromSmiles(s) for s in smiles_batch]
        mgs = [self._featurizer(mol) for mol in mols]
        bmg = BatchMolGraph(mgs)
        bmg.to(self.device)
        with torch.no_grad():
            fp = self._model.fingerprint(bmg)
        return fp.cpu().numpy().astype(np.float32)

    def transform(self, smiles: list[str]) -> np.ndarray:
        """Return (n, dim) CheMeleon fingerprint matrix."""
        cache_key = _smiles_hash(smiles, prefix="chemeleon")
        cached = _load_cache(self.cache_dir, cache_key)
        if cached is not None:
            logger.info(f"CheMeleon: loaded {len(smiles)} embeddings from cache.")
            return cached

        self._load()
        chunks = []
        for start in range(0, len(smiles), self.batch_size):
            batch = smiles[start : start + self.batch_size]
            chunks.append(self._embed_batch(batch))
            if (start // self.batch_size) % 5 == 0:
                logger.info(f"CheMeleon: {min(start + self.batch_size, len(smiles))}/{len(smiles)}")

        result = np.concatenate(chunks, axis=0)
        _save_cache(self.cache_dir, cache_key, result)
        return result


# ---------------------------------------------------------------------------
# ChemBERTa
# ---------------------------------------------------------------------------

CHEMBERTA_MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"


class ChemBERTaEmbedder:
    """Frozen ChemBERTa-77M-MTR SMILES transformer embeddings (384-dim).

    Uses mean pooling over token embeddings.
    """

    def __init__(
        self,
        model_id: str = CHEMBERTA_MODEL_ID,
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 256,
        cache_dir: str | Path | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading {self.model_id} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = (
                self._tokenizer.eos_token
                or self._tokenizer.cls_token
                or self._tokenizer.unk_token
            )
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        logger.info(f"ChemBERTa loaded on {self.device}.")

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask) -> "torch.Tensor":
        import torch
        weights = attention_mask.unsqueeze(-1).float()
        masked = last_hidden_state * weights
        denom = weights.sum(dim=1).clamp(min=1)
        return masked.sum(dim=1) / denom

    def _embed_batch(self, smiles_batch: list[str]) -> np.ndarray:
        import torch

        encoded = self._tokenizer(
            smiles_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = self._model(**encoded)
        pooled = self._mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        return pooled.cpu().numpy().astype(np.float32)

    def transform(self, smiles: list[str]) -> np.ndarray:
        """Return (n, 384) ChemBERTa embedding matrix."""
        cache_key = _smiles_hash(smiles, prefix="chemberta")
        cached = _load_cache(self.cache_dir, cache_key)
        if cached is not None:
            logger.info(f"ChemBERTa: loaded {len(smiles)} embeddings from cache.")
            return cached

        self._load()
        chunks = []
        for start in range(0, len(smiles), self.batch_size):
            batch = smiles[start : start + self.batch_size]
            chunks.append(self._embed_batch(batch))
            if (start // self.batch_size) % 5 == 0:
                logger.info(f"ChemBERTa: {min(start + self.batch_size, len(smiles))}/{len(smiles)}")

        result = np.concatenate(chunks, axis=0)
        _save_cache(self.cache_dir, cache_key, result)
        return result


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _smiles_hash(smiles: list[str], prefix: str) -> str:
    h = hashlib.md5("|".join(smiles).encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def _load_cache(cache_dir: Path | None, key: str) -> np.ndarray | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}.npy"
    if path.exists():
        return np.load(path)
    return None


def _save_cache(cache_dir: Path | None, key: str, arr: np.ndarray) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{key}.npy", arr)
    logger.info(f"Cached embeddings → {cache_dir / key}.npy")
