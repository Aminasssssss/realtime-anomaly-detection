from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import logging

from src.config import (
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_LATENT_DIM,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LR, LSTM_DROPOUT,
    THRESHOLD_PERCENTILE,
)

logger = logging.getLogger(__name__)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.norm(self.fc(h_n[-1]))

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = self.fc(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        outputs, _ = self.lstm(x)
        return self.output_layer(outputs)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=LSTM_HIDDEN_DIM, latent_dim=LSTM_LATENT_DIM, num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class LSTMAutoencoderTrainer:
    def __init__(self, input_dim, seq_len, device="auto"):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.device = self._resolve_device(device)
        self.model = LSTMAutoencoder(input_dim=input_dim, seq_len=seq_len).to(self.device)
        self.threshold: float | None = None
        self.loss_history: list[float] = []

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    def fit(self, windows, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, lr=LSTM_LR):
        logger.info(f"Training LSTM Autoencoder on {self.device} | windows={windows.shape}")
        X = torch.FloatTensor(windows).to(self.device)
        loader = DataLoader(TensorDataset(X, X), batch_size=batch_size, shuffle=True, drop_last=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(self.model(x_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(x_batch)
            scheduler.step()
            avg_loss = total_loss / len(X)
            self.loss_history.append(avg_loss)
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d}/{epochs} | Loss: {avg_loss:.6f}")

        scores = self.score(windows)
        self.threshold = float(np.percentile(scores, THRESHOLD_PERCENTILE))
        logger.info(f"LSTM threshold: {self.threshold:.6f}")
        return self

    def score(self, windows):
        self.model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(windows), LSTM_BATCH_SIZE):
                batch = torch.FloatTensor(windows[i : i + LSTM_BATCH_SIZE]).to(self.device)
                error = ((batch - self.model(batch)) ** 2).mean(dim=(1, 2))
                scores.append(error.cpu().numpy())
        return np.concatenate(scores)

    def predict(self, windows):
        if self.threshold is None:
            raise RuntimeError("Model not trained")
        return (self.score(windows) > self.threshold).astype(int)

    def get_reconstruction(self, window):
        self.model.eval()
        if window.ndim == 2:
            window = window[np.newaxis, ...]
        x = torch.FloatTensor(window).to(self.device)
        with torch.no_grad():
            x_hat = self.model(x)
        return x_hat.squeeze(0).cpu().numpy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "threshold": self.threshold,
            "loss_history": self.loss_history,
        }, path)
        logger.info(f"LSTM saved to {path}")

    @classmethod
    def load(cls, path, device="auto"):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        trainer = cls(input_dim=checkpoint["input_dim"], seq_len=checkpoint["seq_len"], device=device)
        trainer.model.load_state_dict(checkpoint["model_state"])
        trainer.model.to(trainer.device)
        trainer.threshold = checkpoint["threshold"]
        trainer.loss_history = checkpoint.get("loss_history", [])
        return trainer


