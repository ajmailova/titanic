import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Simple feedforward network for binary classification."""

    def __init__(self, input_dim, hidden_dims=(64, 32),
                 dropouts=(0.3, 0.3), activation='relu'):
        super().__init__()
        assert len(hidden_dims) == len(dropouts)

        act_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }
        act = act_map[activation]

        layers = []
        prev_dim = input_dim
        for h, p in zip(hidden_dims, dropouts):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if p > 0:
                layers.append(nn.Dropout(p))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_dl_model(X_train, y_train,
                   hidden_dims, dropouts, activation,
                   lr, epochs, batch_size,
                   optimizer_name='adam', weight_decay=0.0,
                   random_state=42, verbose=True):
    """Fit an MLP on the given data and return the trained model."""
    torch.manual_seed(random_state)

    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)

    model = MLP(X.shape[1], hidden_dims=hidden_dims,
                dropouts=dropouts, activation=activation)

    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr,
                                        weight_decay=weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, loss={epoch_loss / len(dataset):.4f}')

    return model


def predict_dl_model(model, X):
    """Return binary predictions for the given input."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(np.asarray(X, dtype=np.float32))
        probs = torch.sigmoid(model(X_tensor)).numpy()
    return (probs >= 0.5).astype(int)
