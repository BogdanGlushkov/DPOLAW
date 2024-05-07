import torch
import torch.nn as nn


class ModelTrainer:
    def __init__(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = torch.tensor(X_train.values, dtype=torch.float32)
        self.y_train = torch.tensor(y_train.values, dtype=torch.float32)
        self.X_val = torch.tensor(X_val.values, dtype=torch.float32) if X_val is not None else None
        self.y_val = torch.tensor(y_val.values, dtype=torch.float32) if y_val is not None else None

    def train_model(self, model, learning_rate=0.01, num_epochs=100):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(self.X_train)
            loss = criterion(outputs, self.y_train)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        self.model = model

    def get_model(self):
        return self.model
