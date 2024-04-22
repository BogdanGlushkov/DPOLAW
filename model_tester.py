import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn


class ModelTester:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def test_linear_regression(self):
        """
        Tests a linear regression model on the given data.
        :return: Mean squared error.
        """
        # Train model
        model = LinearRegression()
        model.fit(self.data, self.target)

        # Evaluate model
        y_pred = model.predict(self.data)
        mse = mean_squared_error(self.target, y_pred)

        return mse

    def test_neural_network(self, input_size, output_size, learning_rate, num_epochs):
        """
        Tests a neural network model with the given input and output sizes on the given data.
        :param input_size: Number of input features.
        :param output_size: Number of output features.
        :param learning_rate: Learning rate for the optimizer.
        :param num_epochs: Number of training epochs.
        :return: Mean squared error.
        """
        # Build model
        model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

        # Convert data to PyTorch tensors
        data = torch.tensor(self.data.values, dtype=torch.float32)
        target = torch.tensor(self.target.values, dtype=torch.float32)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train model
        for epoch in range(num_epochs):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate model
        with torch.no_grad():
            outputs = model(data)
            mse = mean_squared_error(target.numpy(), outputs.numpy())

        return mse

    def test_models(self, models, data, target):
        results = {}
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                outputs = model(data)
                loss = nn.MSELoss()(outputs, target)
                results[name] = {'loss': loss.item()}
        return results
