from sklearn.metrics import r2_score
import torch
import torch.nn as nn


class ModelEvaluator:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def evaluate_model(self, model):
        evaluation_results = {}
        model.eval()
        with torch.no_grad():
            outputs = model(self.data)
            loss = nn.MSELoss()(outputs, self.target)
            mae = nn.L1Loss()(outputs, self.target)
            r2 = r2_score(self.target.numpy(), outputs.numpy())
            evaluation_results = {
                'mse': loss.item(),
                'mae': mae.item(),
                'r2': r2
            }
        return evaluation_results
