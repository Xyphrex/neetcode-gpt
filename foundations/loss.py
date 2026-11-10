import numpy as np
from numpy.typing import NDArray


class Solution:
    def binary_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss_per_prediction = []
        for i in range(len(y_true)):
            loss_per_prediction.append((y_true[i] * math.log(y_pred[i])) + ((1 - y_true[i]) * math.log(1 - y_pred[i])))
        return round((-1/len(y_true)) * sum(loss_per_prediction), 4)

    def categorical_cross_entropy(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        losses = []
        for i in range(len(y_true)):
            internal_loss = []
            for k in range(len(y_true[i])):
                internal_loss.append(y_true[i][k] * math.log(y_pred[i][k]))
            losses.append(sum(internal_loss))
        print(losses)
        return round((-1/len(y_true)) * sum(losses), 4)