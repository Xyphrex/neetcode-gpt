import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z -= max(z)

        denom = sum(math.e ** z)
        output = []
        for i in z:
            output.append(round(math.pow(math.e, i)/denom, 4))
        return output

