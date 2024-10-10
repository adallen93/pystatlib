"""Utility functions and classes for various statistical methods."""
immport numpy as np

############################# Under Construction ##############################
class RandomVariable:
    """Distribution of a RV, also handles data."""

    def __init__(self, y: list[float], x: np.ndarray = np.zeros(0)) -> None:
        """Initializer."""
        self.y = y
        self.x = x
        self.n = len(y)
        self.mean = 1 / self.n * sum(y)
        self.variance = 1 / (self.n - 1) * sum([(y_i - self.mean) ** 2 for y_i in y])

    def irwls(self, b: np.ndarray) -> np.ndarray:
        """Generic IRWLS. Shouldn't be called as a RV."""
        # Set b_0 and b_m for iterations
        b_0 = b - 1
        b_m = b_0.copy()
        delta = abs(b_0 - b_m)

        while delta > 0.00001:
            # Update b_0 to prior b_m
            b_0 = b_m

            # Calculate/update z_i and w_ii
            eta = self.x @ b_0
            mu = 1 / (1 + np.exp(-eta))
            z = eta + (self.y - mu) * (1 / (mu * (1 - mu)))
            w_ii = (1 / self.variance) * (1 / (mu * (1 - mu))) ** 2
            w = np.diag(w_ii)

            # Estimate parameters
            b_m = self.variance_covariance_matrix @ self.x @ w @ z
            delta = abs(b_0 - b_m)

    @property
    def variance_covariance_matrix(self) -> np.ndarray:
        """Property for the information matrix."""
        raise ValueError("Warning: Cannot perform IRLWS with generic RV")
        return np.zeros(0)

    @property
    def score_matrix(self) -> np.ndarray:
        """Property for the score matrix."""
        raise ValueError("Warning: Cannot perform IRLWS with generic RV")
        return np.zeros(0)


class BinomialDistribution(RandomVariable):
    """Binomial Distribution."""

    def __init__(self, y: list[int], x: np.ndarray = np.zeros(0)) -> None:
        """Doesn't do anything."""
        # Call parent constructor
        super().__init__(y)

    @property
    def variance_covariance(self) -> np.ndarray:
        """Property for the information matrix."""
        # Stop from use if X is empty
        if self.x.empty():
            raise ValueError(
                "Error: X was not passed as an argument\n"
                + "Hint: Try `your_variable.x = np.array(...)\n"
            )

    @property
    def score_matrix(self) -> np.ndarray:
        """Property for the score matrix."""
        # Stop from use if X is empty
        if self.x.empty():
            raise ValueError(
                "Error: X was not passed as an argument\n"
                + "Hint: Try `your_variable.x = np.array(...)\n"
            )


def irwls(y: list[float], x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generic IRWLS. Shouldn't be called as a RV."""
    variance = 1 / (len(y) - 1) * sum([(y_i - sum(y) / len(y)) ** 2 for y_i in y])

    # Set b_0 and b_m for iterations
    b_0 = b - 1
    b_m = b.copy()
    delta = sum(abs(b_0 - b_m))
    print(delta)

    while delta > 0.00001:
        # Update b_0 to prior b_m
        b_0 = b_m

        # Calculate/update z_i and w_ii
        eta = x @ b_0
        mu = 1 / (1 + np.exp(-eta))
        z = eta + (y - mu) * (1 / (mu * (1 - mu)))
        w_ii = (1 / variance) * (1 / (mu * (1 - mu))) ** 2
        w = np.diag(w_ii)

        # Estimate parameters
        b_m = np.linalg.inv(x.T @ w @ x) @ x.T @ w @ z
        delta = sum(abs(b_0 - b_m))
    return b_m
