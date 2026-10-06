import numpy as np

class Sampler:
    """Abstract base class for all sample generators."""

    @classmethod
    def batch_samples(cls, samples: np.ndarray, batch_size: int) -> np.ndarray:
        """Randomly shuffles and batches samples into an array of shape (batch_size, n/batch_size).

           Args:
               samples: The 1D numpy array of generated samples.
               batch_size: The desired size of each batch.

           Returns:
               A 2D numpy array shaped according to the specified dimensions,
               with remaining samples truncated.
        """

        if not isinstance(samples, np.ndarray) or samples.ndim != 1:
            raise ValueError("Samples must be a 1-dimensional numpy array.")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        n = len(samples)
        num_batches = n // batch_size
        
        if num_batches == 0:
             return np.empty((batch_size, 0), dtype=np.float64)

        elements_to_use = num_batches * batch_size
        shuffled_samples = np.random.permutation(n)[:elements_to_use]
        shuffled_samples = shuffled_samples.reshape(num_batches, batch_size)
        return shuffled_samples

    @classmethod
    def mean_batches(cls, batched_samples: np.ndarray) -> np.ndarray:
        """
        Calculates the mean of samples across each batch (averaging along axis=1).

        Args:
            batched_samples: A 2-dimensional numpy array where rows are batches and columns are individual sample values.

        Returns:
            A 1D numpy array containing the mean for each batch.
        """
        
        if not isinstance(batched_samples, np.ndarray) or batched_samples.ndim != 2:
            raise ValueError("Samples must be a 2-dimensional numpy array.")

        return np.mean(batched_samples, axis=1)

class NormalSampler(Sampler):
    """Samples from a Normal (Gaussian) distribution."""

    def __init__(self, mean: float, std_dev: float):
        self.mean = mean
        self.std_dev = std_dev

    def sample(self, n: int) -> np.ndarray:
        """Generates n samples from N(mean, std_dev^2)."""
        return np.random.normal(loc=self.mean, scale=self.std_dev, size=n)

class BinomialSampler(Sampler):
    """Samples from a Binomial distribution (number of successes in n trials)."""

    def __init__(self, trials: int, probability: float):
        if not 0 <= probability <= 1:
            raise ValueError("Probability must be between 0 and 1.")
        self.trials = trials
        self.probability = probability

    def sample(self, n: int) -> np.ndarray:
        """Generates n samples from Binomial(trials, probability)."""
        # np.random.binomial returns integers, cast to ensure return type compliance
        return np.random.binomial(n=self.trials, p=self.probability, size=n)

class PoissonSampler(Sampler):
    """Samples from a Poisson distribution."""

    def __init__(self, rate: float):
        if rate < 0:
            raise ValueError("Rate (lambda) must be non-negative.")
        self.rate = rate

    def sample(self, n: int) -> np.ndarray:
        """Generates n samples from Poisson(rate)."""
        return np.random.poisson(lam=self.rate, size=n)
