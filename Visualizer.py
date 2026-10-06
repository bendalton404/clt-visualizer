import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    """
    Constructs and displays plots comparing raw samples drawn from a distribution
    and the means of those batches, to visualize statistical distributions like CLT.
    """
    def __init__(self, 
                 distribution_name: str, 
                 samples: np.ndarray, 
                 batched_means: np.ndarray):
        """
        Initializes the Visualizer with distribution details and data arrays.

        Args:
            distribution_name: A string name for the distribution (e.g., "Normal").
            parameters: A tuple holding the parameters of the distribution (e.g., (mean, std_dev)).
            samples: The raw array of samples drawn from the distribution.
            batched_means: An array containing the calculated means of batches.
        """
        self.distribution_name = distribution_name
        self.samples = samples
        self.batched_means = batched_means

    def calculate_bins(self, samples) -> int:
        """Calculates the number of bins for a histogram"""

        return 20

    def plot(self):
        """
        Constructs and displays two side-by-side plots: 
        the raw samples (left) and the batch means (right).
        """

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax_samples = axes[0]
        ax_samples.hist(self.samples, bins='auto', edgecolor='black', alpha=0.7)
        ax_samples.set_title(f'Distribution of Raw Samples ({self.distribution_name})')
        ax_samples.set_xlabel('Sample Value')
        ax_samples.set_ylabel('Frequency')

        ax_means = axes[1]
        ax_means.hist(self.batched_means, edgecolor='black', alpha=0.7)
        ax_means.set_title('Distribution of Batch Means')
        ax_means.set_xlabel('Mean Value')
        ax_means.set_ylabel('Frequency')
        ax_means.legend()

        fig.suptitle(f'Central Limit Theorem Visualization for {self.distribution_name}', fontsize=16)
        plt.tight_layout()
        plt.show()
