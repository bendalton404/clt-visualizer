from Sampler import *
from Visualizer import Visualizer

n = 1000000
batch_size = 100

data = [
    {
        "cls": NormalSampler,
        "params": (0, 1)
    },
    {
        "cls": BinomialSampler,
        "params": (50, 0.1)
    },
    {
        "cls": PoissonSampler,
        "params": (3,)
    }
]

for example in data:
    sampler = example["cls"](*example["params"])
    samples = sampler.sample(n)
    batched = Sampler.batch_samples(samples, batch_size)
    means = Sampler.mean_batches(batched)
    visualizer = Visualizer(
        example["cls"].__name__,
        samples,
        means
    )
    visualizer.plot()