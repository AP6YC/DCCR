# 11_comparison

This "experiment" is a folder of many experiments implementing the methodology of the DCCR project on continual learning benchmarks.
More specifically:

1. All experiments target the [ContinualAI/continual-learning-baselines](https://github.com/ContinualAI/continual-learning-baselines) benchmarks using the [`Avalanche`](https://github.com/ContinualAI/avalanche) continual learning framework.
2. All experiments are written in Python, using [`pyjulia`](https://github.com/JuliaPy/pyjulia) to call the ART modules of [`AdaptiveResonance.jl`](https://github.com/AP6YC/AdaptiveResonance.jl).
3. IPython notebooks in this folder are mostly used to draft code and analyze datasets (i.e., through TSNEs of deep-extracted feature spaces).
4. Re-implementations of several `Avalanche` datasets and benchmarks are done for two reasons:
    1. The DCCR methodology requires an understanding of the *statistics of the deep-extracted feature spaces*.
    This is done on only the training dataset to not give an unfair advantage to the module during testing.
    2. The feature extraction stage is static between runs since the feature extractor weights are fixed, so these feature and preprocessing results are cached to speed up actual training.

## File Structure

```bash
├── drafting        # Scripts used in the development and inspection of avalanche experiments
├── inits           # Scripts that simply download (initialize) a dataset
├── models          # Cache destination for the preprocessed features
└── src             # Python driver files for all experiments
    └── datasets    # Avalanche dataset definitions for preprocessed variants
```
