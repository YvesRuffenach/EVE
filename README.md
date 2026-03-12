# EVE - Elemental Variational Expanse (Variational Distributional Neuron)
Code companion for the arXiv paper: **Variational Distributional Neuron**

This repository will host the official code accompanying the arXiv paper introducing EVE (Elemental Variational Expanse): a variational distributional neuron, i.e. a compute unit whose internal state is a distribution rather than a scalar.

## Availability

The code will be made publicly available on **March 21**.

If you need access before then, please email me and I can share a **raw / research snapshot** of the code.

## Contact

Email: **yves [at] ruffenach [dot] net**


# EC-1

This notebook implements a **probabilistic forecasting** pipeline built around a **variational neuron**. The overall flow is straightforward: a current window `x_cur` and a previous window `x_prev` are encoded, each neuron produces a latent distribution `q(z|h)=N(\mu,\sigma^2)` and a prediction head turns the latent state into a forecast over the horizon. A **deterministic** variant (`det`) keeps almost the same interface, but without a true stochastic latent variable.

## Quick check of the main components

**Neuron.** `NeuronVAELayer` is structurally consistent: it computes `mu`, `sigma`, the **KL** term per dimension / per neuron, samples `z`, and aggregates samples when several draws are used. It supports `latent_k > 1`, several `sigma` modes (`scalar`, `per_unit`, `per_unit_input`) and internal pooling (`mean`, `sum`, `sum_sqrtk`). So the “variational neuron” is genuinely implemented as a **local probabilistic computational unit**, not as output noise added at the end.

**Encoder / “decoder”.** The actual encoder is `FeatureDLinearLite`: it separates **trend** and **seasonal** parts, encodes `x_cur` and `x_prev`, then merges persistence and innovation through a gate. `FeatureMLP` is only a compatibility alias here; it is not a second distinct encoder. On the output side, there is no classical VAE decoder reconstructing `x`; the “decoder” role is effectively played by the **prediction head**: `SampleWisePredictionHead` in the variational model, or a simple linear head in the deterministic one.

**Autopilot.** The autopilot is well structured: `run_trial` trains a full model, keeps the best validation checkpoint, and exports `cfg.json`, `cfg_final.json`, `summary.json`, `history.csv/json`, `best_payload.json` and the weights. Then `successive_halving_search` performs a hierarchical search: many short trials first, then fewer longer multi-seed trials later. This is a clean experimental selection mechanism, not just a raw sweep.

**Latent-sample aggregation.** There is a real aggregation mechanism. First, inside `NeuronVAELayer`, multiple latent samples can be aggregated by `mean` or `median`. Then, in `ModelNeuronVAETS`, each latent sample produces its own forecast, and forecasts are aggregated by `mean`, `median` or **softmax score weighting**. The code even keeps a weighted latent `z_vote`.

## projON / projOFF / homeo

- **projON**: periodic, rather “legacy” projection that pushes the `mu` layer back into an admissible `mu^2` band.
- **projOFF**: no explicit projection; training relies only on the loss and penalties.
- **homeo**: the richest mode; a per-neuron regulator gently adjusts `mu`, controls **beta** per neuron, protects `sigma`, and can also regulate the AR share. In practice, this is the real “living neuron” mode.
