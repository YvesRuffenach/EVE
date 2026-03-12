# EC-1
  
This notebook implements a **probabilistic forecasting** pipeline built around a **variational neuron**. The overall flow is straightforward: a current window `x_cur` and a previous window `x_prev` are encoded, each neuron produces a latent distribution `q(z|h)=N(\mu,\sigma^2)`, and a prediction head turns the latent state into a forecast over the horizon. A **deterministic** variant (`det`) keeps almost the same interface, but without a latent variable.  
  
## Quick check of the main components  
  
**Neuron.** `NeuronVAELayer` computes `mu`, `sigma`, the **KL** term per dimension / per neuron, samples `z`, and aggregates samples when several draws are used. It supports `latent_k > 1`, several `sigma` modes (`scalar`, `per_unit`, `per_unit_input`) and internal pooling (`mean`, `sum`, `sum_sqrtk`). So the “variational neuron” is genuinely implemented as a **local probabilistic computational unit**, not as output noise added at the end.  
  
**Encoder / “decoder”.** The actual encoder is `FeatureDLinearLite`: it separates **trend** and **seasonal** parts, encodes `x_cur` and `x_prev`, then merges persistence and innovation through a gate. `FeatureMLP` is only a compatibility alias here; it is not a second distinct encoder. On the output side, there is no classical VAE decoder reconstructing `x`; the “decoder” role is effectively played by the **prediction head**: `SampleWisePredictionHead` in the variational model, or a simple linear head in the deterministic one.  
  
**Autopilot.** The autopilot: `run_trial` trains a full model, keeps the best validation checkpoint, and exports `cfg.json`, `cfg_final.json`, `summary.json`, `history.csv/json`, `best_payload.json`, and the weights. Then `successive_halving_search` performs a hierarchical search: many short trials first, then fewer longer multi-seed trials later.
  
**Latent-sample aggregation.** There is a real aggregation mechanism. First, inside `NeuronVAELayer`, multiple latent samples can be aggregated by `mean` or `median`. Then, in `ModelNeuronVAETS`, each latent sample produces its own forecast, and forecasts are aggregated by `mean`, `median`, or **softmax score weighting**. The code even keeps a weighted latent `z_vote`.  
  
**The band (`mu^2` band).** The band is set up explicitly from a target latent-energy scale `mu2_target`, with lower and upper factors (`band_lo_k`, `band_hi_k`) through `make_mu2_band`; it can be **homogeneous** (same band for all neurons) or **heterogeneous** (different per-neuron budgets around the same target). During training, the model measures each neuron's mean `mu^2`, adds a hinge-style penalty when it falls below or above the admissible interval, and can then enforce the band either by hard-ish periodic **projection** (`projON`) or by softer **homeostatic rescaling** (`homeo`). The effective band can also be globally re-scaled by a calibration step (`band_state`) after warmup, while keeping the relative heterogeneity across neurons. This band is useful because it prevents dead or exploding latent units, makes the neuron's operating regime measurable (`frac_too_low`, `frac_too_high`, `mu2_mean`) and even feeds downstream controls such as AR noise scaling.  
  
## projON / projOFF / homeo  
  
- **projON**: periodic, rather “legacy” projection that pushes the `mu` layer back into an admissible `mu^2` band.  
- **projOFF**: no explicit projection; training relies only on the loss and penalties.  
- **homeo**: the richest mode; a per-neuron regulator gently adjusts `mu`, controls **beta** per neuron, protects `sigma`, and can also regulate the AR share. In practice, this is the real “living neuron” mode.
