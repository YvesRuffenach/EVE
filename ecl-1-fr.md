# EC-1 - fr
  
Ce notebook met en œuvre un pipeline de **prévision probabiliste** construit autour d’un **neurone variationnel**. Le flux général est simple : une fenêtre courante `x_cur` et une fenêtre passée `x_prev` sont encodées, chaque neurone produit une distribution latente `q(z|h)=N(\mu,\sigma^2)`, puis une tête de prédiction transforme cet état latent en prévision sur l’horizon. Une variante **déterministe** (`det`) conserve presque la même interface, mais sans variable latente stochastique.  
  
## Vérification rapide des composants principaux  
  
**Neurone.** `NeuronVAELayer` calcule `mu`, `sigma`, le terme de **KL** par dimension / par neurone, échantillonne `z`, puis agrège les samples lorsqu’on utilise plusieurs tirages. Il prend en charge `latent_k > 1`, plusieurs modes de `sigma` (`scalar`, `per_unit`, `per_unit_input`) et un pooling interne (`mean`, `sum`, `sum_sqrtk`). Le « neurone variationnel » est implémenté comme une **unité probabiliste locale de calcul** et non comme un simple bruit ajouté en sortie.  
  
**Encodeur / « décodeur ».** L’encodeur réel est `FeatureDLinearLite` : il sépare les composantes de **tendance** et de **saisonnalité**, encode `x_cur` et `x_prev`, puis fusionne persistance et innovation via une porte. `FeatureMLP` n’est ici qu’un alias de compatibilité ; ce n’est pas un second encodeur distinct. En sortie, il n’y a pas de décodeur VAE classique reconstruisant `x` ; le rôle de « décodeur » est en pratique assuré par la **tête de prédiction** : `SampleWisePredictionHead` dans le modèle variationnel, ou une tête linéaire simple dans le modèle déterministe.  
  
**Autopilot.** L’autopilot est structuré ainsi : `run_trial` entraîne un modèle complet, conserve le meilleur checkpoint de validation, puis exporte `cfg.json`, `cfg_final.json`, `summary.json`, `history.csv/json`, `best_payload.json` et les poids. Ensuite, `successive_halving_search` organise une recherche hiérarchique : beaucoup d’essais courts au départ, puis moins d’essais mais plus longs et multi-seeds ensuite. C’est un mécanisme de sélection expérimentale, pas un simple balayage brut.  
  
**Agrégation des samples latents.** Il existe un mécanisme d’agrégation. D’abord, à l’intérieur de `NeuronVAELayer`, plusieurs samples latents peuvent être agrégés par `mean` ou `median`. Ensuite, dans `ModelNeuronVAETS`, chaque sample latent produit sa propre prévision, et les prévisions sont agrégées par `mean`, `median` ou **pondération softmax par score**. Le code conserve un latent pondéré `z_vote`.  
  
**La bande (`mu^2` band).** La bande est construite explicitement à partir d’une échelle cible d’énergie latente `mu2_target`, avec des facteurs inférieur et supérieur (`band_lo_k`, `band_hi_k`) via `make_mu2_band` ; elle peut être **homogène** (même bande pour tous les neurones) ou **hétérogène** (budgets différents par neurone autour d’une même cible). Pendant l’entraînement, le modèle mesure le `mu^2` moyen de chaque neurone, ajoute une pénalité de type hinge lorsqu’il passe sous ou au-dessus de l’intervalle admissible, puis peut imposer la bande soit par **projection** périodique plutôt dure (`projON`), soit par **rescaling homéostatique** plus souple (`homeo`). La bande effective peut aussi être re-calibrée globalement après le warmup via une étape de calibration (`band_state`), tout en conservant l’hétérogénéité relative entre neurones. Cette bande est utile car elle évite des unités latentes mortes ou explosives, rend le régime d’activité du neurone mesurable (`frac_too_low`, `frac_too_high`, `mu2_mean`) et alimente même certains contrôles aval, comme l’échelle du bruit AR.  
  
## projON / projOFF / homeo  
  
- **projON** : projection périodique, plutôt « legacy », qui repousse la couche `mu` dans une bande admissible de `mu^2`.  
- **projOFF** : aucune projection explicite ; l’entraînement repose uniquement sur la loss et les pénalités.  
- **homeo** : mode le plus riche ; un régulateur par neurone ajuste doucement `mu`, contrôle **beta** par neurone, protège `sigma`, et peut aussi réguler la part AR. En pratique, c’est le vrai mode de « neurone vivant ».
