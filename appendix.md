# Appendix

## Hyperparameter Configurations

### XGBoost

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `learning_rate` | Log-uniform | [0.0001, 0.5] |
| `min_split_loss` | Log-uniform | [0.0001, 5] - 0.0001 |
| `max_depth` | Quantized uniform (int) | [1, 11], step=1 |
| `min_child_weight` | Quantized uniform (int) | [1, 100], step=1 |
| `subsample` | Uniform | [0.5, 1.0] |
| `colsample_bytree` | Uniform | [0.5, 1.0] |
| `colsample_bylevel` | Uniform | [0.5, 1.0] |
| `reg_alpha` | Log-uniform | [0.0001, 1] - 0.0001 |
| `reg_lambda` | Log-uniform | [1, 4] |
| `n_estimators` | Quantized uniform (int) | [100, 6000], step=200 |

### LightGBM

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `max_depth` | Quantized uniform (int) | [1, 11], step=1 |
| `num_leaves` | Quantized uniform (int) | [2, 121], step=1 |
| `learning_rate` | Log-uniform | [0.0001, 0.5] |
| `n_estimators` | Quantized uniform (int) | [100, 6000], step=200 |
| `min_child_weight` | Log-uniform (int) | [1, 100] |
| `subsample` | Uniform | [0.5, 1.0] |
| `colsample_bytree` | Uniform | [0.5, 1.0] |
| `reg_alpha` | Log-uniform | [0.0001, 1] - 0.0001 |
| `reg_lambda` | Log-uniform | [1, 4] |
| `boosting_type` | Choice | ["gbdt", "dart", "goss"] |

### CatBoost

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `iterations` | Quantized uniform (int) | [100, 6000], step=200 |
| `learning_rate` | Log-uniform | [0.0001, 0.5] |
| `depth` | Quantized uniform (int) | [1, 11], step=1 |
| `l2_leaf_reg` | Log-uniform | [1, 10] |

### HistogramGB (Histogram Gradient Boosting)

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `loss` | Choice | ["squared_error", "absolute_error"] |
| `learning_rate` | Log-uniform | [0.0001, 0.5] |
| `max_leaf_nodes` | Quantized normal (int) | mean=31, std=5, min=2 |
| `max_depth` | Quantized uniform (int) | [1, 11], step=1 |
| `min_samples_leaf` | Quantized normal (int) | mean=20, std=2, min=1 |
| `max_features` | Uniform | [0.5, 1.0] |

### Random Forest

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `criterion` | Choice | ["squared_error", "absolute_error"] |
| `n_estimators` | Quantized log-uniform (int) | [10, 3000] |
| `max_depth` | Probability choice | None (70%), 2 (10%), 3 (10%), 4 (10%) |
| `min_samples_split` | Probability choice | 2 (95%), 3 (5%) |
| `min_samples_leaf` | Choice | 1 or log-uniform [2, 50] |
| `max_features` | Probability choice | "sqrt" (20%), "log2" (10%), None (10%), uniform[0,1] (60%) |
| `max_leaf_nodes` | Probability choice | None (85%), 5 (5%), 10 (5%), 15 (5%) |
| `min_impurity_decrease` | Probability choice | 0.0 (85%), 0.01 (5%), 0.02 (5%), 0.05 (5%) |
| `bootstrap` | Choice | [True, False] |

### KNN (K-Nearest Neighbors)

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `weights` | Choice | ["uniform", "distance"] |
| `algorithm` | Choice | ["auto", "ball_tree", "kd_tree", "brute"] |
| `leaf_size` | Quantized uniform (int) | [20, 40], step=1 |
| `p` | Uniform | [1, 5] |
| `metric` | Choice | ["cityblock", "l1", "l2", "minkowski", "euclidean", "manhattan"] |
| `n_neighbors` | Quantized uniform (int) | [1, 15], step=1 |

### LinearSVR (Linear Support Vector Regression)

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `tol` | Log-uniform | [1e-5, 1e-2] |
| `C` | Uniform | [0.5, 1.5] |
| `intercept_scaling` | Uniform | [0.5, 1.5] |
| `max_iter` | Quantized uniform (int) | [10000, 25000], step=1 |
| `loss` | Choice | ["epsilon_insensitive", "squared_epsilon_insensitive"] |

### Linear Regression

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| - | - | No tunable hyperparameters (uses defaults) |

### Decision Tree

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `splitter` | Choice | ["best", "random"] |
| `max_depth` | Probability choice | None (70%), 2 (10%), 3 (10%), 4 (10%) |
| `min_samples_split` | Probability choice | 2 (95%), 3 (5%) |
| `min_samples_leaf` | Choice | 1 or log-uniform [2, 50] |
| `min_weight_fraction_leaf` | Probability choice | 0.0 (95%), 0.01 (5%) |
| `max_features` | Probability choice | "sqrt" (20%), "log2" (10%), None (10%), uniform[0,1] (60%) |
| `min_impurity_decrease` | Probability choice | 0.0 (85%), 0.01 (5%), 0.02 (5%), 0.05 (5%) |
| `max_leaf_nodes` | Probability choice | None (85%), 5 (5%), 10 (5%), 15 (5%) |
| `criterion` | Choice | ["squared_error", "friedman_mse", "absolute_error"] |

### Gradient Boosting Regressor

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `loss` | Choice | ["squared_error", "absolute_error", "huber"] |
| `alpha` | Uniform | [0.85, 0.95] |
| `learning_rate` | Log-uniform | [0.0001, 0.5] |
| `n_estimators` | Quantized log-uniform (int) | [11, 1000] |
| `criterion` | Choice | ["friedman_mse", "squared_error"] |
| `min_samples_split` | Probability choice | 2 (95%), 3 (5%) |
| `min_samples_leaf` | Choice | 1 or log-uniform [2, 50] |
| `max_depth` | Probability choice | 2 (10%), 3 (70%), 4 (10%), 5 (10%) |
| `min_impurity_decrease` | Probability choice | 0.0 (85%), 0.01 (5%), 0.02 (5%), 0.05 (5%) |
| `max_features` | Probability choice | "sqrt" (20%), "log2" (10%), None (10%), uniform[0,1] (60%) |
| `max_leaf_nodes` | Probability choice | None (85%), 5 (5%), 10 (5%), 15 (5%) |

### Ridge

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `alpha` | Log-uniform | [1e-3, 1e3] |
| `max_iter` | Quantized uniform (int) | [750, 5000], step=50 |
| `tol` | Log-uniform | [1e-5, 1e-2] |

### Lasso

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `alpha` | Log-uniform | [1e-3, 1e3] |
| `max_iter` | Quantized uniform (int) | [750, 5000], step=50 |
| `tol` | Log-uniform | [1e-5, 1e-2] |

### Elastic Net

| Hyperparameter | Distribution | Range/Values |
|----------------|--------------|--------------|
| `alpha` | Log-uniform | [1e-3, 1e3] |
| `max_iter` | Quantized uniform (int) | [750, 5000], step=50 |
| `tol` | Log-uniform | [1e-5, 1e-2] |
| `l1_ratio` | Choice | 0.5 or uniform[0.0, 1.0] |

