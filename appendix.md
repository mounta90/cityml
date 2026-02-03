# Appendix

## Building Envelope Composite Components

This section describes the material layers and thermal properties for building envelope composites used in commercial (steel-framed) and residential (wood-framed) buildings, for the CitySim simulations.

### Surface Resistance Values (ISO Standard)

Based on [U-Value Calculator](https://www.changeplan.co.uk/u_value_calculator.php)

| Surface Type | External (Rse) | Internal (Rsi) |
|--------------|----------------|----------------|
| Wall | 0.04 m²·K/W | 0.13 m²·K/W |
| Ground Floor | 0.02 m²·K/W | 0.17 m²·K/W |
| Roof | 0.04 m²·K/W | 0.10 m²·K/W |

### Exterior Wall Composite

**Applies to:** Both commercial and residential buildings

#### Layer Stack (Outside → Inside)

| Layer | Thickness (m) | Conductivity (W/m·K) | Specific Heat (J/kg·K) | Density (kg/m³) | R-Value (m²·K/W) |
|-------|---------------|---------------------|------------------------|-----------------|------------------|
| Stucco | 0.0254 | 0.7195 | 839.4584 | 1856.0042 | 0.0353 |
| Gypsum Board | 0.0159 | 0.1599 | 1089.2972 | 800.0018 | 0.0994 |
| Polyisocyanurate Insulation | Variable* | 0.023 | 900.0 | 32.0 | Variable* |
| Gypsum Board | 0.0159 | 0.1599 | 1089.2972 | 800.0018 | 0.0994 |

\* **Variable thickness calculation:**
```
Target R-value = 1 / Wall U-value (input)
Insulation R-value = Target R-value - Stucco R - 2×Gypsum R - Rse - Rsi
Insulation thickness = Insulation R-value × 0.023 W/m·K
```

**Material References:**
- Polyisocyanurate: [Wikipedia](https://en.wikipedia.org/wiki/Polyisocyanurate), [IES VE Thermal Properties](https://help.iesve.com/ve2021/table_6_thermal_conductivity__specific_heat_capacity_and_density.htm)
- Density: 2 pcf = 32 kg/m³ ([Distribution International](https://www.distributioninternational.com/insulation/polyisocyanurate/polyisocyanurate-board))
- R-value performance: [Rmax Blog](https://www.rmax.com/blog/polyiso-r-value-over-time)

---

### Ground Floor Composite

**Applies to:** Both commercial and residential buildings

#### Layer Stack (Outside → Inside)

| Layer | Thickness (m) | Conductivity (W/m·K) | Specific Heat (J/kg·K) | Density (kg/m³) | R-Value (m²·K/W) |
|-------|---------------|---------------------|------------------------|-----------------|------------------|
| Expanded Polystyrene (XPS) | Variable* | 0.035 | 1213.0 | 29.0 | Variable* |
| Normal Weight Concrete (8") | 0.2032 | 2.3085 | 831.4635 | 2322.0053 | 0.0880 |
| Cellular Rubber Underlay (Carpet Pad) | 0.02165 | 0.1 | 400.0 | 1360.0 | 0.2165 |

\* **Variable thickness calculation:**
```
Target R-value = 1 / Floor U-value (input)
Insulation R-value = Target R-value - Concrete R - Pad R - Rse - Rsi
Insulation thickness = Insulation R-value × 0.035 W/m·K
```

**Material References:**
- XPS and Cellular Rubber: [IES VE Thermal Properties](https://help.iesve.com/ve2021/table_6_thermal_conductivity__specific_heat_capacity_and_density.htm)

---

### Exterior Roof Composite - Commercial Building

**Applies to:** Steel-framed commercial buildings

#### Layer Stack (Outside → Inside)

| Layer | Thickness (m) | Conductivity (W/m·K) | Specific Heat (J/kg·K) | Density (kg/m³) | R-Value (m²·K/W) |
|-------|---------------|---------------------|------------------------|-----------------|------------------|
| Roof Membrane | 0.0095 | 0.1599 | 1459.0586 | 1121.2926 | 0.0594 |
| Polyisocyanurate Insulation | Variable* | 0.023 | 900.0 | 32.0 | Variable* |
| Metal Roof Surface | 0.0008 | 45.2497 | 499.6776 | 7824.0179 | 0.0000 |

\* **Variable thickness calculation:**
```
Target R-value = 1 / Roof U-value (input)
Insulation R-value = Target R-value - Membrane R - Metal R - Rse - Rsi
Insulation thickness = Insulation R-value × 0.023 W/m·K
```

**Material References:**
- Polyisocyanurate: [Rmax Blog](https://www.rmax.com/blog/polyiso-r-value-over-time)

---

### Exterior Roof Composite - Residential Building

**Applies to:** Wood-framed residential buildings

#### Layer Stack (Outside → Inside)

| Layer | Thickness (m) | Conductivity (W/m·K) | Specific Heat (J/kg·K) | Density (kg/m³) | R-Value (m²·K/W) |
|-------|---------------|---------------------|------------------------|-----------------|------------------|
| Plywood Deck | 0.0159 | 0.116† | 1585.0‡ | 524.0 | 0.1371 |
| Polyisocyanurate Insulation | Variable* | 0.023 | 900.0 | 32.0 | Variable* |
| Gypsum Board (5/8") | 0.0159 | 0.1599 | 1089.2972 | 800.0018 | 0.0994 |

\* **Variable thickness calculation:**
```
Target R-value = 1 / Roof U-value (input)
Insulation R-value = Target R-value - Gypsum R - Plywood R - Rse - Rsi
Insulation thickness = Insulation R-value × 0.023 W/m·K
```

† Interpolated between values at 15°C and 45°C  
‡ Value at 30°C

**Material References:**
- Plywood: [FSRI Material Database](https://materials.fsri.org/materialdetail/exterior-plywood)

---

### Interior Zone Divider Composite - Commercial Building

**Applies to:** Steel-framed commercial buildings (ceiling assembly)

#### Layer Stack (Outside → Inside)

| Layer | Thickness (m) | Conductivity (W/m·K) | Specific Heat (J/kg·K) | Density (kg/m³) |
|-------|---------------|---------------------|------------------------|-----------------|
| Generic Acoustic Tile | 0.1 | 0.53 | 840.0 | 1280.0 |
| Generic Ceiling Air Gap | 0.1 | 0.556 | 1000.0 | 1.28 |
| Generic LW Concrete | 0.2 | 0.06 | 590.0 | 368.0 |

---

### Interior Zone Divider Composite - Residential Building

**Applies to:** Wood-framed residential buildings (ceiling assembly)

#### Layer Stack (Outside → Inside)

| Layer | Thickness (m) | Conductivity (W/m·K) | Specific Heat (J/kg·K) | Density (kg/m³) |
|-------|---------------|---------------------|------------------------|-----------------|
| Generic Acoustic Tile | 0.02 | 0.06 | 590.0 | 368.0 |
| Generic Ceiling Air Gap | 0.1 | 0.556 | 1000.0 | 1.28 |
| Generic LW Concrete | 0.1 | 0.53 | 840.0 | 1280.0 |

---

---

## Feature Transformation Stage

### Table: Feature Transformation Options

| Feature Skew Type | Transformation Technique |
|---|---|
| Moderately Right | Natural Log |
| | Square Root |
| | Cube Root |
| | Yeo-Johnson |
| Highly Right | Quantile Transformation |
| | Yeo-Johnson |
| Moderately Left | Yeo-Johnson |
| | Box-Cox |
| | Squaring |
| | Cubing |

### Table: Quantile Transformation Hyper-parameters

| Hyper-Parameter | Distribution | Range |
|---|---|---|
| Number of Quantiles | Integer | [100, 500] |
| Max Number of Samples | Integer | [500, 1000] |

---

## Feature Scaling Stage

### Table: Robust Scaling

| Hyper-Parameter | Distribution | Range |
|---|---|---|
| With Centering? | Categorical | { True, False } |
| With Scaling? | Categorical | { True, False } |
| Unit Variance Scaling? | Categorical | { True, False } |
| Quantile Range | Tuple | Uniform[0.0, 0.45] |
| | | Uniform[0.55, 1.0] |

### Table: Min-Max Scaling

| Hyper-Parameter | Distribution | Range |
|---|---|---|
| Feature Range | Categorical | { (-1.0, 1.0), (0.0, 1.0) } |

### Table: Sample Unit Norm Normalization

| Hyper-Parameter | Distribution | Range |
|---|---|---|
| Norm Type | Categorical | { L1, L2, Max } |

---

## Classical Machine Learning Stage

### Table: Classical Machine Learning Algorithm Options

| Category | Algorithm |
|---|---|
| Tree-Based Ensembling | Random Forest |
| | Gradient Boosting |
| | XGBoost |
| | LightGBM |
| | CatBoost |
| | HistogramGB |
| Single Decision Trees | Decision Tree |
| Linear Models | Linear Regression |
| | Ridge |
| | Lasso |
| | Elastic Net |
| | Linear SVM |
| Neighbors-Based | K-Nearest Neighbors |

---

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

