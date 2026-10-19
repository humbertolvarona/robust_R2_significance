# Robust R-Squared Significance Analysis

This project provides a robust statistical toolset designed to rigorously evaluate the structural significance of the coefficient of determination (R-squared). 

When empirical data violates standard theoretical assumptions such as normality, homoscedasticity, and the absence of outliers, evaluating the true structural significance of the R-squared coefficient requires robust testing. This methodology combines intensive resampling, non-parametric approaches, and heteroscedasticity-consistent covariance matrices to ensure that the explained variance is statistically genuine and not a product of mathematical overfitting or residual structure. The updated version allows for dynamic parameterization of all statistical tests, providing granular control over confidence levels, estimators, and iterations.

## Features and Statistical Analysis

1. **Bootstrapping Test**
   * **Advantage:** This intensive resampling method builds an empirical confidence interval for the R-squared metric without imposing theoretical restrictions on the underlying population distribution.
   * **Disadvantage:** The primary limitation is its high algorithmic processing demand and computational intensity, especially with large datasets.
   * **Interpretation:** It is interpreted by checking the lower limit of the empirical confidence interval. If this lower bound is strictly and substantially greater than zero, the explained variance is genuinely significant.
   * **Parameters:** 
     * `n_bootstraps`: The number of computational resampling iterations (default: 1000).
     * `boot_alpha`: The significance level used to determine the width of the empirical confidence interval (default: 0.05 for a 95% CI).

2. **Permutation Test**
   * **Advantage:** This non-parametric approach offers the undeniable advantage of generating an exact null distribution for the sample by systematically destroying any real relationship between the independent and dependent variables.
   * **Disadvantage:** Its limitation lies in the processing time required for thousands of iterations on large data arrays.
   * **Interpretation:** A low empirical p-value demonstrates that the model's explanatory capacity is incompatible with random chance, proving that the structural dependencies are real.
   * **Parameters:** 
     * `n_permutations`: The number of random shuffles applied to the simulated array (default: 1000).

3. **Wald Test with Robust Standard Errors**
   * **Advantage:** This parametric correction evaluates the explanatory significance of the model under severe heteroscedasticity conditions by adjusting the covariance matrix estimators.
   * **Disadvantage:** The disadvantage is that it still assumes an underlying linear functional form for the regression.
   * **Interpretation:** If the Wald statistic yields a significant p-value (rejecting the null hypothesis that all coefficients are zero), it confirms the model explains a non-zero fraction of the population variance, despite residual dispersion inequality.
   * **Parameters:** 
     * `cov_type`: The type of robust covariance estimator to apply, such as 'HC0', 'HC1', 'HC2', or 'HC3' for Huber-White corrections (default: 'HC3').

4. **Cross-Validation**
   * **Advantage:** This empirical validation technique isolates mathematical overfitting by partitioning the data to measure the out-of-sample R-squared.
   * **Disadvantage:** Its operational disadvantage is that it reduces the sample size available for training the model in each iterative fold.
   * **Interpretation:** It is interpreted through the mean score across all folds. A consistently positive and stable cross-validated R-squared acts as irrefutable proof that the original metric captures true structural signals rather than fitting to sample noise.
   * **Parameters:** 
     * `cv_folds`: The number of data splits or subsets used for training and testing (default: 5).
     * `cv_random_state`: The numerical seed for the random number generator to ensure analytical reproducibility (default: 42).

## Quick Start

```python
import numpy as np
from robust_r2_significance import evaluate_r2_significance

# Example 1: Basic usage with default robust parameters
observed_data = np.array([3.4, 4.1, 3.8, 4.5, 5.0, 4.8, 5.2])
simulated_data = np.array([3.2, 4.0, 4.0, 4.3, 4.8, 4.9, 5.1])

evaluate_r2_significance(
    observed_data, 
    simulated_data, 
    file_name="basic_r2_report.txt"
)
```

```python
import numpy as np
from robust_r2_significance import evaluate_r2_significance

# Example 2: High precision testing with increased iterations
observed_yield = np.array([10.2, 11.5, 12.1, 13.4, 14.0])
simulated_yield = np.array([10.0, 11.2, 12.5, 13.1, 14.2])

evaluate_r2_significance(
    observed_yield, 
    simulated_yield, 
    file_name="high_precision_r2_report.txt",
    n_bootstraps=5000,
    n_permutations=5000,
    cv_folds=3
)
```

```python
import numpy as np
from robust_r2_significance import evaluate_r2_significance

# Example 3: Modifying confidence levels and covariance estimators
obs_temp = np.array([22.1, 23.4, 24.5, 25.1, 26.3, 27.0])
sim_temp = np.array([21.8, 23.6, 24.1, 25.4, 26.1, 27.2])

evaluate_r2_significance(
    obs_temp, 
    sim_temp, 
    file_name="custom_estimator_r2_report.txt",
    boot_alpha=0.01,          # 99% Confidence Interval
    cov_type='HC1',           # Different heteroscedasticity correction
    cv_random_state=101       # Custom seed for reproducibility
)
```

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
