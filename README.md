# Robust $R^2$ Significance Evaluator

## Description
The `evaluate_r2_significance` function is a robust statistical utility designed to validate the coefficient of determination ($R^2$). Classical statistical tests for $R^2$ (like the F-test) rely on strict assumptions such as normality, homoscedasticity, and the absence of outliers. This tool implements four distinct, advanced methodologies—Bootstrapping, Permutation Testing, the HC3 Wald Test, and Cross-Validation—to rigorously determine if your model's explained variance is statistically genuine or merely an artifact of random noise.

---

## Statistical Tests Detailed Breakdown

### 1. Bootstrapping Test
**Explanation:** Bootstrapping is a non-parametric resampling technique. The function draws 1,000 random samples (with replacement) from your original dataset, calculating the $R^2$ for each. This creates an empirical distribution of the $R^2$ statistic, from which a 95% Confidence Interval (CI) is derived.
*   **Strengths:** Does not require assumptions about the underlying population distribution. Highly effective for small to medium sample sizes.
*   **Weaknesses:** Computationally intensive. It assumes that the observations are independent and identically distributed (i.i.d.), which may not hold in time-series data.
*   **Interpretation:** Look at the 95% Confidence Interval. If the lower bound is strictly greater than $0$, you can be 95% confident that the model explains a statistically significant, non-zero portion of the variance.

### 2. Permutation Test
**Explanation:** This test generates a null distribution by randomly shuffling the simulated (predicted) values 1,000 times while keeping the observed values fixed. By destroying any actual relationship between the variables, it calculates what the $R^2$ would look like purely by chance. The p-value is the proportion of shuffled $R^2$ scores that are equal to or greater than the original $R^2$.
*   **Strengths:** Provides an exact, non-parametric test for the null hypothesis (that there is no relationship). It is highly robust against outliers.
*   **Weaknesses:** It tests the strict null hypothesis of *zero* relationship. It can also be computationally expensive for very large datasets.
*   **Interpretation:** A p-value less than standard significance levels (e.g., $p < 0.05$) indicates that your baseline $R^2$ is highly unlikely to have occurred by chance, thus confirming its significance.

### 3. Wald Test with Robust Standard Errors (HC3)
**Explanation:** This approach fits an Ordinary Least Squares (OLS) regression between the observed and simulated values, but applies the HC3 (Huber-White) covariance matrix correction. This correction adjusts the standard errors of the coefficients to account for unequal variance in the residuals.
*   **Strengths:** Highly robust to heteroscedasticity (when the variance of errors changes across the range of values). HC3 is specifically recommended for small samples.
*   **Weaknesses:** Still relies on asymptotic theory and assumes a fundamentally linear relationship. 
*   **Interpretation:** A p-value $< 0.05$ indicates that the relationship between simulated and observed values is statistically significant, even when accounting for potential heteroscedasticity.

### 4. K-Fold Cross-Validation (5 Folds)
**Explanation:** The function splits the data into 5 distinct subsets (folds). It trains a linear model on 4 folds and predicts the 5th, rotating until all folds have been used as a test set. It then calculates the mean out-of-sample $R^2$.
*   **Strengths:** Directly measures the model's ability to generalize to unseen data. It is the gold standard for detecting mathematical overfitting.
*   **Weaknesses:** Reduces the effective sample size for training in each fold. In very small datasets, the cross-validated $R^2$ can be highly volatile.
*   **Interpretation:** If the Mean Out-of-Sample $R^2$ remains positive and relatively close to your baseline $R^2$, your model has true predictive power. A negative or near-zero value indicates severe overfitting.

---

## Examples of Usage

### Example 1: Basic Usage with Synthetic Data
This is the standard way to call the function using simple lists or numpy arrays.

```python
import numpy as np
from robust_r2_significance import evaluate_r2_significance

# Generate highly correlated synthetic data
np.random.seed(42)
observed = np.random.normal(100, 15, 200)
simulated = observed + np.random.normal(0, 5, 200) # Added minor noise

# Run the evaluation
report_path = evaluate_r2_significance(observed, simulated, "example1_report.txt")
print(f"Report successfully saved to: {report_path}")
```

### Example 2: Using Pandas DataFrames
You can easily pass columns from a pandas DataFrame directly into the function.

```python
import pandas as pd
import numpy as np
from robust_r2_significance import evaluate_r2_significance

# Assume we have a CSV with 'Actual_Sales' and 'Predicted_Sales'
# df = pd.read_csv('sales_data.csv')

# Mocking the DataFrame for this example
df = pd.DataFrame({
    'Actual_Sales': np.random.uniform(50, 500, 150),
    'Predicted_Sales': np.random.uniform(50, 500, 150)
})

# Pass the series directly
evaluate_r2_significance(
    df['Actual_Sales'], 
    df['Predicted_Sales'], 
    file_name="sales_validation_report.txt"
)
```

### Example 3: Evaluating a Poor Model (High Noise)
This example demonstrates how the function behaves when the model has no real predictive power, helping you identify false positives.

```python
import numpy as np
from robust_r2_significance import evaluate_r2_significance

# Generate completely independent data (No actual relationship)
np.random.seed(99)
observed_pure_noise = np.random.normal(50, 10, 300)
simulated_pure_noise = np.random.normal(50, 10, 300)

# Run the evaluation
# The resulting report should show:
# - A CI crossing or near zero
# - Permutation p-value > 0.05
# - Negative Cross-Validation R-squared
evaluate_r2_significance(
    observed_pure_noise, 
    simulated_pure_noise, 
    file_name="bad_model_report.txt"
)
```

---

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.