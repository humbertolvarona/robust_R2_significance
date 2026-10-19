# Concordance, Agreement, and R-Squared Significance Analysis

This project provides a statistical toolset designed to rigorously evaluate the structural and predictive agreement between numerical simulations and empirical measurements, as well as test the robustness of the coefficient of determination (R-squared).

Agreement analysis is essential to certify the validity of predictive models and data assimilation systems. The implemented methodology combines parametric and non-parametric approaches, ensuring that discrepancies are exhaustively quantified regardless of the topological distribution of the data. The updated versions allow for dynamic parameterization of the statistical tests, providing granular control over confidence levels, hypotheses, and equivalence bounds.

## Part 1: Concordance and Agreement Analysis

 1. **Mann-Whitney U Test:** The primary advantage of this test is being non-parametric, which frees the researcher from assuming a normal distribution in the measurements. Its main disadvantage is evident in a decrease in statistical power compared to parametric tests when population normality is present. The metric is interpreted directly through its p-value. A result that fails to achieve statistical significance indicates that no structural difference is observed between the central tendencies of the observed and simulated time series. **Parameterizable:** `mw_alternative` (e.g., 'two-sided', 'less', 'greater').

 2. **Median Relative Difference:** The use of this metric offers the advantage of providing a quantification of the central percentage error that is highly resistant to extreme outliers. Its limitation lies in the possibility of masking large errors if the occurrence of severe discrepancies falls exclusively in the asymmetric tails of the distribution. The interpretation is carried out by observing the convergence toward the origin. A coefficient converging to zero certifies an optimal positional correspondence between the empirical system and the computational model.

 3. **Bootstrap Confidence Interval:** Intensive resampling provides the undeniable advantage of estimating the parametric stability of the mean difference without imposing theoretical restrictions on the underlying sample shape. The inherent disadvantages of this approach are the high algorithmic processing demand and potential structural instability when applied to extremely small data vectors. Its interpretation requires verifying the containment of the null parameter. If the confidence interval encompasses absolute zero, the detected difference has no discernible statistical relevance. **Parameterizable:** `n_boot` (number of iterations) and `boot_alpha` (significance level).

 4. **Normalized Root Mean Square Error (NRMSE):** The calculation of this variable grants the advantage of facilitating cross-model analytical comparisons by standardizing the absolute error magnitudes, dividing them by the total range of the observations. Its operational vulnerability is the susceptibility to extreme anomalous measurements, which can artificially expand the range and distort the resulting index. It is interpreted as the percentage of modeling error within the context of the total variability spectrum. Marginal values affirm excellent general predictive precision.

 5. **Scatter Index:** The formulation of this coefficient advantageously contextualizes the absolute error magnitude relative to the equilibrium point or natural mean of the recorded phenomenon. The analytical restriction arises when measurements oscillate or converge around zero, generating asymmetries or undefined divisions. The statistic is interpreted equivalently to a coefficient of variation applied to the bias. Minimal proportions demonstrate that the unmodeled component is negligible compared to the typical behavior of the evaluated ecosystem.

 6. **Spearman Rank Correlation:** This ordinal evaluation guarantees the advantage of revealing complex monotonic synchronies while maintaining robustness against extreme perturbations in the data record. Its structural deficiency is the total omission of the quantitative scale of differences, focusing exclusively on relative positional rank. Its interpretation deciphers strict directional covariation. Significant values close to positive one indicate a proportional homologous reaction to environmental variance. **Parameterizable:** `corr_alternative` ('two-sided', 'less', 'greater').

 7. **Kendall Tau Correlation:** Estimation through this coefficient has the advantage of offering an unbiased and highly efficient ordinal agreement measurement against high densities of repeated values or numerical ties. Its disadvantage manifests in higher computational iteration requirements and the usual generation of nominally lower magnitudes than those of Spearman. The interpretation is performed by observing the dominance of cross-sectional agreement probability. Coefficients approaching one show that the model flawlessly replicates the sample evolutionary pattern. **Parameterizable:** `corr_alternative` ('two-sided', 'less', 'greater').

 8. **TOST Equivalence Test:** Its systematic application provides the advantage of statistically and affirmatively demonstrating that two metric groups are indistinguishable within a precise instrumental boundary. The underlying disadvantage demands that the operator set tolerance limits a priori, a decision that lacks viability when there is no prior theoretical consensus. The interpretive framework requires rejecting the null hypothesis for both one-sided tests. A significant p-value empirically validates that the simulated series is operationally equivalent to the empirical one. **Parameterizable:** `tost_bound_fraction` (fraction of the observed mean).

 9. **Bland-Altman Analysis:** The deployment of this technique offers the notable advantage of isolating systematic bias from random error, establishing exact empirical tolerance limits for future individual discrepancies. Its disadvantage requires assuming the normality of the observed residual differences between homologous methods. The evaluation is interpreted by examining the bias dispersion against its own variance. Biases close to the origin combined with tight agreement limits certify the technical interchangeability of the computational model. **Parameterizable:** `bland_altman_alpha` (confidence level for limits of agreement).

10. **Lin's Concordance Correlation Coefficient (CCC):** The structural strength of this metric rests on the parametric unification of mathematical accuracy and strict geometric fit relative to the ideal diagonal of equality. Its disadvantage is associated with the typical sensitivity of variance-based estimates when high-amplitude atypical instrumental noise is introduced. It is interpreted as a definitive verdict of instrumental and predictive congruence. Records exceeding conventional excellence bounds confirm a superlative degree of analytical validation.

## Part 2: Robust R-Squared Significance Analysis

When empirical data violates standard theoretical assumptions (normality, homoscedasticity), evaluating the true structural significance of the R-squared coefficient requires robust testing.

 1. **Bootstrapping Test:** This resampling method has the advantage of building an empirical confidence interval for the R-squared without relying on underlying population distribution assumptions. The disadvantage is its computational intensity. It is interpreted by checking the lower limit of the interval; if it is substantially greater than zero, the explained variance is genuinely significant. **Parameterizable:** `n_bootstraps` (number of resampling iterations) and `boot_alpha` (significance level for the confidence interval).

 2. **Permutation Test:** This non-parametric approach offers the advantage of generating an exact null distribution by destroying any real relationship between variables. Its limitation lies in processing time for large datasets. A low p-value empirically demonstrates that the model's explanatory capacity is incompatible with random chance. **Parameterizable:** `n_permutations` (number of random shuffles).

 3. **Wald Test with Robust Standard Errors:** The advantage of this parametric correction is its ability to evaluate significance under severe heteroscedasticity conditions. The disadvantage is that it still assumes a linear functional form. If the Wald statistic yields a significant p-value, it confirms the model explains a non-zero fraction of the population variance, despite residual dispersion inequality. **Parameterizable:** `cov_type` (estimator type, such as 'HC0', 'HC1', 'HC2', 'HC3' for Huber-White corrections).

 4. **Cross-Validation:** This empirical validation has the major advantage of isolating mathematical overfitting by measuring out-of-sample R-squared. Its disadvantage is that it reduces the sample size available for training in each fold. It is interpreted through the mean score; a consistently positive cross-validated R-squared proves the original metric captures true structural signals. **Parameterizable:** `cv_folds` (number of data splits) and `cv_random_state` (seed for reproducibility).

## Quick Start

### Concordance and Agreement

```python
import numpy as np
from concordance_analysis_v2 import evaluate_concordance

# Example: Customizing bootstrap iterations, equivalence bounds, and hypotheses
sea_surface_temp_obs = np.array([27.5, 27.8, 28.1, 27.9, 28.2])
sea_surface_temp_croco = np.array([27.4, 27.7, 28.2, 28.0, 28.1])

evaluate_concordance(
    sea_surface_temp_obs, 
    sea_surface_temp_croco, 
    file_name="croco_sst_agreement.txt",
    mw_alternative='two-sided',
    n_boot=2000,
    boot_alpha=0.01,
    corr_alternative='greater',
    tost_bound_fraction=0.15,
    bland_altman_alpha=0.05
)
```

### Robust R-Squared Significance

```python
import numpy as np
from robust_r2_significance_v2 import evaluate_r2_significance

# Example: Testing R-squared significance with customized robust parameters
observed_yield = np.array([3.4, 4.1, 3.8, 4.5, 5.0, 4.8, 5.2])
simulated_yield = np.array([3.2, 4.0, 4.0, 4.3, 4.8, 4.9, 5.1])

evaluate_r2_significance(
    observed_yield, 
    simulated_yield, 
    file_name="yield_model_r2_report.txt",
    n_bootstraps=2000,
    boot_alpha=0.01,
    n_permutations=1500,
    cov_type='HC3',
    cv_folds=3,
    cv_random_state=101
)
```

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
