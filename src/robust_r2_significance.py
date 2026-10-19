import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

def evaluate_r2_significance(observed, simulated, file_name="r2_robust_report.txt"):
    observed = np.array(observed)
    simulated = np.array(simulated)
    n = len(observed)
    base_r2 = r2_score(observed, simulated)
    n_bootstraps = 1000
    boot_r2 = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        boot_r2.append(r2_score(observed[indices], simulated[indices]))
    boot_lower = np.percentile(boot_r2, 2.5)
    boot_upper = np.percentile(boot_r2, 97.5)
    n_permutations = 1000
    perm_r2 = []
    for _ in range(n_permutations):
        shuffled_simulated = np.random.permutation(simulated)
        perm_r2.append(r2_score(observed, shuffled_simulated))
    p_value_perm = np.sum(np.array(perm_r2) >= base_r2) / n_permutations
    X = sm.add_constant(simulated)
    model = sm.OLS(observed, X)
    results = model.fit(cov_type='HC3')
    wald_p_value = results.f_pvalue
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = []
    for train_index, test_index in kf.split(simulated):
        X_train = simulated[train_index].reshape(-1, 1)
        y_train = observed[train_index]
        X_test = simulated[test_index].reshape(-1, 1)
        y_test = observed[test_index]
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        preds = lr.predict(X_test)
        cv_r2.append(r2_score(y_test, preds))
    mean_cv_r2 = np.mean(cv_r2)
    report_lines = []
    report_lines.append("Robust R-squared Significance Report")
    report_lines.append("=" * 40)
    report_lines.append(f"Sample size: {n}")
    report_lines.append(f"Baseline R-squared: {base_r2:.4f}")
    report_lines.append("")
    report_lines.append("1. Bootstrapping Test")
    report_lines.append(f"Iterations: {n_bootstraps}")
    report_lines.append(f"95% Confidence Interval: [{boot_lower:.4f}, {boot_upper:.4f}]")
    report_lines.append("")
    report_lines.append("2. Permutation Test")
    report_lines.append(f"Permutations: {n_permutations}")
    report_lines.append(f"p-value: {p_value_perm:.4f}")
    report_lines.append("")
    report_lines.append("3. Wald Test with Robust Standard Errors HC3")
    report_lines.append(f"p-value: {wald_p_value:.4e}")
    report_lines.append("")
    report_lines.append("4. Cross-Validation")
    report_lines.append("Folds: 5")
    report_lines.append(f"Mean Out-of-Sample R-squared: {mean_cv_r2:.4f}")
    with open(file_name, "w") as f:
        f.write(chr(10).join(report_lines))
    return file_name
