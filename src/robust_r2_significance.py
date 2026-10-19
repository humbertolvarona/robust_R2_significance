import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

def evaluate_r2_significance(
    observed, 
    simulated, 
    file_name="r2_robust_report.txt",
    n_bootstraps=1000,
    boot_alpha=0.05,
    n_permutations=1000,
    cov_type='HC3',
    cv_folds=5,
    cv_random_state=42
):
    observed = np.array(observed)
    simulated = np.array(simulated)
    n = len(observed)
    
    # Baseline R-squared
    base_r2 = r2_score(observed, simulated)
    
    # 1. Bootstrapping
    boot_r2 = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(n, size=n, replace=True)
        boot_r2.append(r2_score(observed[indices], simulated[indices]))
    boot_lower = np.percentile(boot_r2, (boot_alpha / 2) * 100)
    boot_upper = np.percentile(boot_r2, (1 - boot_alpha / 2) * 100)
    
    # 2. Permutation Test
    perm_r2 = []
    for _ in range(n_permutations):
        shuffled_simulated = np.random.permutation(simulated)
        perm_r2.append(r2_score(observed, shuffled_simulated))
    p_value_perm = np.sum(np.array(perm_r2) >= base_r2) / n_permutations
    
    # 3. Wald Test with Robust Standard Errors
    X = sm.add_constant(simulated)
    model = sm.OLS(observed, X)
    results = model.fit(cov_type=cov_type)
    wald_p_value = results.f_pvalue
    
    # 4. Cross-Validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=cv_random_state)
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
    
    # Generate Report
    report_lines = []
    report_lines.append("Robust R-squared Significance Report")
    report_lines.append("=" * 40)
    report_lines.append(f"Sample size: {n}")
    report_lines.append(f"Baseline R-squared: {base_r2:.4f}")
    report_lines.append("")
    report_lines.append("1. Bootstrapping Test")
    report_lines.append(f"Parameters: iterations={n_bootstraps}, alpha={boot_alpha}")
    report_lines.append(f"{(1-boot_alpha)*100:.0f}% Confidence Interval: [{boot_lower:.4f}, {boot_upper:.4f}]")
    report_lines.append("")
    report_lines.append("2. Permutation Test")
    report_lines.append(f"Parameters: permutations={n_permutations}")
    report_lines.append(f"p-value: {p_value_perm:.4e}")
    report_lines.append("")
    report_lines.append(f"3. Wald Test with Robust Standard Errors")
    report_lines.append(f"Parameters: covariance type='{cov_type}'")
    report_lines.append(f"p-value: {wald_p_value:.4e}")
    report_lines.append("")
    report_lines.append("4. Cross-Validation")
    report_lines.append(f"Parameters: folds={cv_folds}, random_state={cv_random_state}")
    report_lines.append(f"Mean Out-of-Sample R-squared: {mean_cv_r2:.4f}")
    
    with open(file_name, "w") as f:
        f.write(chr(10).join(report_lines))
        
    return file_name
  
