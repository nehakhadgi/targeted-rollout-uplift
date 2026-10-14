import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from causalml.inference.tree import UpliftRandomForestClassifier
from causalml.metrics import qini_score, plot_qini
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_effectsize,
    test_proportions_2indep,
)

# --- Constants & Config ---
SEED = 20261013
N_CUSTOMERS = 30000
FEATURES = ["new_customer", "mobile", "prior_purchases", "discount_sensitivity", "region_west"]
VALUE_PER_CONVERSION = 100.0
COST_PER_TREATMENT = 1.50

def generate_simulation_data():
    print("[1/5] Generating simulated experiment dataset...")
    rng = np.random.default_rng(SEED)
    new_customer = rng.binomial(1, 0.40, size=N_CUSTOMERS)
    mobile = rng.binomial(1, 0.60, size=N_CUSTOMERS)
    prior_purchases = np.where(
        new_customer == 1,
        rng.poisson(0.6, size=N_CUSTOMERS),
        rng.poisson(6.0, size=N_CUSTOMERS),
    )
    discount_sensitivity = rng.beta(2.0, 3.0, size=N_CUSTOMERS)
    region_west = rng.binomial(1, 0.30, size=N_CUSTOMERS)
    treated = rng.binomial(1, 0.50, size=N_CUSTOMERS)
    treatment = np.where(treated == 1, "treatment", "control")
    
    baseline_logit = (
        -2.0
        - 0.35 * new_customer
        + 0.18 * mobile
        + 0.05 * np.minimum(prior_purchases, 10)
        - 0.25 * discount_sensitivity
        + 0.10 * region_west
    )
    baseline_probability = 1 / (1 + np.exp(-baseline_logit))
    true_uplift = np.where(new_customer == 1, 0.06, -0.02) + 0.01 * mobile * new_customer
    
    conversion_probability = np.clip(baseline_probability + treated * true_uplift, 0.01, 0.95)
    conversion = rng.binomial(1, conversion_probability, size=N_CUSTOMERS)
    
    return pd.DataFrame({
        "customer_id": np.arange(1, N_CUSTOMERS + 1),
        "treatment": treatment,
        "conversion": conversion,
        "new_customer": new_customer,
        "customer_segment": np.where(new_customer == 1, "New", "Loyal"),
        "mobile": mobile,
        "prior_purchases": prior_purchases,
        "discount_sensitivity": discount_sensitivity,
        "region_west": region_west,
        "true_uplift": true_uplift,
    })

def ab_summary(frame):
    treated_rows = frame[frame["treatment"] == "treatment"]
    control_rows = frame[frame["treatment"] == "control"]
    treatment_successes = int(treated_rows["conversion"].sum())
    control_successes = int(control_rows["conversion"].sum())
    treatment_n, control_n = len(treated_rows), len(control_rows)
    treatment_rate = treatment_successes / treatment_n
    control_rate = control_successes / control_n

    test_result = test_proportions_2indep(
        treatment_successes, treatment_n, control_successes, control_n,
        method="score", compare="diff", alternative="two-sided", correction=True
    )
    ci_low, ci_high = confint_proportions_2indep(
        treatment_successes, treatment_n, control_successes, control_n,
        method="score", compare="diff", alpha=0.05, correction=True
    )
    return pd.Series({
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_lift": treatment_rate - control_rate,
        "relative_lift": treatment_rate / control_rate - 1,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": test_result.pvalue,
        "treatment_n": treatment_n,
        "control_n": control_n
    })

def policy_value(frame, target_actions, assignment_probability=0.50):
    observed_actions = (frame["treatment"] == "treatment").astype(int).to_numpy()
    target_actions = np.asarray(target_actions, dtype=int)
    matched = observed_actions == target_actions
    weighted_outcomes = frame["conversion"].to_numpy() * matched / assignment_probability
    return weighted_outcomes.mean()

if __name__ == "__main__":
    # 1. Prepare data
    df = generate_simulation_data()
    
    # 2. Base A/B Test Summary
    print("\n[2/5] Performing base A/B evaluation...")
    overall = ab_summary(df)
    print(overall.to_frame("Overall Experiment Results").T)
    
    # 3. Model Training & Scoring
    print("\n[3/5] Splitting data and training Uplift RandomForest Classifier...")
    train, test = train_test_split(df, test_size=0.30, random_state=111, stratify=df["treatment"])
    train = train.copy()
    test = test.copy()
    
    uplift_model = UpliftRandomForestClassifier(
        control_name="control", n_estimators=30, max_features=5, random_state=42,
        max_depth=4, min_samples_leaf=200, min_samples_treatment=100, n_reg=100,
        evaluationFunction="KL", normalization=True, honesty=True, estimation_sample_size=0.50
    )
    uplift_model.fit(
        X=train[FEATURES].to_numpy(),
        treatment=train["treatment"].to_numpy(),
        y=train["conversion"].to_numpy()
    )
    test["uplift_score"] = uplift_model.predict(test[FEATURES].to_numpy()).ravel()
    
    # 4. Evaluate Causal Policies
    print("\n[4/5] Evaluating and validating personalization policies...")
    policies = {
        "No rollout": np.zeros(len(test), dtype=int),
        "Full rollout": np.ones(len(test), dtype=int),
        "New customers only": test["new_customer"].to_numpy(),
        "Positive predicted uplift": (test["uplift_score"] > 0).astype(int).to_numpy(),
    }
    
    policy_table = pd.DataFrame({
        "estimated_conversion": {name: policy_value(test, actions) for name, actions in policies.items()},
        "target_share": {name: np.mean(actions) for name, actions in policies.items()}
    })
    policy_table["incremental_vs_control_pp"] = 100 * (
        policy_table["estimated_conversion"] - policy_table.loc["No rollout", "estimated_conversion"]
    )
    print(policy_table.sort_values("estimated_conversion", ascending=False))
    
    # 5. Profit Optimization
    print("\n[5/5] Performing profit-aware evaluation...")
    break_even_uplift = COST_PER_TREATMENT / VALUE_PER_CONVERSION
    policies["Profit-aware uplift"] = (test["uplift_score"] > break_even_uplift).astype(int).to_numpy()
    
    profit_table = pd.DataFrame({
        "estimated_conversion": {name: policy_value(test, actions) for name, actions in policies.items()},
        "target_share": {name: np.mean(actions) for name, actions in policies.items()}
    })
    profit_table["net_value_per_customer"] = (
        VALUE_PER_CONVERSION * profit_table["estimated_conversion"] - COST_PER_TREATMENT * profit_table["target_share"]
    )
    print(f"Break-even Threshold: {break_even_uplift:.2%}")
    print(profit_table.sort_values("net_value_per_customer", ascending=False))
    print("\nPipeline execution completed successfully!")
