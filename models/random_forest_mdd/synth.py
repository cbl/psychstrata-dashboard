from __future__ import annotations

import numpy as np
import pandas as pd


_DRUGS_BY_CLASS: dict[str, list[str]] = {
    "SSRI": ["fluoxetine", "sertraline", "citalopram", "escitalopram", "paroxetine"],
    "SNRI": ["venlafaxine", "duloxetine"],
    "TCA":  ["amitriptyline", "dosulepin", "lofepramine"],
    "MAOI": ["trazodone"],
    "Other_AD": ["mirtazapine", "trazodone"],
}


def generate_synthetic_dataset(n: int = 2500, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(random_state)

    age_at_first_rx = np.clip(rng.normal(45, 14, n), 18, 80).round().astype(int)
    sex = rng.choice(["Female", "Male"], size=n, p=[0.6, 0.4])
    ethnicity = rng.choice(
        ["White", "Asian", "Black", "Mixed", "Chinese", "Other"],
        size=n, p=[0.92, 0.03, 0.02, 0.01, 0.005, 0.015],
    )

    townsend_deprivation = np.clip(rng.normal(0, 3, n), -7, 12).round(1)
    education = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.12, 0.20, 0.25, 0.20, 0.15])

    age_at_first_psych_dx = np.clip(
        age_at_first_rx - rng.exponential(3, n), 10, 80
    ).round().astype(int)
    n_prior_episodes = rng.poisson(1.5, n).clip(0, 30)
    n_prior_psych_admissions = rng.poisson(0.3, n).clip(0, 20)
    ever_admitted = n_prior_psych_admissions > 0
    total_days_psych_admission_pre_rx = np.where(
        ever_admitted, rng.exponential(30, n) * n_prior_psych_admissions, 0,
    ).clip(0, 730).round().astype(int)
    days_since_last_psych_admission = np.where(
        ever_admitted, rng.uniform(0, 3650, n), 3650,
    ).round().astype(int)
    prior_self_harm = rng.binomial(1, 0.08, n)
    prior_ect = rng.binomial(1, 0.02, n)
    prior_involuntary_admission = rng.binomial(1, 0.05, n)

    neuroticism_score = np.clip(rng.normal(5, 3, n), 0, 12).round().astype(int)

    anxiety_p = 0.15 + 0.02 * neuroticism_score
    comorbid_anxiety = rng.binomial(1, np.clip(anxiety_p, 0, 1))
    comorbid_substance_use = rng.binomial(1, 0.08, n)
    comorbid_personality_disorder = rng.binomial(1, 0.04, n)
    comorbid_ptsd = rng.binomial(1, 0.06, n)

    comorbid_thyroid_dx = rng.binomial(1, 0.08, n)
    comorbid_diabetes = rng.binomial(1, 0.10, n)
    comorbid_chronic_pain = rng.binomial(1, 0.15, n)
    multimorbidity_count = rng.poisson(0.5, n).clip(0, 10)

    time_rx_vs_dx_years = np.clip(rng.normal(0.5, 2.0, n), -1.0, 30.0).round(2)
    rx_before_dx = (time_rx_vs_dx_years < 0).astype(int)
    n_prior_prescriptions = rng.poisson(5, n).clip(0, 500)
    n_prior_benzodiazepine_rx = rng.poisson(1.5, n).clip(0, 200)
    n_prior_hypnotic_rx = rng.poisson(0.8, n).clip(0, 200)
    n_prior_opioid_rx = rng.poisson(1.0, n).clip(0, 200)
    n_prior_psych_classes_tried = rng.poisson(1.0, n).clip(0, 10)
    year_first_rx = rng.integers(2000, 2021, n)

    first_rx_class = rng.choice(
        ["SSRI", "SNRI", "TCA", "MAOI", "Other_AD"],
        size=n, p=[0.65, 0.15, 0.10, 0.02, 0.08],
    )
    first_rx_drug = np.array([
        rng.choice(_DRUGS_BY_CLASS[c]) for c in first_rx_class
    ])

    n_gp_visits_pre_rx_12mo = rng.poisson(5, n).clip(0, 100)
    n_distinct_read_codes_pre_rx_12mo = rng.poisson(10, n).clip(0, 200)
    years_gp_registration_pre_rx = np.clip(rng.normal(10, 5, n), 0, 40).round(1)

    bmi = np.clip(rng.normal(27, 5, n), 15, 60).round(1)
    smoking_status = rng.choice(["Never", "Previous", "Current"], size=n, p=[0.55, 0.28, 0.17])
    alcohol_freq = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.10, 0.15, 0.20, 0.25, 0.20, 0.10])
    employment = rng.choice(
        ["Employed", "Retired", "Unable", "Other"],
        size=n, p=[0.55, 0.25, 0.12, 0.08],
    )
    live_alone = rng.binomial(1, 0.20, n)

    crp = np.clip(rng.lognormal(0.4, 0.8, n), 0.1, 30.0).round(2)
    hba1c_base = rng.normal(36, 4, n)
    hba1c = np.clip(hba1c_base + 10 * comorbid_diabetes, 20, 120).round(1)
    vitamin_d = np.clip(rng.normal(50, 22, n), 5, 200).round(1)
    neutrophil_lymphocyte_ratio = np.clip(rng.lognormal(0.7, 0.4, n), 0.5, 10.0).round(2)

    family_history_depression = rng.binomial(1, 0.25, n)
    family_history_severe_mental_illness = rng.binomial(1, 0.10, n)

    logit = (
        -2.4
        + 0.80 * prior_self_harm
        + 0.70 * prior_ect
        + 0.10 * np.minimum(n_prior_benzodiazepine_rx, 10)
        + 0.08 * neuroticism_score
        + 0.30 * n_prior_psych_classes_tried
        + 0.15 * np.minimum(n_prior_episodes, 10)
        + 0.20 * np.minimum(n_prior_psych_admissions, 5)
        + 0.30 * prior_involuntary_admission
        + 0.35 * comorbid_substance_use
        + 0.45 * comorbid_personality_disorder
        + 0.20 * comorbid_anxiety
        + 0.30 * comorbid_ptsd
        + 0.20 * comorbid_chronic_pain
        + 0.15 * multimorbidity_count
        + 0.08 * np.minimum(crp, 10)
        + 0.05 * townsend_deprivation
        - 0.08 * education
        - 0.005 * (age_at_first_rx - 45)
        + 0.05 * (sex == "Female")
        + 0.03 * alcohol_freq
        + 0.15 * (smoking_status == "Current")
        + 0.01 * (bmi - 25)
        + 0.05 * np.minimum(n_prior_opioid_rx, 10)
        + 0.10 * (first_rx_class == "TCA")
        - 0.0001 * days_since_last_psych_admission
        + 0.25 * family_history_severe_mental_illness
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob, n)

    df = pd.DataFrame({
        "age_at_first_rx": age_at_first_rx,
        "sex": sex,
        "ethnicity": ethnicity,
        "townsend_deprivation": townsend_deprivation,
        "education": education,
        "age_at_first_psych_dx": age_at_first_psych_dx,
        "n_prior_episodes": n_prior_episodes,
        "n_prior_psych_admissions": n_prior_psych_admissions,
        "total_days_psych_admission_pre_rx": total_days_psych_admission_pre_rx,
        "days_since_last_psych_admission": days_since_last_psych_admission,
        "prior_self_harm": prior_self_harm,
        "prior_ect": prior_ect,
        "prior_involuntary_admission": prior_involuntary_admission,
        "neuroticism_score": neuroticism_score,
        "comorbid_anxiety": comorbid_anxiety,
        "comorbid_substance_use": comorbid_substance_use,
        "comorbid_personality_disorder": comorbid_personality_disorder,
        "comorbid_ptsd": comorbid_ptsd,
        "comorbid_thyroid_dx": comorbid_thyroid_dx,
        "comorbid_diabetes": comorbid_diabetes,
        "comorbid_chronic_pain": comorbid_chronic_pain,
        "multimorbidity_count": multimorbidity_count,
        "time_rx_vs_dx_years": time_rx_vs_dx_years,
        "rx_before_dx": rx_before_dx,
        "n_prior_prescriptions": n_prior_prescriptions,
        "n_prior_benzodiazepine_rx": n_prior_benzodiazepine_rx,
        "n_prior_hypnotic_rx": n_prior_hypnotic_rx,
        "n_prior_opioid_rx": n_prior_opioid_rx,
        "n_prior_psych_classes_tried": n_prior_psych_classes_tried,
        "year_first_rx": year_first_rx,
        "first_rx_class": first_rx_class,
        "first_rx_drug": first_rx_drug,
        "n_gp_visits_pre_rx_12mo": n_gp_visits_pre_rx_12mo,
        "n_distinct_read_codes_pre_rx_12mo": n_distinct_read_codes_pre_rx_12mo,
        "years_gp_registration_pre_rx": years_gp_registration_pre_rx,
        "bmi": bmi,
        "smoking_status": smoking_status,
        "alcohol_freq": alcohol_freq,
        "employment": employment,
        "live_alone": live_alone,
        "crp": crp,
        "hba1c": hba1c,
        "vitamin_d": vitamin_d,
        "neutrophil_lymphocyte_ratio": neutrophil_lymphocyte_ratio,
        "family_history_depression": family_history_depression,
        "family_history_severe_mental_illness": family_history_severe_mental_illness,
    })
    return df, pd.Series(y, name="treatment_resistant")
