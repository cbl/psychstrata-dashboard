from __future__ import annotations

from ..base import FeatureSpec, Option


def _opts(*pairs: tuple[str, object]) -> tuple[Option, ...]:
    return tuple(Option(label=lbl, value=val) for lbl, val in pairs)


DEMOGRAPHICS = [
    FeatureSpec(
        id="age_at_first_rx", label="Age at first prescription (years)",
        category="Demographics", kind="numeric",
        default=45, min=18, max=80, step=1,
    ),
    FeatureSpec(
        id="sex", label="Sex",
        category="Demographics", kind="categorical", default="Female",
        options=_opts(("Female", "Female"), ("Male", "Male")),
    ),
    FeatureSpec(
        id="ethnicity", label="Ethnicity",
        category="Demographics", kind="categorical", default="White",
        options=_opts(
            ("White", "White"), ("Asian", "Asian"), ("Black", "Black"),
            ("Mixed", "Mixed"), ("Chinese", "Chinese"), ("Other", "Other"),
        ),
    ),
]

SOCIOECONOMIC = [
    FeatureSpec(
        id="townsend_deprivation", label="Townsend deprivation index",
        category="Socioeconomic", kind="numeric",
        default=0.0, min=-7.0, max=12.0, step=0.1,
        description="Area-level deprivation at recruitment. Higher = more deprived.",
    ),
    FeatureSpec(
        id="education", label="Education",
        category="Socioeconomic", kind="ordinal", default=3,
        options=_opts(
            ("None", 0), ("CSE", 1), ("O-levels / GCSE", 2),
            ("A-levels / equivalent", 3), ("Vocational / college", 4),
            ("Higher degree", 5),
        ),
    ),
]

CLINICAL_SEVERITY = [
    FeatureSpec(
        id="age_at_first_psych_dx", label="Age at first psychiatric diagnosis (years)",
        category="Clinical severity", kind="numeric",
        default=35, min=10, max=80, step=1,
    ),
    FeatureSpec(
        id="n_prior_episodes", label="Prior depressive episodes (count)",
        category="Clinical severity", kind="numeric",
        default=1, min=0, max=30, step=1,
    ),
    FeatureSpec(
        id="n_prior_psych_admissions", label="Prior psychiatric admissions (count)",
        category="Clinical severity", kind="numeric",
        default=0, min=0, max=20, step=1,
    ),
    FeatureSpec(
        id="total_days_psych_admission_pre_rx", label="Total days in psychiatric admission",
        category="Clinical severity", kind="numeric",
        default=0, min=0, max=730, step=1,
    ),
    FeatureSpec(
        id="days_since_last_psych_admission", label="Days since last psychiatric admission",
        category="Clinical severity", kind="numeric",
        default=3650, min=0, max=10000, step=30,
        description="Use a large value (e.g. 3650) for never-admitted.",
    ),
    FeatureSpec(
        id="prior_self_harm", label="Prior self-harm",
        category="Clinical severity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="prior_ect", label="Prior ECT",
        category="Clinical severity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="prior_involuntary_admission", label="Prior involuntary admission",
        category="Clinical severity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
]

PERSONALITY = [
    FeatureSpec(
        id="neuroticism_score", label="Neuroticism (EPQ-R-S N-12)",
        category="Personality", kind="numeric",
        default=5, min=0, max=12, step=1,
    ),
]

PSYCH_COMORBID = [
    FeatureSpec(
        id="comorbid_anxiety", label="Comorbid anxiety",
        category="Psychiatric comorbidity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="comorbid_substance_use", label="Comorbid substance use disorder",
        category="Psychiatric comorbidity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="comorbid_personality_disorder", label="Comorbid personality disorder",
        category="Psychiatric comorbidity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="comorbid_ptsd", label="Comorbid PTSD",
        category="Psychiatric comorbidity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
]

PHYSICAL_COMORBID = [
    FeatureSpec(
        id="comorbid_thyroid_dx", label="Thyroid disorder",
        category="Physical comorbidity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="comorbid_diabetes", label="Diabetes",
        category="Physical comorbidity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="comorbid_chronic_pain", label="Chronic pain",
        category="Physical comorbidity", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="multimorbidity_count", label="Other chronic conditions (count)",
        category="Physical comorbidity", kind="numeric",
        default=0, min=0, max=10, step=1,
    ),
]

TREATMENT_HISTORY = [
    FeatureSpec(
        id="time_rx_vs_dx_years", label="Time between first Rx and first diagnosis (years)",
        category="Treatment history", kind="numeric",
        default=0.0, min=-1.0, max=30.0, step=0.1,
        description="Negative if Rx precedes diagnosis.",
    ),
    FeatureSpec(
        id="rx_before_dx", label="First Rx before first diagnosis",
        category="Treatment history", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="n_prior_prescriptions", label="Prior psychiatric prescriptions (count)",
        category="Treatment history", kind="numeric",
        default=0, min=0, max=500, step=1,
    ),
    FeatureSpec(
        id="n_prior_benzodiazepine_rx", label="Prior benzodiazepine prescriptions (count)",
        category="Treatment history", kind="numeric",
        default=0, min=0, max=200, step=1,
    ),
    FeatureSpec(
        id="n_prior_hypnotic_rx", label="Prior Z-drug / hypnotic prescriptions (count)",
        category="Treatment history", kind="numeric",
        default=0, min=0, max=200, step=1,
    ),
    FeatureSpec(
        id="n_prior_opioid_rx", label="Prior opioid prescriptions (count)",
        category="Treatment history", kind="numeric",
        default=0, min=0, max=200, step=1,
    ),
    FeatureSpec(
        id="n_prior_psych_classes_tried", label="Distinct psychiatric drug classes tried",
        category="Treatment history", kind="numeric",
        default=0, min=0, max=10, step=1,
    ),
    FeatureSpec(
        id="year_first_rx", label="Calendar year of first prescription",
        category="Treatment history", kind="numeric",
        default=2015, min=1995, max=2023, step=1,
    ),
]

INDEX_TREATMENT = [
    FeatureSpec(
        id="first_rx_class", label="First antidepressant class",
        category="Index treatment", kind="categorical", default="SSRI",
        options=_opts(
            ("SSRI", "SSRI"), ("SNRI", "SNRI"), ("TCA", "TCA"),
            ("MAOI", "MAOI"), ("Other", "Other_AD"),
        ),
    ),
    FeatureSpec(
        id="first_rx_drug", label="First antidepressant drug",
        category="Index treatment", kind="categorical", default="sertraline",
        options=_opts(
            ("Fluoxetine", "fluoxetine"),
            ("Sertraline", "sertraline"),
            ("Citalopram", "citalopram"),
            ("Escitalopram", "escitalopram"),
            ("Paroxetine", "paroxetine"),
            ("Venlafaxine", "venlafaxine"),
            ("Duloxetine", "duloxetine"),
            ("Amitriptyline", "amitriptyline"),
            ("Dosulepin", "dosulepin"),
            ("Lofepramine", "lofepramine"),
            ("Mirtazapine", "mirtazapine"),
            ("Trazodone", "trazodone"),
        ),
    ),
]

HEALTHCARE_ENGAGEMENT = [
    FeatureSpec(
        id="n_gp_visits_pre_rx_12mo", label="GP visits in 12mo before first Rx",
        category="Healthcare engagement", kind="numeric",
        default=4, min=0, max=100, step=1,
    ),
    FeatureSpec(
        id="n_distinct_read_codes_pre_rx_12mo", label="Distinct Read codes in 12mo before first Rx",
        category="Healthcare engagement", kind="numeric",
        default=8, min=0, max=200, step=1,
    ),
    FeatureSpec(
        id="years_gp_registration_pre_rx", label="Years of GP registration before first Rx",
        category="Healthcare engagement", kind="numeric",
        default=5.0, min=0.0, max=40.0, step=0.5,
    ),
]

LIFESTYLE = [
    FeatureSpec(
        id="bmi", label="BMI (kg/m²)",
        category="Lifestyle", kind="numeric",
        default=26.0, min=15.0, max=60.0, step=0.1,
    ),
    FeatureSpec(
        id="smoking_status", label="Smoking status",
        category="Lifestyle", kind="categorical", default="Never",
        options=_opts(
            ("Never", "Never"), ("Previous", "Previous"), ("Current", "Current"),
        ),
    ),
    FeatureSpec(
        id="alcohol_freq", label="Alcohol frequency",
        category="Lifestyle", kind="ordinal", default=3,
        options=_opts(
            ("Never", 0), ("Special occasions only", 1),
            ("1–3 times/month", 2), ("Once/week", 3),
            ("2–4 times/week", 4), ("Daily / almost daily", 5),
        ),
    ),
    FeatureSpec(
        id="employment", label="Employment",
        category="Lifestyle", kind="categorical", default="Employed",
        options=_opts(
            ("Employed", "Employed"), ("Retired", "Retired"),
            ("Unable to work", "Unable"), ("Other", "Other"),
        ),
    ),
    FeatureSpec(
        id="live_alone", label="Lives alone",
        category="Lifestyle", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
]

BIOMARKERS = [
    FeatureSpec(
        id="crp", label="C-reactive protein (mg/L)",
        category="Biomarkers", kind="numeric",
        default=1.5, min=0.1, max=30.0, step=0.1,
    ),
    FeatureSpec(
        id="hba1c", label="HbA1c (mmol/mol)",
        category="Biomarkers", kind="numeric",
        default=36.0, min=20.0, max=120.0, step=0.5,
    ),
    FeatureSpec(
        id="vitamin_d", label="Vitamin D (nmol/L)",
        category="Biomarkers", kind="numeric",
        default=50.0, min=5.0, max=200.0, step=1.0,
    ),
    FeatureSpec(
        id="neutrophil_lymphocyte_ratio", label="Neutrophil/lymphocyte ratio",
        category="Biomarkers", kind="numeric",
        default=2.0, min=0.5, max=10.0, step=0.1,
    ),
]

FAMILY_HISTORY = [
    FeatureSpec(
        id="family_history_depression", label="Family history of depression",
        category="Family history", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
    FeatureSpec(
        id="family_history_severe_mental_illness", label="Family history of severe mental illness",
        category="Family history", kind="binary", default=0,
        options=_opts(("No", 0), ("Yes", 1)),
    ),
]


MDD_FEATURES: list[FeatureSpec] = [
    *DEMOGRAPHICS,
    *SOCIOECONOMIC,
    *CLINICAL_SEVERITY,
    *PERSONALITY,
    *PSYCH_COMORBID,
    *PHYSICAL_COMORBID,
    *TREATMENT_HISTORY,
    *INDEX_TREATMENT,
    *HEALTHCARE_ENGAGEMENT,
    *LIFESTYLE,
    *BIOMARKERS,
    *FAMILY_HISTORY,
]

CATEGORY_ORDER: list[str] = [
    "Demographics",
    "Socioeconomic",
    "Clinical severity",
    "Personality",
    "Psychiatric comorbidity",
    "Physical comorbidity",
    "Treatment history",
    "Index treatment",
    "Healthcare engagement",
    "Lifestyle",
    "Biomarkers",
    "Family history",
]
