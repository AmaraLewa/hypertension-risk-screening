"""Synthetic demonstration profiles.

These are invented, illustrative inputs. They are NOT NHANES participants and
were not drawn from the cohort. No category is assigned here: every result is
produced by the trained model when the profile is run.
"""

DEMO_PROFILES = {
    "Profile A — Synthetic: 19-year-old, slim build, non-smoker": {
        "age": 19, "waist_cm": 72.0, "hip_cm": 100.0, "cigarettes_per_day": 0,
        "alcohol_frequency": "Less than once a month",
    },
    "Profile B — Synthetic: 29-year-old, average build, non-smoker": {
        "age": 29, "waist_cm": 88.0, "hip_cm": 104.0, "cigarettes_per_day": 0,
        "alcohol_frequency": "1–2 times a week",
    },
    "Profile C — Synthetic: 37-year-old, central adiposity, smoker, daily drinker": {
        "age": 37, "waist_cm": 102.0, "hip_cm": 105.0, "cigarettes_per_day": 12,
        "alcohol_frequency": "Daily or almost daily",
    },
}
