# LDL cholesterol and discharge treatment in type 2 diabetes

Code for the study of inpatient LDL-C and lipid-lowering recommendations at
hospital discharge in Endocrinology and Diabetology, 2017–September 2026.
Hospitalizations are the unit of analysis.

## Data

**This repository contains no real patient data or study results.** The example
records are generated in memory from fixed artificial values. They are not
sampled, altered or reconstructed from patients. All example identifiers start
with `SYNTHETIC-`; the short diagnostic labels are authored examples, not clinical
excerpts. The scripts do not read any external dataset.

The synthetic workflow demonstrates the study code. Its numerical outputs do
not reproduce the paper. Individual-level clinical data cannot be publicly
shared under the study's data governance arrangements.

## Files and use

- `derive.py`: cohort eligibility, ASCVD, laboratory selection and treatment classification.
- `run_analysis.py`: main and supplementary tables, sensitivity analyses and Figure 1.
- `mock_data.py`: six artificial linked input tables.
- `validate.py`: checks of selection, classification, denominators and bootstrap calculations.
- `variable_dictionary.csv`: input fields and principal derived variables.
- `code_lists.csv`: codes and classification rules, including the source-cohort review.

Use Python 3.13.5 and the pinned packages in `requirements.txt`:

```sh
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
python validate.py
```

Aggregate example tables and figures are written to `outputs/`, which is excluded
from Git. Admission-level records stay in memory. The `.gitignore` permits only
the ten source and documentation files listed in it.

## Selection and definitions

The hospital source cohort was identified using ICD-10 E11 in any diagnosis
position in the AMMS system. Diabetes type was reviewed using diagnosis codes
and clinical documentation. All admissions of patients with documented type 1
diabetes, LADA or explicitly secondary diabetes were excluded. E10-only coding
without a documented diabetes type was also excluded. Explicitly documented T2D
was retained despite conflicting E10 coding unless another exclusion applied.
Insulin use alone did not determine diabetes type.

This review preceded the analysis extract. It is documented in `code_lists.csv`,
not recreated as an automated diagnosis algorithm. The code begins with the
reviewed T2D input tables, then selects discharged adults and excludes
admission-linked, positively documented pregnancy. Missing LDL or a missing
discharge medication list does not exclude an admission from cohort description.
Tables from one hospital record system were linked by patient and admission IDs.
The synthetic examples represent these linked tables after diabetes-type review.

ASCVD is classified separately for each admission using the codes in
`code_lists.csv` and the Polish text rules in `derive.py`. Affirmative recorded
history qualifies; negative, uncertain or ruled-out documentation does not.
Status is not carried between admissions. No documented ASCVD does not establish
clinical absence. The code starts from structured diagnosis and history fields;
it does not reconstruct the upstream extraction of assertions from full records.

LDL-C is the first eligible exact numeric inpatient result in mg/dL. Collection
time falls back to execution time; execution must be no later than discharge
day. Ties use execution time and record ID. Bands are <55, 55–<70, 70–<100 and
≥100 mg/dL. HbA1c is the first eligible result and creatinine the last. eGFR uses
the 2021 CKD-EPI creatinine equation. BMI uses calculated encounter BMI, then the
mean of linked plausible inpatient measurements, then documented encounter BMI.
Other lipids use admission means. Non-HDL-C uses total cholesterol and HDL-C
recorded at the same time.

Treatment classification uses eligible discharge recommendations and the
structured ingredient, product and component-dose fields. Stopped, held,
replaced, reference-only and conditional-replacement records are ineligible.
High intensity means atorvastatin 40–80 mg/day or rosuvastatin 20–40 mg/day.
Missing or ambiguous doses remain unknown. An uncertain statin combination can
count as any statin and unknown intensity while remaining unclassified for
regimen. Table denominators retain such admissions; Figure 1 excludes them.
Medication parsing and component-dose resolution preceded this analysis and
are represented by the documented input fields.

## Analyses

The main period is 2023–September 2026; 2017–2022 is summarized separately.
The code produces cohort characteristics, LDL distributions by ASCVD,
discharge treatment overall and in ASCVD with LDL-C ≥55 mg/dL, annual summaries,
other-lipid summaries, and the stacked treatment figure.

Proportions are admission-weighted. Patient-bootstrap 95% confidence intervals
use 5,000 resamples, keeping each sampled patient's admissions together within
the subgroup. Intervals are omitted for 0% or 100% and flagged when fewer than
five patients have or do not have the outcome. The analysis is descriptive:
there are no regression models, hypothesis tests or imputations.

Sensitivities use the first LDL-tested admission per patient in the contemporary
period, admission mean LDL-C, exclusion of 2026, Diabetology only, and exclusion
of LDL results paired at the same recorded time with triglycerides ≥400 mg/dL.
The last analysis selects the first remaining LDL result; unmatched triglyceride
results do not trigger exclusion. LDL measurement methods were unavailable.

The derivations and calculations are adapted from the study scripts. Clinical
file loading and patient-level audit exports are omitted; example captions
identify synthetic output and the figure uses a bundled font for portability.
Code and artificial examples use the MIT License.
