"""Artificial linked tables. No clinical data or patient excerpts are read."""

import numpy as np
import pandas as pd

ID = "encounter_study_id"
PID = "patient_study_id"


def make_tables():
    """Generate deterministic examples from fixed values and arithmetic sequences."""
    tables = {
        k: []
        for k in [
            "encounters",
            "diagnoses",
            "clinical_history",
            "labs",
            "measurements",
            "medications",
        ]
    }
    for i in range(160):
        year = 2017 + i // 16
        date = pd.Timestamp(year=year, month=7, day=10)
        keys = {
            ID: f"SYNTHETIC-ADMISSION-{i:03}",
            PID: f"SYNTHETIC-PATIENT-{i % 80:03}",
        }
        tables["encounters"].append(
            {
                **keys,
                "clinic": "Diabetology" if year >= 2023 and i % 4 else "Endocrinology",
                "admission_date": date,
                "discharge_date": date + pd.Timedelta(days=3),
                "hospitalization_status": "ongoing" if i == 2 else "discharged",
                "sex": "female" if i % 2 else "male",
                "age_at_admission_min_years": 16 if i == 3 else 39 + i % 40,
                "age_at_admission_calendar_years": 17 if i == 3 else 40 + i % 40,
                "discharge_medication_context": "unavailable"
                if i % 19 == 0
                else "discharge_recommendation",
                "calculated_bmi_kg_m2": np.nan if i % 5 == 0 else 24.0 + i % 12,
                "documented_bmi_kg_m2": 27.0 if i % 7 == 0 else np.nan,
            }
        )
        # Source input has already undergone patient-level diabetes-type review.
        tables["diagnoses"].append(
            {
                **keys,
                "diagnosis_record_id": f"SYNTHETIC-DIAGNOSIS-{i:03}",
                "icd10_code": "I25.1" if i % 4 < 2 else "E11.9",
                "diagnosis_label": "choroba niedokrwienna serca"
                if i % 4 < 2
                else "cukrzyca typu 2",
            }
        )
        tables["clinical_history"].append(
            {
                **keys,
                "history_record_id": f"SYNTHETIC-HISTORY-{i:03}",
                "concept": "pregnancy" if i == 1 else "ischemic_stroke",
                "icd10_code": "" if i == 1 else "I63.9",
                "concept_label_source": "ciaza" if i == 1 else "udar niedokrwienny",
                "assertion": "present" if i == 1 else "explicitly_absent",
            }
        )
        tables["measurements"].append(
            {
                **keys,
                "measurement": "body_mass_index",
                "value_numeric": np.nan if i % 10 == 0 else 26.0,
                "value_quality_status": "plausible",
                "encounter_link_status": "linked",
                "measured_at": date,
            }
        )

        def lab(analyte, value, suffix, day=1, **changes):
            record = {
                **keys,
                "analyte": analyte,
                "value": value,
                "unit": "%" if analyte == "hba1c_percent" else "mg/dL",
                "collected_at": date + pd.Timedelta(days=day),
                "executed_at": date + pd.Timedelta(days=day, hours=1),
                "encounter_link_status": "linked",
                "result_status": "parsed",
                "comparison_operator": "=",
                "lab_result_id": f"SYNTHETIC-LAB-{i:03}-{suffix}",
            }
            record.update(changes)
            tables["labs"].append(record)

        if i % 11:
            lab("ldl_cholesterol_mg_dl", [42.0, 55.0, 70.0, 110.0][(i // 4) % 4], "LDL")
            lab("ldl_cholesterol_mg_dl", 90.0, "LDL-SECOND", day=2)
            lab("ldl_cholesterol_mg_dl", 30.0, "LDL-AFTER-DISCHARGE", day=5)
        for analyte, value, suffix in [
            ("total_cholesterol_mg_dl", 180.0 + i % 10, "TC"),
            ("hdl_cholesterol_mg_dl", 45.0 + i % 5, "HDL"),
            ("triglycerides_mg_dl", 400.0 if i % 13 == 0 else 120.0, "TG"),
            ("hba1c_percent", 7.0 + (i % 5) / 10, "HBA1C"),
            ("serum_creatinine_mg_dl", 0.8 + (i % 6) / 10, "CREATININE"),
        ]:
            if suffix == "TG" and i % 17 == 0:
                continue
            lab(analyte, value, suffix)
        lab("serum_creatinine_mg_dl", 1.2, "CREATININE-LAST", day=2)

        def medication(
            ingredient, statin=None, dose=np.nan, product="other", eligible=True
        ):
            tables["medications"].append(
                {
                    **keys,
                    "ingredient": ingredient,
                    "statin_ingredient": statin,
                    "statin_daily_dose_mg": dose,
                    "statin_dose_analysis_eligible": not pd.isna(dose),
                    "statin_regimen_ambiguity": False,
                    "lipid_product_type": product,
                    "medication_class": "fibrate"
                    if ingredient == "fenofibrate"
                    else "other",
                    "medication_context": "discharge_recommendation",
                    "discharge_exposure_eligible": eligible,
                    "medication_use_status": "active" if eligible else "stopped",
                }
            )

        regimen = i % 8
        if regimen == 0:
            medication("atorvastatin", "atorvastatin", 40.0, "single_statin")
        elif regimen == 1:
            medication(
                "rosuvastatin; ezetimibe",
                "rosuvastatin",
                20.0,
                "statin_ezetimibe_combination",
            )
        elif regimen == 2:
            medication("ezetimibe", product="single_ezetimibe")
        elif regimen == 3:
            medication("metformin")
        elif regimen == 4:
            medication(
                "rosuvastatin", "rosuvastatin", product="ambiguous_statin_product"
            )
        elif regimen == 5:
            medication("atorvastatin", "atorvastatin", product="single_statin")
        elif regimen == 6:
            medication("atorvastatin", "atorvastatin", 10.0, "single_statin")
        else:
            medication("rosuvastatin", "rosuvastatin", 10.0, "single_statin")
            medication("ezetimibe", product="single_ezetimibe")
        medication(
            "atorvastatin", "atorvastatin", 80.0, "single_statin", eligible=False
        )
        if i % 23 == 0:
            medication("fenofibrate")
        if i % 37 == 0:
            medication("evolocumab")
        if i % 41 == 0:
            medication("bempedoic acid")
    return {name: pd.DataFrame(rows) for name, rows in tables.items()}
