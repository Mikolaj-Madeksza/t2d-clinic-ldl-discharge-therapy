"""Study derivations, using artificial inputs for the public demonstration."""

from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd
from mock_data import make_tables

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "outputs"
ID, PID = "encounter_study_id", "patient_study_id"
BANDS = ["<55", "55–<70", "70–<100", "≥100"]
REGIMENS = [
    "Statin without ezetimibe",
    "Statin + ezetimibe",
    "Ezetimibe without statin",
    "Neither statin nor ezetimibe",
    "Uncertain regimen",
]

PATTERNS = {
    "cad": r"chorob\w*\s+niedokrwien\w*\s+serc|(?:angioplast|\bpoba\b|\bpta\b|lima).{0,55}(?:\blad\b|\brca\b|\bcx\b|\bom1?\b|\bgpz\b|\bgo\b)|wiencow|wienczow|zawal\w*\s+(?:miesnia\s+)?serc|\b(?:stemi|nstemi|cabg|ptca|pcta|pci)\b|(?:stent|pomost).{0,65}(?:\blad\b|\brca\b|\bcx\b|serca|wiencow)",
    "ischemic_stroke": r"udar\w*.{0,45}niedokrw|niedokrw\w*.{0,35}udar|\btia\b|przemijaj\w*.{0,25}niedokrw\w*.{0,15}mozg",
    "pad": r"miazdzyc\w*.{0,45}konczyn|niedokrw\w*.{0,35}konczyn|(?:stent|angioplast|rewaskular|pomost|\bpta\b).{0,90}(?:udow|podkolan|biodrow|piszczel|\bsfa\b)|(?:udow|podkolan|biodrow|piszczel|\bsfa\b).{0,70}(?:stent|angioplast|pomost)|protezy?\s+aortalno.dwuudow",
    "unspecified_stroke": r"udar\w*.{0,35}(?:mozg|oun)|(?:po\s+|przebyt\w*\s+)udar",
}


def normalize(text):
    return "".join(
        c
        for c in unicodedata.normalize("NFKD", str(text).lower().replace("ł", "l"))
        if not unicodedata.combining(c)
    )


def code_flags(code):
    c = str(code).upper().strip()
    return {
        "cad": bool(re.match(r"^(I2[0-5](?:\.|$)|Z95\.[15](?:\D|$))", c)),
        "ischemic_stroke": bool(
            re.match(r"^(I63|I65|I66)(?:\.|$)|^I69\.3(?:\D|$)|^G45(?:\.|$)", c)
        )
        and not c.startswith("G45.4"),
        "pad": bool(re.match(r"^I70\.2(?:\D|$)", c)),
    }


def text_flags(text, code=""):
    t = normalize(text)
    flags = {k: bool(re.search(v, t)) for k, v in PATTERNS.items()}
    for c in [str(code)] + re.findall(r"\b[IGZ]\d{2}(?:\.\d+)?\b", str(text).upper()):
        for k, v in code_flags(c).items():
            flags[k] |= v
    # Exclude heatstroke/hemorrhage as an ischemic basis unless a separate explicit TIA exists.
    if re.search(r"ciepln|krwotocz|krwawien.{0,15}mozg", t) and not re.search(
        r"\btia\b|niedokrw", t
    ):
        flags["ischemic_stroke"] = False
        flags["unspecified_stroke"] = False
    flags["unspecified_stroke"] &= not flags["ischemic_stroke"]
    return flags


def disease_status(e, history, diagnoses):
    rows = []
    for source, d, label, rid in [
        ("history", history, "concept_label_source", "history_record_id"),
        ("diagnoses", diagnoses, "diagnosis_label", "diagnosis_record_id"),
    ]:
        for r in d[d[ID].isin(e[ID])].to_dict("records"):
            flags = text_flags(r[label], r["icd10_code"])
            if not any(flags.values()):
                continue
            uncertain = bool(
                re.search(
                    r"podejrze|wyklucz|bez cech|nie stwierdz", normalize(r[label])
                )
            )
            assertion = r.get("assertion", "present")
            qualifies = any(flags[k] for k in ["cad", "ischemic_stroke", "pad"])
            rows.append(
                {
                    ID: r[ID],
                    PID: r[PID],
                    "source": source,
                    "record_id": r[rid],
                    "source_text": r[label],
                    "code": r["icd10_code"],
                    "assertion": assertion,
                    "text_uncertain": uncertain,
                    "qualifies": qualifies,
                    "positive": qualifies and assertion == "present" and not uncertain,
                    **flags,
                }
            )
    audit = pd.DataFrame(rows)
    positive = set(audit.loc[audit.positive, ID])
    e["ascvd"] = e[ID].isin(positive)
    e["ascvd_group"] = np.where(e.ascvd, "Documented ASCVD", "No documented ASCVD")
    e["unspecified_stroke"] = e[ID].isin(
        audit.loc[audit.unspecified_stroke & audit.assertion.eq("present"), ID]
    )
    return e


def select_labs(e, labs):
    names = {
        "ldl": ("ldl_cholesterol_mg_dl", "mg/dL"),
        "tc": ("total_cholesterol_mg_dl", "mg/dL"),
        "hdl": ("hdl_cholesterol_mg_dl", "mg/dL"),
        "tg": ("triglycerides_mg_dl", "mg/dL"),
        "hba1c": ("hba1c_percent", "%"),
        "creatinine": ("serum_creatinine_mg_dl", "mg/dL"),
    }
    q = labs[labs.analyte.isin([v[0] for v in names.values()])].copy()
    q = q.merge(
        e[[ID, PID, "admission_date", "discharge_date"]],
        on=[ID, PID],
        how="left",
        validate="many_to_one",
    )
    for c in ["collected_at", "executed_at"]:
        q[c] = pd.to_datetime(q[c])
    q["time"] = q.collected_at.fillna(q.executed_at)
    q["execution_fallback"] = q.collected_at.isna()
    q["reason"] = "included"
    units = {a: u for a, u in names.values()}
    checks = [
        ("outside_cohort", q.admission_date.isna()),
        ("unlinked", ~q.encounter_link_status.eq("linked")),
        (
            "not_exact",
            ~q.result_status.eq("parsed")
            | q.value.isna()
            | ~q.comparison_operator.fillna("=").eq("="),
        ),
        ("nonpositive", q.value.le(0)),
        ("unit", ~q.unit.eq(q.analyte.map(units))),
        ("missing_time", q.time.isna() | q.executed_at.isna()),
        (
            "outside_admission",
            ~q.time.dt.normalize().between(q.admission_date, q.discharge_date),
        ),
        ("after_discharge", q.executed_at.dt.normalize().gt(q.discharge_date)),
    ]
    for reason, mask in checks:
        q.loc[q.reason.eq("included") & mask, "reason"] = reason
    selected = {}
    for name, (analyte, unit) in names.items():
        d = (
            q[q.reason.eq("included") & q.analyte.eq(analyte)]
            .sort_values(["time", "executed_at", "lab_result_id"])
            .copy()
        )
        assert d.lab_result_id.is_unique
        selected[name] = d
        first = d.drop_duplicates(ID)
        last = d.drop_duplicates(ID, keep="last")
        e[name] = e[ID].map(
            (last if name == "creatinine" else first).set_index(ID).value
        )
        e[name + "_mean"] = e[ID].map(d.groupby(ID).value.mean())
        e[name + "_record"] = e[ID].map(
            (last if name == "creatinine" else first).set_index(ID).lab_result_id
        )
    # Match recorded times; any paired triglyceride result ≥400 triggers exclusion.
    tg = selected["tg"].groupby([ID, PID, "time"]).value.max().rename("paired_tg")
    ldl = selected["ldl"].join(tg, on=[ID, PID, "time"])
    ldl["high_tg"] = ldl.paired_tg.ge(400)
    safe = ldl[~ldl.high_tg].drop_duplicates(ID)
    e["ldl_without_high_tg"] = e[ID].map(safe.set_index(ID).value)
    first = ldl.drop_duplicates(ID)
    e["ldl_paired_tg"] = e[ID].map(first.set_index(ID).paired_tg)
    e["ldl_without_high_tg_record"] = e[ID].map(safe.set_index(ID).lab_result_id)
    tc = selected["tc"].groupby([ID, PID, "time"]).value.mean().rename("tc")
    hdl = selected["hdl"].groupby([ID, PID, "time"]).value.mean().rename("hdl")
    pairs = tc.to_frame().join(hdl, how="inner").reset_index()
    pairs["non_hdl"] = pairs.tc - pairs.hdl
    assert pairs.non_hdl.ge(0).all(), "Negative non-HDL value"
    e["non_hdl"] = e[ID].map(pairs.groupby(ID).non_hdl.mean())
    e["egfr"] = egfr(e.creatinine, e.age, e.sex.eq("female"))
    q.groupby(["analyte", "reason", "execution_fallback"], dropna=False).size().rename(
        "records"
    ).reset_index().to_csv(RESULTS / "laboratory_checks.csv", index=False)
    return e, ldl


def egfr(creatinine, age, female):
    k = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = creatinine / k
    return (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** (-1.2)
        * 0.9938**age
        * np.where(female, 1.012, 1)
    )


def bmi(e, measurements):
    q = measurements[
        measurements.measurement.eq("body_mass_index")
        & measurements.encounter_link_status.eq("linked")
        & measurements.value_quality_status.eq("plausible")
    ]
    q = q.merge(
        e[[ID, PID, "admission_date", "discharge_date"]],
        on=[ID, PID],
        validate="many_to_one",
    )
    q = q[
        pd.to_datetime(q.measured_at)
        .dt.normalize()
        .between(q.admission_date, q.discharge_date)
    ]
    e["bmi"] = e.calculated_bmi_kg_m2.fillna(
        e[ID].map(q.groupby(ID).value_numeric.mean())
    ).fillna(e.documented_bmi_kg_m2)
    return e


def medications(e, m):
    q = m.merge(e[[ID, PID, "regimen_available"]], on=[ID, PID], validate="many_to_one")
    active = (
        q.regimen_available
        & q.discharge_exposure_eligible.eq(True)
        & q.medication_context.eq("discharge_recommendation")
    )
    q["active"] = active
    d = q[active].copy()
    assert not d.medication_use_status.isin(
        [
            "stopped",
            "temporarily_held",
            "replaced",
            "reference_only",
            "conditional_replacement",
        ]
    ).any()
    ing = d.ingredient.fillna("").str.lower()
    d["statin"] = d.statin_ingredient.notna()
    d["ezetimibe"] = d.lipid_product_type.isin(
        ["single_ezetimibe", "statin_ezetimibe_combination"]
    ) | ing.str.contains("ezetim")
    d["uncertain_product"] = d.lipid_product_type.eq("ambiguous_statin_product")
    d["fibrate"] = ing.str.contains("fibrat") | d.medication_class.eq("fibrate")
    d["pcsk9"] = ing.str.contains("aliroc|evoloc|inclis")
    d["bempedoic"] = ing.str.contains("bemped")
    for flag in [
        "statin",
        "ezetimibe",
        "uncertain_product",
        "fibrate",
        "pcsk9",
        "bempedoic",
    ]:
        e[flag] = e[ID].isin(d.loc[d[flag], ID])
    e["regimen"] = np.select(
        [
            ~e.regimen_available,
            e.uncertain_product,
            e.statin & e.ezetimibe,
            e.statin,
            e.ezetimibe,
        ],
        [
            "Unavailable",
            "Uncertain regimen",
            "Statin + ezetimibe",
            "Statin without ezetimibe",
            "Ezetimibe without statin",
        ],
        default="Neither statin nor ezetimibe",
    )
    s = d[d.statin].copy()
    dose_ok = (
        s.statin_dose_analysis_eligible.eq(True)
        & s.statin_daily_dose_mg.notna()
        & ~s.statin_regimen_ambiguity
    )
    s["high"] = dose_ok & (
        (
            s.statin_ingredient.eq("atorvastatin")
            & s.statin_daily_dose_mg.between(40, 80)
        )
        | (
            s.statin_ingredient.eq("rosuvastatin")
            & s.statin_daily_dose_mg.between(20, 40)
        )
    )
    s["unknown"] = ~dose_ok
    s["intensity"] = np.select(
        [s.unknown, s.high], ["Unknown", "High"], default="Non-high"
    )
    ambiguous = e[ID].isin(s.loc[s.unknown, ID])
    high = e[ID].isin(s.loc[s.high, ID])
    e["intensity"] = np.select(
        [~e.regimen_available, ~e.statin, ambiguous, high],
        ["Unavailable", "No statin", "Unknown", "High"],
        default="Non-high",
    )
    e["statin_combination"] = e.regimen.eq("Statin + ezetimibe")
    e["high_statin"] = e.intensity.eq("High")
    e["unknown_intensity"] = e.intensity.eq("Unknown")
    e["high_statin_ezetimibe"] = e.high_statin & e.statin_combination
    e["statin_conflict"] = e[ID].isin(s.loc[s.statin_regimen_ambiguity, ID])
    q.groupby(
        ["active", "medication_context", "medication_use_status"], dropna=False
    ).size().rename("records").reset_index().to_csv(
        RESULTS / "medication_checks.csv", index=False
    )
    return e


def build():
    RESULTS.mkdir(exist_ok=True)
    t = make_tables()
    e = t["encounters"].copy()
    flow = [
        {
            "step": "T2D input admissions",
            "admissions": len(e),
            "patients": e[PID].nunique(),
        }
    ]
    pregnant = set(
        t["clinical_history"].loc[
            t["clinical_history"].concept.eq("pregnancy")
            & t["clinical_history"].assertion.eq("present"),
            ID,
        ]
    )
    for label, mask in [
        ("Discharged", e.hospitalization_status.eq("discharged")),
        ("Adults", e.age_at_admission_min_years.ge(18)),
        ("No documented pregnancy", ~e[ID].isin(pregnant)),
    ]:
        e = e[mask.reindex(e.index)].copy()
        flow.append({"step": label, "admissions": len(e), "patients": e[PID].nunique()})
    for c in ["admission_date", "discharge_date"]:
        e[c] = pd.to_datetime(e[c])
    assert e.admission_date.le(e.discharge_date).all()
    e["year"] = e.admission_date.dt.year
    e["period"] = np.where(e.year.ge(2023), "2023–September 2026", "2017–2022")
    e["age"] = e.age_at_admission_calendar_years
    e["regimen_available"] = e.discharge_medication_context.eq(
        "discharge_recommendation"
    )
    e = disease_status(e, t["clinical_history"], t["diagnoses"])
    e, ldl = select_labs(e, t["labs"])
    e = bmi(e, t["measurements"])
    e = medications(e, t["medications"])
    e["ldl_band"] = pd.cut(
        e.ldl, [-np.inf, 55, 70, 100, np.inf], right=False, labels=BANDS
    )
    pd.DataFrame(flow).to_csv(RESULTS / "cohort_flow.csv", index=False)
    return e, ldl
