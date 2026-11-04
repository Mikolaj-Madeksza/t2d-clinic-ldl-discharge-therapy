"""Study tables, sensitivity analyses and figure using synthetic inputs only."""

import hashlib
import json
import numpy as np
import pandas as pd
from derive import build, ID, PID, BANDS, REGIMENS, RESULTS

UNCERTAIN_NOTE = "Confirmed statin use with uncertain ezetimibe use remains in any-statin and unknown-intensity categories but is unclassified for regimen."
OTHER_TREATMENT_NOTE = (
    "PCSK9 inhibitor and bempedoic acid counts are available in estimates.csv."
)
FIGURE_NOTE = "Percentages use admissions with LDL-C and a classifiable discharge regimen. Unclassifiable regimens are excluded."

BOOTSTRAPS = 5000
SEED = 20261103
PLOT_REGIMENS = [r for r in REGIMENS if r != "Uncertain regimen"]
estimates = []
tables = []


def proportions(d, outcomes):
    """Percentile intervals from resampling patients with all their admissions."""
    if d.empty:
        return {
            k: {
                "n": 0,
                "N": 0,
                "patients": 0,
                "percent": None,
                "lower": None,
                "upper": None,
                "event_patients": 0,
                "nonevent_patients": 0,
            }
            for k in outcomes
        }
    x = pd.DataFrame(
        {k: np.asarray(v, dtype=int) for k, v in outcomes.items()}, index=d.index
    )
    x[PID] = d[PID]
    x["N"] = 1
    grouped = x.groupby(PID, sort=True).sum()
    keys = list(outcomes)
    values = grouped[keys].to_numpy()
    sizes = grouped.N.to_numpy()
    seed = SEED + int(
        hashlib.sha256("|".join(grouped.index).encode()).hexdigest()[:8], 16
    )
    rng = np.random.default_rng(seed)
    samples = []
    g = len(grouped)
    for start in range(0, BOOTSTRAPS, 250):
        weights = rng.multinomial(
            g, np.full(g, 1 / g), size=min(250, BOOTSTRAPS - start)
        )
        samples.append((weights @ values) / (weights @ sizes)[:, None] * 100)
    draws = np.concatenate(samples)
    results = {}
    for j, key in enumerate(keys):
        n = int(x[key].sum())
        event_patients = int(grouped[key].gt(0).sum())
        nonevent_patients = int(grouped[key].lt(grouped.N).sum())
        lo, hi = np.quantile(draws[:, j], [0.025, 0.975])
        if n == 0 or n == len(d) or g < 2:
            lo = hi = None
        results[key] = {
            "n": n,
            "N": len(d),
            "patients": g,
            "percent": 100 * n / len(d),
            "lower": None if lo is None else float(lo),
            "upper": None if hi is None else float(hi),
            "event_patients": event_patients,
            "nonevent_patients": nonevent_patients,
        }
    return results


def fmt_stat(s, ci=True):
    if not s["N"]:
        return "—"
    value = f"{s['n']}/{s['N']} ({s['percent']:.1f}%)"
    if ci and s["lower"] is not None:
        value += f"\n[{s['lower']:.1f}–{s['upper']:.1f}]"
        if min(s["event_patients"], s["nonevent_patients"]) < 5:
            value += "*"
    return value


def continuous(s):
    v = s.dropna()
    if v.empty:
        return "—"
    return (
        f"{v.median():.1f} [{v.quantile(0.25):.1f}–{v.quantile(0.75):.1f}]; n={len(v)}"
    )


def count(mask, n):
    return f"{int(mask.sum())} ({100 * mask.sum() / n:.1f}%)" if n else "—"


def save_table(key, title, columns, rows, note, abbreviations="", document=None):
    title = "SYNTHETIC EXAMPLE — " + title
    table = {
        "key": key,
        "title": title,
        "columns": columns,
        "rows": rows,
        "note": note,
        "abbreviations": abbreviations,
        "document": document,
    }
    tables.append(table)
    pd.DataFrame(rows, columns=columns).to_csv(RESULTS / (key + ".csv"), index=False)
    return table


def cohort_table(d, key, title, document):
    groups = {
        "Overall": d,
        "LDL available": d[d.ldl.notna()],
        "LDL unavailable": d[d.ldl.isna()],
    }
    metrics = {
        "Admissions": lambda g: str(len(g)),
        "Patients": lambda g: str(g[PID].nunique()),
        "LDL available": lambda g: count(g.ldl.notna(), len(g)),
        "Discharge list available": lambda g: count(g.regimen_available, len(g)),
        "Age, years": lambda g: continuous(g.age),
        "Female sex": lambda g: count(g.sex.eq("female"), len(g)),
        "Documented ASCVD": lambda g: count(g.ascvd, len(g)),
        "Diabetology": lambda g: count(g.clinic.eq("Diabetology"), len(g)),
        "Endocrinology": lambda g: count(g.clinic.eq("Endocrinology"), len(g)),
        "HbA1c, %": lambda g: continuous(g.hba1c),
        "HbA1c unavailable": lambda g: count(g.hba1c.isna(), len(g)),
        "BMI, kg/m²": lambda g: continuous(g.bmi),
        "BMI unavailable": lambda g: count(g.bmi.isna(), len(g)),
        "eGFR, mL/min/1.73 m²": lambda g: continuous(g.egfr),
        "eGFR unavailable": lambda g: count(g.egfr.isna(), len(g)),
    }
    for year in sorted(d.year.unique()):
        metrics[str(year)] = lambda g, y=year: count(g.year.eq(y), len(g))
    rows = [[label] + [fn(g) for g in groups.values()] for label, fn in metrics.items()]
    return save_table(
        key,
        title,
        ["Characteristic"] + list(groups),
        rows,
        "Values are n (%) or median [interquartile range], with the number measured. Percentages use all admissions in each column. Patients may contribute to both testing groups. Age is estimated from birth year."
        + (" Data for 2026 are incomplete." if d.year.eq(2026).any() else ""),
        "ASCVD, atherosclerotic cardiovascular disease; BMI, body mass index; eGFR, estimated glomerular filtration rate; HbA1c, glycated hemoglobin; LDL, low-density lipoprotein.",
        document,
    )


def ldl_table(d, key, title, document=None):
    groups = {
        "Overall": d,
        "Documented ASCVD": d[d.ascvd],
        "No documented ASCVD": d[~d.ascvd],
    }
    rows = {
        label: []
        for label in ["Admissions with LDL", "Patients with LDL", "LDL-C, mg/dL"]
        + BANDS
        + ["LDL-C <70 mg/dL"]
    }
    for group, g in groups.items():
        a = g[g.ldl.notna()]
        band = pd.cut(a.ldl, [-np.inf, 55, 70, 100, np.inf], right=False, labels=BANDS)
        outcomes = {b: band.eq(b) for b in BANDS}
        outcomes["LDL-C <70 mg/dL"] = a.ldl.lt(70)
        stats = proportions(a, outcomes)
        rows["Admissions with LDL"].append(str(len(a)))
        rows["Patients with LDL"].append(str(a[PID].nunique()))
        rows["LDL-C, mg/dL"].append(continuous(a.ldl))
        for metric, s in stats.items():
            rows[metric].append(fmt_stat(s))
            estimates.append({"table": key, "group": group, "metric": metric, **s})
    return save_table(
        key,
        title,
        ["LDL-C measure"] + list(groups),
        [[k] + v for k, v in rows.items()],
        "LDL-C is the first eligible inpatient result unless stated otherwise. Continuous values are median [interquartile range]. Proportions are n/N (%) with patient-bootstrap 95% confidence intervals below. No documented ASCVD does not mean confirmed absence. *Fewer than five patients with or without the outcome; interpret the interval cautiously. Intervals are not estimated for all-zero or all-one outcomes.",
        "ASCVD, atherosclerotic cardiovascular disease; LDL-C, low-density lipoprotein cholesterol.",
        document,
    )


def treatment_table(d, key, title, document=None):
    groups = {
        "Overall": d,
        "Documented ASCVD": d[d.ascvd],
        "No documented ASCVD": d[~d.ascvd],
        "ASCVD and LDL-C ≥55": d[d.ascvd & d.ldl.ge(55)],
    }
    row_names = (
        [
            "Eligible admissions",
            "Available discharge lists",
            "Patients with a discharge list",
        ]
        + REGIMENS
        + [
            "Any statin",
            "High-intensity statin",
            "Non-high-intensity statin",
            "Unknown statin intensity",
            "High-intensity statin + ezetimibe",
            "Fibrate",
            "PCSK9 inhibitor",
            "Bempedoic acid",
        ]
    )
    rows = {k: [] for k in row_names}
    for group, g in groups.items():
        a = g[g.regimen_available]
        outcomes = {r: a.regimen.eq(r) for r in REGIMENS}
        outcomes.update(
            {
                "Any statin": a.statin,
                "High-intensity statin": a.high_statin,
                "Non-high-intensity statin": a.intensity.eq("Non-high"),
                "Unknown statin intensity": a.unknown_intensity,
                "High-intensity statin + ezetimibe": a.high_statin_ezetimibe,
                "Fibrate": a.fibrate,
                "PCSK9 inhibitor": a.pcsk9,
                "Bempedoic acid": a.bempedoic,
            }
        )
        stats = proportions(a, outcomes)
        rows["Eligible admissions"].append(str(len(g)))
        rows["Available discharge lists"].append(str(len(a)))
        rows["Patients with a discharge list"].append(str(a[PID].nunique()))
        for metric, s in stats.items():
            rows[metric].append(fmt_stat(s))
            estimates.append({"table": key, "group": group, "metric": metric, **s})
    if key in ["table_3", "table_s3"]:
        for label in ["Uncertain regimen", "PCSK9 inhibitor", "Bempedoic acid"]:
            del rows[label]
    note = (
        "Values are n/N (%) with patient-bootstrap 95% confidence intervals below. All treatment percentages use admissions with an available discharge list, including unknown intensity. The ASCVD/LDL-C ≥55 subgroup overlaps the ASCVD column. The first "
        + ("four" if key in ["table_3", "table_s3"] else "five")
        + " treatment categories are mutually exclusive; the remaining rows overlap. High-intensity statin means atorvastatin 40–80 mg/day or rosuvastatin 20–40 mg/day. *Fewer than five patients with or without the outcome. Intervals are not estimated for all-zero or all-one outcomes."
    )
    if key == "table_3":
        note += " " + UNCERTAIN_NOTE
    if key == "table_s3":
        note += " " + OTHER_TREATMENT_NOTE
    abbreviations = "ASCVD, atherosclerotic cardiovascular disease; LDL-C, low-density lipoprotein cholesterol"
    if key != "table_3":
        abbreviations += "; PCSK9, proprotein convertase subtilisin/kexin type 9"
    return save_table(
        key,
        title,
        ["Discharge recommendation"] + list(groups),
        [[k] + v for k, v in rows.items()],
        note,
        abbreviations + ".",
        document,
    )


def annual_tables(e):
    coverage = []
    clinical = []
    for (year, clinic), g in e.groupby(["year", "clinic"]):
        a = g[g.ldl.notna()]
        v = g[g.regimen_available]
        coverage.append(
            [
                str(year),
                clinic,
                str(len(g)),
                str(g[PID].nunique()),
                f"{len(a)}/{len(g)} ({100 * len(a) / len(g):.1f}%)",
                str(len(v)),
            ]
        )
    for year, g in e.groupby("year"):
        a = g[g.ldl.notna()]
        v = g[g.regimen_available]
        risk = a[a.ascvd]
        clinical.append(
            [
                str(year),
                continuous(a.ldl),
                f"{int(a.ldl.lt(70).sum())}/{len(a)} ({100 * a.ldl.lt(70).mean():.1f}%)",
                "—"
                if risk.empty
                else f"{int(risk.ldl.lt(55).sum())}/{len(risk)} ({100 * risk.ldl.lt(55).mean():.1f}%)",
                f"{int(v.high_statin.sum())}/{len(v)} ({100 * v.high_statin.mean():.1f}%)",
                f"{int(v.statin_combination.sum())}/{len(v)} ({100 * v.statin_combination.mean():.1f}%)",
            ]
        )
    save_table(
        "table_s4",
        "Table S4. Annual admission counts and data availability",
        [
            "Year",
            "Clinic",
            "Admissions",
            "Patients",
            "LDL available",
            "Discharge lists",
        ],
        coverage,
        "Patients may contribute to more than one clinic or year. Data for 2026 are incomplete. Empty clinic-year combinations are not shown.",
        "LDL, low-density lipoprotein.",
        "supplement",
    )
    save_table(
        "table_s5",
        "Table S5. Annual LDL-C and discharge recommendations",
        [
            "Year",
            "LDL-C, mg/dL",
            "LDL-C <70",
            "ASCVD and LDL-C <55",
            "High-intensity statin",
            "Statin + ezetimibe",
        ],
        clinical,
        "LDL-C is median [interquartile range], with n measured. Proportions are n/N (%). LDL-C <70 uses tested admissions; LDL-C <55 uses tested admissions with ASCVD. Treatment denominators are available discharge lists. Data for 2026 are incomplete.",
        "ASCVD, atherosclerotic cardiovascular disease; LDL-C, low-density lipoprotein cholesterol.",
        "supplement",
    )


def other_lipids(e):
    for number, (period, d) in enumerate(e.groupby("period", sort=True), 6):
        groups = {
            "Overall": d,
            "Documented ASCVD": d[d.ascvd],
            "No documented ASCVD": d[~d.ascvd],
        }
        rows = [["Admissions"] + [str(len(g)) for g in groups.values()]]
        for col, label in [
            ("tc_mean", "Total cholesterol"),
            ("hdl_mean", "HDL cholesterol"),
            ("tg_mean", "Triglycerides"),
            ("non_hdl", "Non-HDL cholesterol"),
        ]:
            rows.append(
                [label + ", mg/dL"] + [continuous(g[col]) for g in groups.values()]
            )
            rows.append(
                [label + " unavailable"]
                + [count(g[col].isna(), len(g)) for g in groups.values()]
            )
        save_table(
            f"table_s{number}",
            f"Table S{number}. Other lipid measurements, {period}",
            ["Measure"] + list(groups),
            rows,
            "Concentrations are admission means, summarized as median [interquartile range], with n measured. Non-HDL cholesterol uses total cholesterol and HDL measured at the same recorded time. Missing values are n (%).",
            "ASCVD, atherosclerotic cardiovascular disease; HDL, high-density lipoprotein.",
            "supplement",
        )


def sensitivity_tables(recent):
    first = (
        recent[recent.ldl.notna()]
        .sort_values(["admission_date", "discharge_date", ID])
        .drop_duplicates(PID)
    )
    mean = recent.copy()
    mean["ldl"] = mean.ldl_mean
    no_tg = recent.copy()
    no_tg["ldl"] = no_tg.ldl_without_high_tg
    scenarios = [
        ("Primary", recent),
        ("First tested admission per patient", first),
        ("Admission mean LDL-C", mean),
        ("Excluding 2026", recent[recent.year.lt(2026)]),
        ("Diabetology only", recent[recent.clinic.eq("Diabetology")]),
        ("Excluding paired triglycerides ≥400", no_tg),
    ]
    ldl_rows = []
    treatment_rows = []
    overview = []
    for i, (name, d) in enumerate(scenarios):
        a = d[d.ldl.notna()]
        risk = a[a.ascvd]
        sub = d[d.ascvd & d.ldl.ge(55) & d.regimen_available]
        low70 = proportions(a, {"low": a.ldl.lt(70)})["low"]
        low55 = proportions(risk, {"low": risk.ldl.lt(55)})["low"]
        therapy = proportions(
            sub,
            {
                "high": sub.high_statin,
                "combination": sub.statin_combination,
                "unknown": sub.unknown_intensity,
            },
        )
        ldl_rows.append(
            [
                name,
                str(len(a)),
                str(a[PID].nunique()),
                continuous(a.ldl),
                fmt_stat(low70),
                fmt_stat(low55),
            ]
        )
        treatment_rows.append(
            [
                name,
                str(len(sub)),
                str(sub[PID].nunique()),
                fmt_stat(therapy["high"]),
                fmt_stat(therapy["combination"]),
                fmt_stat(therapy["unknown"]),
            ]
        )
        overview.append(
            {
                "analysis": name,
                "eligible_admissions": len(d),
                "eligible_patients": d[PID].nunique(),
                "ldl_admissions": len(a),
                "ldl_patients": a[PID].nunique(),
                "ldl_median": a.ldl.median(),
                "ldl_lt70": low70,
                "ascvd_ldl_lt55": low55,
                "ascvd_ldl_ge55_treatment": therapy,
            }
        )
        if i:
            ldl_table(d, f"sensitivity_{i}_ldl", name)
            treatment_table(d, f"sensitivity_{i}_treatment", name)
    save_table(
        "table_s8",
        "Table S8. Sensitivity analyses of LDL-C",
        [
            "Analysis",
            "Admissions with LDL",
            "Patients with LDL",
            "LDL-C, mg/dL",
            "LDL-C <70",
            "ASCVD and LDL-C <55",
        ],
        ldl_rows,
        "Values are median [interquartile range] or n/N (%) with patient-bootstrap 95% confidence intervals. The first-admission analysis selects the first LDL-tested admission within 2023–September 2026. High-triglyceride exclusion removes paired LDL measurements before selecting the first remaining result. "
        + str(int((recent.ldl.notna() & recent.ldl_paired_tg.isna()).sum()))
        + " primary LDL results have no matching triglyceride result and are retained.",
        "ASCVD, atherosclerotic cardiovascular disease; LDL-C, low-density lipoprotein cholesterol.",
        "supplement",
    )
    save_table(
        "table_s9",
        "Table S9. Sensitivity analyses of treatment in documented ASCVD with LDL-C ≥55 mg/dL",
        [
            "Analysis",
            "Admissions",
            "Patients",
            "High-intensity statin",
            "Statin + ezetimibe",
            "Unknown intensity",
        ],
        treatment_rows,
        "All denominators are admissions with documented ASCVD, LDL-C ≥55 mg/dL and an available discharge list. Values are n/N (%) with patient-bootstrap 95% confidence intervals. *Fewer than five patients with or without the outcome. The full subgroup results for each sensitivity analysis are provided in the accompanying CSV tables.",
        "ASCVD, atherosclerotic cardiovascular disease; LDL-C, low-density lipoprotein cholesterol.",
        "supplement",
    )
    (RESULTS / "sensitivity_summary.json").write_text(
        json.dumps(overview, indent=2, ensure_ascii=False) + "\n"
    )
    return overview


def figure(e):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.family": "DejaVu Serif", "font.size": 9, "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.268, 4.2), sharey=True)
    # Rose is slightly lighter than #AD6077 to separate it from blue in grayscale.
    colors = ["#4A7185", "#D3B178", "#BE8194", "#D9DEE2"]
    label_colors = ["white", "black", "black", "black"]
    rows = []
    for ax, (label, g) in zip(
        axes,
        e[e.ldl.notna() & e.regimen_available & e.regimen.isin(PLOT_REGIMENS)].groupby(
            "ascvd_group", sort=True
        ),
    ):
        bands = pd.cut(g.ldl, [-np.inf, 55, 70, 100, np.inf], right=False, labels=BANDS)
        bottom = np.zeros(4)
        for regimen, color, label_color in zip(PLOT_REGIMENS, colors, label_colors):
            counts = np.array(
                [int((bands.eq(b) & g.regimen.eq(regimen)).sum()) for b in BANDS]
            )
            totals = np.array([int(bands.eq(b).sum()) for b in BANDS])
            values = (
                np.divide(
                    counts, totals, out=np.zeros(4, dtype=float), where=totals > 0
                )
                * 100
            )
            ax.bar(
                np.arange(4),
                values,
                bottom=bottom,
                width=0.68,
                color=color,
                edgecolor="#6E7377",
                linewidth=0.35,
                label=regimen,
            )
            for j, (n, N, p) in enumerate(zip(counts, totals, values)):
                rows.append(
                    {
                        "ascvd": label,
                        "ldl_band": BANDS[j],
                        "regimen": regimen,
                        "n": int(n),
                        "N": int(N),
                        "percent": float(p),
                    }
                )
                if p >= 8:
                    ax.text(
                        j,
                        bottom[j] + p / 2,
                        f"{p:.0f}%",
                        ha="center",
                        va="center",
                        color=label_color,
                        fontsize=8,
                    )
            bottom += values
        for j, N in enumerate(totals):
            ax.text(j, 103, f"n={N}", ha="center", fontsize=8)
        ax.set_title(label, pad=14, fontsize=10)
        ax.set_xticks(range(4), BANDS)
        ax.set_xlabel("LDL-C, mg/dL")
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.spines[["top", "right"]].set_visible(False)
        assert np.allclose(bottom[totals > 0], 100)
    axes[0].set_ylabel("Admissions, %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("SYNTHETIC EXAMPLE", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.31, wspace=0.18)
    for ext in ["png", "pdf"]:
        fig.savefig(RESULTS / f"figure_1.{ext}", dpi=300)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / "figure_1_data.csv", index=False)
    (RESULTS / "figure_1_caption.txt").write_text(
        "SYNTHETIC EXAMPLE. Figure 1. Discharge treatment categories across LDL-C groups by documented ASCVD, 2023–September 2026.\n\nNotes: "
        + FIGURE_NOTE
        + "\nAbbreviations: ASCVD, atherosclerotic cardiovascular disease; LDL-C, low-density lipoprotein cholesterol.\n"
    )


def main():
    e, ldl = build()
    recent = e[e.year.ge(2023)].copy()
    historical = e[e.year.lt(2023)].copy()
    cohort_table(
        recent,
        "table_1",
        "Table 1. Cohort characteristics and LDL testing, 2023–September 2026",
        "main",
    )
    ldl_table(
        recent,
        "table_2",
        "Table 2. LDL-C by documented ASCVD, 2023–September 2026",
        "main",
    )
    treatment_table(
        recent,
        "table_3",
        "Table 3. Discharge lipid-lowering recommendations, 2023–September 2026",
        "main",
    )
    cohort_table(
        historical,
        "table_s1",
        "Table S1. Cohort characteristics and LDL testing, 2017–2022",
        "supplement",
    )
    ldl_table(
        historical,
        "table_s2",
        "Table S2. LDL-C by documented ASCVD, 2017–2022",
        "supplement",
    )
    treatment_table(
        historical,
        "table_s3",
        "Table S3. Discharge lipid-lowering recommendations, 2017–2022",
        "supplement",
    )
    annual_tables(e)
    other_lipids(e)
    sensitivity_tables(recent)
    figure(recent)
    pd.DataFrame(estimates).to_csv(RESULTS / "estimates.csv", index=False)
    (RESULTS / "tables.json").write_text(
        json.dumps(tables, ensure_ascii=False, indent=2) + "\n"
    )
    print("Synthetic demonstration complete. Aggregate outputs are in outputs/.")


if __name__ == "__main__":
    main()
