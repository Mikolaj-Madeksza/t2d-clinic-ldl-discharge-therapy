"""Validate the artificial demonstration without reading clinical data."""

import json
import numpy as np
import pandas as pd
from derive import build, code_flags, text_flags, ID, PID, BANDS, REGIMENS, RESULTS
from derive import disease_status
from mock_data import make_tables
from run_analysis import proportions, PLOT_REGIMENS


def main():
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append(name)

    inputs = make_tables()
    encounters = inputs["encounters"]
    check(
        "Artificial identifiers",
        all(
            frame[column].str.startswith("SYNTHETIC-").all()
            for frame in inputs.values()
            for column in [ID, PID]
        ),
    )
    check("Unique admissions", encounters[ID].is_unique)
    for name, frame in inputs.items():
        linked = frame.merge(
            encounters[[ID, PID]], on=ID, suffixes=("", "_enc"), validate="many_to_one"
        )
        check("Linked patient IDs: " + name, linked[PID].eq(linked[PID + "_enc"]).all())
    e, ldl = build()
    recent = e[e.year.ge(2023)]
    check(
        "Eligibility exclusions",
        len(e) == 157
        and not e[ID]
        .isin(
            [
                "SYNTHETIC-ADMISSION-001",
                "SYNTHETIC-ADMISSION-002",
                "SYNTHETIC-ADMISSION-003",
            ]
        )
        .any(),
    )
    check(
        "Missing tests and lists retained",
        e.ldl.isna().any() and (~e.regimen_available).any(),
    )
    ids = ["SYNTHETIC-ADMISSION-000", "SYNTHETIC-ADMISSION-006"]
    paired = e[e[ID].isin(ids)].copy()
    diagnoses = inputs["diagnoses"][inputs["diagnoses"][ID].isin(ids)].copy()
    history = inputs["clinical_history"][
        inputs["clinical_history"][ID].isin(ids)
    ].copy()
    for frame in [paired, diagnoses, history]:
        frame[PID] = "SYNTHETIC-SHARED-PATIENT"
    paired = disease_status(paired, history, diagnoses).set_index(ID)
    check(
        "ASCVD does not carry between admissions",
        paired.loc[ids[0], "ascvd"] and not paired.loc[ids[1], "ascvd"],
    )
    check(
        "Negative history is excluded",
        not e.loc[e[ID].eq("SYNTHETIC-ADMISSION-007"), "ascvd"].item(),
    )
    check(
        "ASCVD code boundaries",
        code_flags("I63.9")["ischemic_stroke"]
        and not code_flags("I64")["ischemic_stroke"]
        and not code_flags("G45.4")["ischemic_stroke"]
        and code_flags("I70.2")["pad"]
        and not code_flags("I70.9")["pad"],
    )
    check(
        "Word boundaries",
        not text_flags("gastropatia")["ischemic_stroke"]
        and not text_flags("udar cieplny")["ischemic_stroke"],
    )
    first = (
        ldl.sort_values(["time", "executed_at", "lab_result_id"])
        .drop_duplicates(ID)
        .set_index(ID)
    )
    selected = e[e.ldl.notna()].set_index(ID)
    check(
        "First eligible LDL selected",
        selected.ldl_record.eq(first.lab_result_id.reindex(selected.index)).all(),
    )
    check(
        "No post-discharge LDL",
        ldl.executed_at.dt.normalize().le(ldl.discharge_date).all(),
    )
    check(
        "LDL category boundaries",
        pd.cut(
            pd.Series([54.0, 55.0, 69.0, 70.0, 99.0, 100.0]),
            [-np.inf, 55, 70, 100, np.inf],
            right=False,
            labels=BANDS,
        )
        .astype(str)
        .tolist()
        == [BANDS[0], BANDS[1], BANDS[1], BANDS[2], BANDS[2], BANDS[3]],
    )
    check(
        "Mean LDL sensitivity",
        np.allclose(
            selected.ldl_mean, ldl.groupby(ID).value.mean().reindex(selected.index)
        ),
    )
    safe = (
        ldl[~ldl.high_tg]
        .sort_values(["time", "executed_at", "lab_result_id"])
        .drop_duplicates(ID)
        .set_index(ID)
    )
    filtered = e[e.ldl_without_high_tg.notna()].set_index(ID)
    check(
        "High-TG exclusion precedes reselection",
        filtered.ldl_without_high_tg_record.eq(
            safe.lab_result_id.reindex(filtered.index)
        ).all(),
    )
    check(
        "High-TG case selects second LDL",
        e.loc[e[ID].eq("SYNTHETIC-ADMISSION-013"), "ldl_without_high_tg"].item()
        == 90.0,
    )
    check(
        "Unmatched TG retained",
        set(ldl.loc[ldl.paired_tg.isna(), ID]).issubset(set(safe.index)),
    )
    uncertain = e[e.regimen.eq("Uncertain regimen")]
    check(
        "Uncertain statin retained",
        len(uncertain) > 0
        and uncertain.statin.all()
        and uncertain.unknown_intensity.all(),
    )
    check(
        "Statin intensity totals",
        int(
            e.high_statin.sum()
            + e.intensity.eq("Non-high").sum()
            + e.unknown_intensity.sum()
        )
        == int(e.statin.sum()),
    )
    check(
        "Stopped high dose does not determine intensity",
        e.loc[e[ID].eq("SYNTHETIC-ADMISSION-006"), "intensity"].item() == "Non-high",
    )
    check(
        "Unavailable lists separate",
        e.loc[~e.regimen_available, "regimen"].eq("Unavailable").all(),
    )
    check(
        "Regimens exhaustive",
        e.loc[e.regimen_available, "regimen"].isin(REGIMENS).all(),
    )
    check("Non-HDL pairing arithmetic", np.allclose(e.non_hdl, e.tc_mean - e.hdl_mean))

    cluster = pd.DataFrame({PID: ["SYNTHETIC-A"] * 10 + ["SYNTHETIC-B"]})
    event = pd.Series([True] * 10 + [False])
    s = proportions(cluster, {"event": event})["event"]
    doubled = proportions(
        pd.concat([cluster, cluster], ignore_index=True),
        {"event": pd.concat([event, event], ignore_index=True)},
    )["event"]
    check(
        "Patient bootstrap preserves clusters",
        s["lower"] == 0
        and s["upper"] == 100
        and all(s[k] == doubled[k] for k in ["percent", "lower", "upper"]),
    )
    check(
        "Boundary intervals omitted",
        proportions(cluster, {"all": np.ones(len(cluster))})["all"]["lower"] is None,
    )

    estimates = pd.read_csv(RESULTS / "estimates.csv")
    check(
        "Proportion arithmetic",
        np.allclose(estimates.percent, 100 * estimates.n / estimates.N, equal_nan=True),
    )
    intervals = estimates[estimates.lower.notna()]
    check(
        "Interval bounds",
        intervals.lower.between(0, 100).all()
        and intervals.upper.between(0, 100).all()
        and intervals.lower.le(intervals.upper).all(),
    )
    fig = pd.read_csv(RESULTS / "figure_1_data.csv")
    check("Four plotted regimens", set(fig.regimen) == set(PLOT_REGIMENS))
    check(
        "Figure denominators",
        int(fig.groupby(["ascvd", "ldl_band"]).N.first().sum())
        == int(
            (
                recent.ldl.notna()
                & recent.regimen_available
                & recent.regimen.isin(PLOT_REGIMENS)
            ).sum()
        ),
    )
    check(
        "Stacked percentages",
        np.allclose(fig.groupby(["ascvd", "ldl_band"]).percent.sum(), 100),
    )
    tables = json.loads((RESULTS / "tables.json").read_text())
    check(
        "All planned tables",
        {
            "table_1",
            "table_2",
            "table_3",
            *[f"table_s{i}" for i in range(1, 10)],
        }.issubset({t["key"] for t in tables}),
    )
    check(
        "Synthetic captions",
        all(t["title"].startswith("SYNTHETIC EXAMPLE") for t in tables),
    )
    for t in tables:
        actual = pd.read_csv(
            RESULTS / (t["key"] + ".csv"), dtype=str, keep_default_na=False
        )
        check(
            "Saved table: " + t["key"],
            actual.columns.tolist() == t["columns"]
            and actual.values.tolist() == t["rows"],
        )
        if t["key"] in ["table_3", "table_s3"]:
            check(
                "Simplified treatment rows: " + t["key"],
                not {"Uncertain regimen", "PCSK9 inhibitor", "Bempedoic acid"}
                & {r[0] for r in t["rows"]},
            )
    print(f"{len(checks)} checks passed on synthetic examples.")


if __name__ == "__main__":
    main()
