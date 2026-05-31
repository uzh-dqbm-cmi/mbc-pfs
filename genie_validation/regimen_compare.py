from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from lifelines import KaplanMeierFitter  # noqa: E402
from lifelines.statistics import logrank_test  # noqa: E402
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu  # noqa: E402

from config import EVENT_COL, TIME_COL
from . import labels, mapping as M, paths

OUTDIR = Path("validation_outputs/regimen_overlap")
EXCLUDED_SUBTYPES = {"BONE TREATMENT"}  # GENIE does not record supportive/bone agents
PLANNED_WINDOW_DAYS = 28  # same window MSK uses for PLANNED_* (treatment.py)
SUFFICIENT_N = 20
INVESTIGATIVE_AGENTS = frozenset({"INVESTIGATIVE"})


def _regimen_str(reg: FrozenSet[str]) -> str:
    return " + ".join(sorted(reg))


def build_msk_line_regimens() -> pd.DataFrame:
    """One row per MSK modelled line with its exact (pre-collapse) agent set."""
    dm = pd.read_csv(paths.MSK_DESIGN_MATRIX)
    dm = dm[dm[EVENT_COL] != -1][
        ["PATIENT_ID", "LINE", "LINE_START", EVENT_COL, TIME_COL]
    ].copy()
    dm["PATIENT_ID"] = dm["PATIENT_ID"].astype(str)

    tx = pd.read_csv(paths.MSK_TREATMENT_TIMELINE, sep="\t", dtype=str)
    tx["PATIENT_ID"] = tx["PATIENT_ID"].astype(str)
    tx["SUBTYPE"] = tx["SUBTYPE"].astype(str).str.strip().str.upper()
    tx["AGENT"] = tx["AGENT"].astype(str).str.strip().str.upper()
    tx = tx[~tx["SUBTYPE"].isin(EXCLUDED_SUBTYPES)]
    tx = tx[tx["AGENT"].notna() & (tx["AGENT"] != "NAN")]
    tx["START_DATE"] = pd.to_numeric(tx["START_DATE"], errors="coerce")
    tx = tx.dropna(subset=["START_DATE"])
    tx["STOP_DATE"] = pd.to_numeric(tx["STOP_DATE"], errors="coerce")
    tx["STOP_DATE"] = tx["STOP_DATE"].fillna(tx["START_DATE"] + 1)
    tx["START_DATE"] = tx["START_DATE"].astype(int)
    tx["STOP_DATE"] = tx["STOP_DATE"].astype(int)

    by_pid: Dict[str, np.ndarray] = {
        pid: g[["START_DATE", "STOP_DATE", "AGENT"]].to_numpy()
        for pid, g in tx.groupby("PATIENT_ID", sort=False)
    }

    regimens: List[FrozenSet[str]] = []
    for row in dm.itertuples(index=False):
        line_start = int(row.LINE_START)
        planned_end = line_start + PLANNED_WINDOW_DAYS
        agents = set()
        arr = by_pid.get(row.PATIENT_ID)
        if arr is not None:
            for start, stop, agent in arr:
                if (start <= planned_end) and (stop >= line_start):
                    agents.add(agent)
        regimens.append(M.canonical_regimen(agents))

    dm["REGIMEN_SET"] = regimens
    dm["regimen"] = dm["REGIMEN_SET"].map(_regimen_str)
    dm["n_agents"] = dm["REGIMEN_SET"].map(len)
    return dm.rename(columns={"LINE_START": "start_date"})[
        ["PATIENT_ID", "LINE", "start_date", EVENT_COL, TIME_COL,
         "n_agents", "regimen", "REGIMEN_SET"]
    ]


def build_genie_line_regimens() -> pd.DataFrame:
    """One row per GENIE candidate line (no regimen filter) with its agent set."""
    g = labels.build_labels(apply_drug_filter=False)
    g["REGIMEN_SET"] = g["REGIMEN_DRUGS"].map(
        lambda s: M.canonical_regimen(str(s).split(","))
    )
    g["regimen"] = g["REGIMEN_SET"].map(_regimen_str)
    g["n_agents"] = g["REGIMEN_SET"].map(len)
    return g.rename(columns={"LINE_START": "start_date"})[
        ["PATIENT_ID", "LINE", "start_date", EVENT_COL, TIME_COL,
         "n_agents", "regimen", "REGIMEN_SET", "REGIMEN_DRUGS"]
    ]


def _event_rate_test(m: pd.DataFrame, g: pd.DataFrame):
    em, eg = int(m[EVENT_COL].sum()), int(g[EVENT_COL].sum())
    table = [[em, len(m) - em], [eg, len(g) - eg]]
    try:
        _, p, _, _ = chi2_contingency(table)
        # Fisher is more reliable for small cells
        if min(min(r) for r in table) < 5:
            _, p = fisher_exact(table)
    except Exception:
        p = float("nan")
    return em, eg, em / len(m), eg / len(g), float(p)


def _time_tests(m: pd.DataFrame, g: pd.DataFrame):
    try:
        mw = float(mannwhitneyu(m[TIME_COL], g[TIME_COL], alternative="two-sided")[1])
    except Exception:
        mw = float("nan")
    try:
        lr = float(
            logrank_test(
                m[TIME_COL], g[TIME_COL],
                event_observed_A=m[EVENT_COL], event_observed_B=g[EVENT_COL],
            ).p_value
        )
    except Exception:
        lr = float("nan")
    return mw, lr


def analyze(msk: pd.DataFrame, genie: pd.DataFrame) -> Dict[str, object]:
    msk_set = {r for r in msk["REGIMEN_SET"] if r}
    genie_set = {r for r in genie["REGIMEN_SET"] if r}
    overlap = msk_set & genie_set

    msk_overlap_lines = msk[msk["REGIMEN_SET"].isin(overlap)]
    genie_overlap_lines = genie[genie["REGIMEN_SET"].isin(overlap)]

    rows = []
    for reg in overlap:
        m = msk[msk["REGIMEN_SET"] == reg]
        g = genie[genie["REGIMEN_SET"] == reg]
        em, eg, erm, erg, pe = _event_rate_test(m, g)
        mw, lr = _time_tests(m, g)
        rows.append({
            "regimen": _regimen_str(reg),
            "n_agents": len(reg),
            "n_msk": len(m),
            "n_genie": len(g),
            "events_msk": em,
            "events_genie": eg,
            "event_rate_msk": erm,
            "event_rate_genie": erg,
            "event_rate_diff": erg - erm,
            "p_event_ratio": pe,
            "median_pfs_msk": float(m[TIME_COL].median()),
            "median_pfs_genie": float(g[TIME_COL].median()),
            "p_mannwhitney_time": mw,
            "p_logrank": lr,
            "sufficient": len(m) >= SUFFICIENT_N and len(g) >= SUFFICIENT_N,
        })
    comparison = pd.DataFrame(rows).sort_values(
        ["n_genie", "n_msk"], ascending=False
    ).reset_index(drop=True)

    # Benjamini-Hochberg q-values on the log-rank p among "sufficient" regimens.
    comparison["q_logrank_bh"] = np.nan
    suff = comparison[comparison["sufficient"] & comparison["p_logrank"].notna()]
    if len(suff):
        order = suff["p_logrank"].to_numpy().argsort()
        ps = suff["p_logrank"].to_numpy()[order]
        n = len(ps)
        q = ps * n / (np.arange(n) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
        idx = suff.index.to_numpy()[order]
        comparison.loc[idx, "q_logrank_bh"] = q

    pooled = _pooled_stats(msk_overlap_lines, genie_overlap_lines)
    summary = {
        "msk_modelled_lines": int(len(msk)),
        "msk_distinct_regimens": int(len(msk_set)),
        "genie_candidate_lines": int(len(genie)),
        "genie_distinct_regimens": int(len(genie_set)),
        "overlapping_regimens": int(len(overlap)),
        "overlap_pct_of_msk_regimens": round(100 * len(overlap) / max(1, len(msk_set)), 1),
        "overlap_pct_of_genie_regimens": round(100 * len(overlap) / max(1, len(genie_set)), 1),
        "msk_lines_in_overlap": int(len(msk_overlap_lines)),
        "msk_lines_in_overlap_pct": round(100 * len(msk_overlap_lines) / max(1, len(msk)), 1),
        "genie_lines_in_overlap": int(len(genie_overlap_lines)),
        "genie_lines_in_overlap_pct": round(100 * len(genie_overlap_lines) / max(1, len(genie)), 1),
        "pooled_overlap": pooled,
    }
    return {
        "comparison": comparison,
        "summary": summary,
        "msk_overlap_lines": msk_overlap_lines,
        "genie_overlap_lines": genie_overlap_lines,
    }


def _pooled_stats(m: pd.DataFrame, g: pd.DataFrame) -> Dict[str, float]:
    em, eg, erm, erg, pe = _event_rate_test(m, g)
    mw, lr = _time_tests(m, g)
    return {
        "n_msk": int(len(m)),
        "n_genie": int(len(g)),
        "event_rate_msk": round(erm, 4),
        "event_rate_genie": round(erg, 4),
        "p_event_ratio": pe,
        "median_pfs_msk": float(m[TIME_COL].median()),
        "median_pfs_genie": float(g[TIME_COL].median()),
        "p_mannwhitney_time": mw,
        "p_logrank": lr,
    }


def _km_pooled(m: pd.DataFrame, g: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(m[TIME_COL], m[EVENT_COL], label=f"MSK (n={len(m)})")
    kmf.plot_survival_function(ax=ax, ci_show=True)
    kmf.fit(g[TIME_COL], g[EVENT_COL], label=f"GENIE (n={len(g)})")
    kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title("PFS for shared regimens: MSK vs GENIE (pooled)")
    ax.set_xlabel("Days from line start")
    ax.set_ylabel("PFS probability")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _km_top_regimens(msk: pd.DataFrame, genie: pd.DataFrame,
                     comparison: pd.DataFrame, path: Path, top: int = 6) -> None:
    regs = comparison[comparison["sufficient"]].head(top)["regimen"].tolist()
    if not regs:
        return
    ncol = 2
    nrow = int(np.ceil(len(regs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.4 * nrow), squeeze=False)
    for i, reg in enumerate(regs):
        ax = axes[i // ncol][i % ncol]
        m = msk[msk["regimen"] == reg]
        g = genie[genie["regimen"] == reg]
        kmf = KaplanMeierFitter()
        kmf.fit(m[TIME_COL], m[EVENT_COL], label=f"MSK n={len(m)}")
        kmf.plot_survival_function(ax=ax, ci_show=False)
        kmf.fit(g[TIME_COL], g[EVENT_COL], label=f"GENIE n={len(g)}")
        kmf.plot_survival_function(ax=ax, ci_show=False)
        ax.set_title(reg if len(reg) < 48 else reg[:45] + "...", fontsize=9)
        ax.set_xlabel("Days"); ax.set_ylabel("PFS")
    for j in range(len(regs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("PFS by shared regimen: MSK vs GENIE", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(outdir: Path = OUTDIR, exclude_agents: FrozenSet[str] = frozenset()) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    print("Building MSK pre-collapse line regimens...")
    msk = build_msk_line_regimens()
    print("Building GENIE line regimens...")
    genie = build_genie_line_regimens()

    if exclude_agents:
        before = len(msk), len(genie)
        msk = msk[~msk["REGIMEN_SET"].map(lambda s: bool(s & exclude_agents))].reset_index(drop=True)
        genie = genie[~genie["REGIMEN_SET"].map(lambda s: bool(s & exclude_agents))].reset_index(drop=True)
        print(f"  excluding lines containing {sorted(exclude_agents)}: "
              f"MSK {before[0]}->{len(msk)}, GENIE {before[1]}->{len(genie)}")

    drop = ["REGIMEN_SET"]
    msk.drop(columns=drop).to_csv(outdir / "msk_line_regimens.csv", index=False)
    genie.drop(columns=drop).to_csv(outdir / "genie_line_regimens.csv", index=False)

    result = analyze(msk, genie)
    result["summary"]["excluded_agents"] = sorted(exclude_agents)
    result["comparison"].to_csv(outdir / "regimen_pfs_comparison.csv", index=False)
    with open(outdir / "overlap_summary.json", "w", encoding="utf-8") as fh:
        json.dump(result["summary"], fh, indent=2)

    _km_pooled(result["msk_overlap_lines"], result["genie_overlap_lines"],
               outdir / "km_pooled_msk_vs_genie.png")
    _km_top_regimens(msk, genie, result["comparison"],
                     outdir / "km_top_regimens.png")

    s = result["summary"]
    print(f"  MSK regimens={s['msk_distinct_regimens']}, "
          f"GENIE regimens={s['genie_distinct_regimens']}, "
          f"overlap={s['overlapping_regimens']} "
          f"({s['overlap_pct_of_msk_regimens']}% of MSK)")
    print(f"  GENIE lines covered by shared regimens: "
          f"{s['genie_lines_in_overlap']}/{s['genie_candidate_lines']} "
          f"({s['genie_lines_in_overlap_pct']}%)")
    print(f"  pooled event rate MSK={s['pooled_overlap']['event_rate_msk']} vs "
          f"GENIE={s['pooled_overlap']['event_rate_genie']} "
          f"(p_event={s['pooled_overlap']['p_event_ratio']:.2e}, "
          f"p_logrank={s['pooled_overlap']['p_logrank']:.2e})")
    print(f"  wrote outputs to {outdir}/")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-investigative", action="store_true",
        help="drop lines whose regimen contains INVESTIGATIVE (established regimens only)",
    )
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    exclude = INVESTIGATIVE_AGENTS if args.exclude_investigative else frozenset()
    outdir = args.outdir or (
        OUTDIR.parent / "regimen_overlap_established" if exclude else OUTDIR
    )
    run(outdir, exclude)


if __name__ == "__main__":
    main()
