from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from config import AMPLIFICATION_GENES, DELETION_GENES, MUTATION_GENES

from .base import ModalityFrame

CNA_FILE = "data_cna_transposed.tsv"
MUTATION_FILE = "data_mutations.tsv"


def _load_mutation_data(ctx) -> pd.DataFrame:
    mutation_path = ctx.data_dir / MUTATION_FILE
    mutation = pd.read_csv(
        mutation_path,
        sep="\t",
        dtype=str,
        usecols=["Tumor_Sample_Barcode", "Hugo_Symbol"],
    )
    mutation["PATIENT_ID"] = mutation["Tumor_Sample_Barcode"].apply(_extract_patient_id)
    mutation = mutation[mutation["PATIENT_ID"].isin(ctx.type_specific_patients)]
    mutation["Hugo_Symbol"] = mutation["Hugo_Symbol"].fillna("").astype(str).str.strip()
    mutation = mutation[mutation["Hugo_Symbol"] != ""]
    return mutation


def _load_cna_data(ctx) -> pd.DataFrame:
    cna_path = ctx.data_dir / CNA_FILE
    cna = pd.read_csv(cna_path, sep="\t", dtype=str)
    cna["PATIENT_ID"] = cna["SAMPLE_ID"].apply(_extract_patient_id)
    cna = cna[cna["PATIENT_ID"].isin(ctx.type_specific_patients)]
    return cna.drop(columns=["SAMPLE_ID"])


def _build_mutation_flags(
    mutation: pd.DataFrame,
    patient_index: pd.Index,
    genes: Sequence[str],
    measured_patients: Sequence[str],
) -> pd.DataFrame:
    filtered = mutation[mutation["Hugo_Symbol"].isin(genes)].copy()
    if filtered.empty:
        matrix = pd.DataFrame(index=patient_index, columns=genes, dtype="float")
    else:
        filtered = filtered.drop_duplicates(["PATIENT_ID", "Hugo_Symbol"])
        filtered["value"] = 1
        matrix = filtered.pivot(
            index="PATIENT_ID", columns="Hugo_Symbol", values="value"
        ).reindex(index=patient_index, columns=genes)
    measured_index = pd.Index(measured_patients).intersection(matrix.index)
    if not matrix.empty and len(measured_index) > 0:
        matrix.loc[measured_index] = matrix.loc[measured_index].fillna(0)
    return matrix


def _build_cna_flags(
    cna: pd.DataFrame,
    patient_index: pd.Index,
    genes: Sequence[str],
    target_value: float,
) -> pd.DataFrame:
    genes = list(genes)
    selected = cna[["PATIENT_ID"] + genes].copy()
    numeric = selected.copy()
    for gene in genes:
        numeric[gene] = pd.to_numeric(numeric[gene])

    grouped = numeric.groupby("PATIENT_ID", sort=False)[genes]
    has_target = grouped.agg(
        lambda col: 1 if (col == target_value).any() else 0
    ).astype("float")
    has_measurement = grouped.agg(lambda col: col.notna().any())

    flags = has_target.where(has_measurement, other=pd.NA)
    flags = flags.reindex(index=patient_index, columns=genes)
    return flags


def _extract_patient_id(sample_id: str) -> str:
    return "-".join(str(sample_id).split("-")[:2])


def _intersect_genes(
    config_genes: Sequence[str], available: Iterable[str]
) -> list[str]:
    available_set = {str(gene) for gene in available if str(gene)}
    return sorted(set(config_genes) & available_set)


def build_genomics_features(ctx) -> ModalityFrame:
    patient_index = pd.Index(
        sorted(str(pid) for pid in ctx.type_specific_patients), name="PATIENT_ID"
    )
    frame = pd.DataFrame(index=patient_index)

    cna = _load_cna_data(ctx)
    amp_genes = [gene for gene in AMPLIFICATION_GENES if gene in cna.columns]
    del_genes = [gene for gene in DELETION_GENES if gene in cna.columns]

    # assert sorted(cna["PATIENT_ID"].unique()) == ctx.type_specific_patients
    assert cna[amp_genes].notna().all().all()
    assert cna[del_genes].notna().all().all()

    if amp_genes:
        amp_flags = _build_cna_flags(cna, patient_index, amp_genes, target_value=2.0)
        amp_flags = amp_flags.rename(columns=lambda g: f"GENOMICS_AMPLIFICATION_{g}")
        frame = frame.join(amp_flags, how="inner", validate="one_to_one")

    if del_genes:
        del_flags = _build_cna_flags(cna, patient_index, del_genes, target_value=-2.0)
        del_flags = del_flags.rename(columns=lambda g: f"GENOMICS_DELETION_{g}")
        frame = frame.join(del_flags, how="inner", validate="one_to_one")

    mutation = _load_mutation_data(ctx)
    mutation_patients = pd.Index(sorted(set(mutation["PATIENT_ID"])))
    mutation_genes = _intersect_genes(MUTATION_GENES, mutation["Hugo_Symbol"].unique())
    if mutation_genes:
        mutation_flags = _build_mutation_flags(
            mutation,
            patient_index,
            mutation_genes,
            mutation_patients,
        )
        mutation_flags = mutation_flags.rename(
            columns=lambda g: f"GENOMICS_MUTATION_{g}"
        )
        # for patients that are in the main frame but have no mutation data,
        # we set GENOMICS_MISSING flag as 1, and impute the GENOMICS_MUTATION_* flag as 0
        missing_mutation_patients = patient_index.difference(mutation_patients)
        mutation_flags.loc[missing_mutation_patients, "GENOMICS_MISSING"] = 1
        mutation_flags["GENOMICS_MISSING"] = mutation_flags["GENOMICS_MISSING"].fillna(
            0
        )
        mutation_flags.loc[
            missing_mutation_patients, mutation_flags.columns != "GENOMICS_MISSING"
        ] = 0
        frame = frame.join(mutation_flags, how="inner", validate="one_to_one")

    frame = frame.astype("Int8").reset_index()
    return ModalityFrame(name="GENOMICS", frame=frame, temporal=False)
