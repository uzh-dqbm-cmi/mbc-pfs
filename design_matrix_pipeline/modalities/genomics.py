from __future__ import annotations

import pandas as pd

from config import AMPLIFICATION_GENES, DELETION_GENES, MUTATION_GENES

from .base import ModalityFrame

CNA_FILE = "data_cna_transposed.tsv"
MUTATION_FILE = "data_mutations.tsv"
GENE_PANEL_FILE = "data_gene_panel_matrix.tsv"


def _build_mutation_frame(
    ctx, sequencing_sample_ids: pd.Index
) -> tuple[pd.DataFrame, list[str]]:
    mutations = pd.read_csv(
        ctx.data_dir / MUTATION_FILE,
        sep="\t",
        dtype=str,
        usecols=["Tumor_Sample_Barcode", "Hugo_Symbol"],
    )
    mutations = mutations[mutations["Hugo_Symbol"].isin(MUTATION_GENES)].copy()

    mutation_by_sample = (
        mutations.assign(value=1)
        .drop_duplicates(["Tumor_Sample_Barcode", "Hugo_Symbol"])
        .pivot(
            index="Tumor_Sample_Barcode",
            columns="Hugo_Symbol",
            values="value",
        )
        .rename_axis("SAMPLE_ID")
        .reindex(index=sequencing_sample_ids, columns=MUTATION_GENES)
        .fillna(0)
        .astype("Int8")
        .reset_index()
    )
    mutation_cols = []
    for gene in MUTATION_GENES:
        col = f"GENOMICS_MUTATION_{gene}"
        mutation_by_sample = mutation_by_sample.rename(columns={gene: col})
        mutation_cols.append(col)

    if not mutation_by_sample["SAMPLE_ID"].is_unique:
        raise ValueError("Expected one mutation row per SAMPLE_ID after aggregation")
    return mutation_by_sample, mutation_cols


def _build_cna_frame(
    ctx, sequencing_sample_ids: pd.Index
) -> tuple[pd.DataFrame, list[str]]:
    cna_header = pd.read_csv(ctx.data_dir / CNA_FILE, sep="\t", dtype=str, nrows=0)
    available_amp_genes = [gene for gene in AMPLIFICATION_GENES if gene in cna_header]
    available_del_genes = [gene for gene in DELETION_GENES if gene in cna_header]

    cna = pd.read_csv(
        ctx.data_dir / CNA_FILE,
        sep="\t",
        dtype=str,
        usecols=["SAMPLE_ID", *available_amp_genes, *available_del_genes],
    )
    if not cna["SAMPLE_ID"].is_unique:
        raise ValueError("Expected one CNA row per SAMPLE_ID")

    cna = cna.set_index("SAMPLE_ID").reindex(sequencing_sample_ids)
    if cna.index.hasnans or cna.isna().all(axis=1).any():
        missing_samples = cna.index[cna.isna().all(axis=1)].tolist()
        raise ValueError(
            f"{len(missing_samples)} sequencing samples missing from CNA data: {missing_samples[:10]}"
        )

    available_curated_genes = available_amp_genes + available_del_genes
    if cna[available_curated_genes].isna().any(axis=None):
        raise ValueError("Unexpected NaN values in curated CNA inputs")

    cna_by_sample = pd.DataFrame({"SAMPLE_ID": sequencing_sample_ids})
    cna_cols: list[str] = []
    for gene in available_amp_genes:
        col = f"GENOMICS_AMPLIFICATION_{gene}"
        cna_by_sample[col] = cna[gene].eq("2").astype("Int8").to_numpy()
        cna_cols.append(col)
    for gene in available_del_genes:
        col = f"GENOMICS_DELETION_{gene}"
        cna_by_sample[col] = cna[gene].eq("-2").astype("Int8").to_numpy()
        cna_cols.append(col)

    return cna_by_sample, cna_cols


def build_genomics_features(ctx) -> ModalityFrame:
    patient_index = pd.Index(
        sorted(str(pid) for pid in ctx.type_specific_patients), name="PATIENT_ID"
    )

    samples = pd.read_csv(
        ctx.data_dir / "data_clinical_sample.tsv", sep="\t", dtype=str, skiprows=4
    )
    sequencing_samples = samples.loc[
        samples["CANCER_TYPE"].eq("Breast Cancer")
        & samples["PATIENT_ID"].isin(patient_index),
        ["SAMPLE_ID", "PATIENT_ID"],
    ].copy()
    if not sequencing_samples["PATIENT_ID"].is_unique:
        duplicate_patients = (
            sequencing_samples.loc[
                sequencing_samples["PATIENT_ID"].duplicated(keep=False), "PATIENT_ID"
            ]
            .sort_values()
            .tolist()
        )
        raise ValueError(
            "Expected one sample per patient; found duplicates for "
            f"{duplicate_patients[:10]}"
        )
    if not sequencing_samples["SAMPLE_ID"].is_unique:
        raise ValueError("Expected SAMPLE_ID values to be unique")
    if not sequencing_samples.shape[0] == len(patient_index):
        raise ValueError(
            f"Expected number of samples ({sequencing_samples.shape[0]}) to match number of patients ({len(patient_index)})"
        )

    sequencing_dates = pd.read_csv(
        ctx.data_dir / "data_timeline_specimen.tsv",
        sep="\t",
        dtype=str,
        usecols=["SAMPLE_ID", "START_DATE"],
    )
    sequencing_samples = sequencing_samples.merge(
        sequencing_dates,
        on="SAMPLE_ID",
        how="left",
        validate="one_to_one",
    ).rename(columns={"START_DATE": "SEQUENCING_DATE"})
    if sequencing_samples["SEQUENCING_DATE"].isna().any():
        raise ValueError("Missing sequencing date present")
    sequencing_samples["SEQUENCING_DATE"] = pd.to_numeric(
        sequencing_samples["SEQUENCING_DATE"], errors="coerce"
    ).astype("Int64")

    gene_panel = pd.read_csv(
        ctx.data_dir / GENE_PANEL_FILE,
        sep="\t",
        dtype=str,
        usecols=["SAMPLE_ID", "mutations", "cna"],
    )
    if not gene_panel["SAMPLE_ID"].is_unique:
        raise ValueError("Expected one gene-panel row per SAMPLE_ID")
    sequencing_samples = sequencing_samples.merge(
        gene_panel,
        on="SAMPLE_ID",
        how="left",
        validate="one_to_one",
    )
    if sequencing_samples[["mutations", "cna"]].isna().any(axis=None):
        missing_panel_samples = sequencing_samples.loc[
            sequencing_samples[["mutations", "cna"]].isna().any(axis=1), "SAMPLE_ID"
        ].tolist()
        raise ValueError(
            "Missing gene-panel assay metadata for sequencing samples: "
            f"{missing_panel_samples[:10]}"
        )

    sequencing_sample_ids = pd.Index(sequencing_samples["SAMPLE_ID"], name="SAMPLE_ID")
    mutation_by_sample, mutation_cols = _build_mutation_frame(
        ctx, sequencing_sample_ids
    )
    cna_by_sample, cna_cols = _build_cna_frame(ctx, sequencing_sample_ids)
    genomics_feature_cols = mutation_cols + cna_cols

    frame = sequencing_samples.merge(
        mutation_by_sample,
        on="SAMPLE_ID",
        how="left",
        validate="one_to_one",
    ).merge(
        cna_by_sample,
        on="SAMPLE_ID",
        how="left",
        validate="one_to_one",
    )
    if frame[genomics_feature_cols].isna().any(axis=None):
        raise ValueError("Unexpected NaN values after mutation/CNA feature assembly")

    frame["GENOMICS_MISSING"] = 0
    frame["GENOMICS_MISSING"] = frame["GENOMICS_MISSING"].astype("Int8")
    frame = frame[
        ["PATIENT_ID", "SEQUENCING_DATE", *genomics_feature_cols, "GENOMICS_MISSING"]
    ].copy()
    frame = frame.sort_values("PATIENT_ID").reset_index(drop=True)

    if not frame["PATIENT_ID"].is_unique:
        raise ValueError("Expected one genomics row per PATIENT_ID")
    if set(frame["PATIENT_ID"]) != set(patient_index):
        raise ValueError("Genomics frame patient coverage does not match outcomes")

    return ModalityFrame(name="GENOMICS", frame=frame, temporal=False)
