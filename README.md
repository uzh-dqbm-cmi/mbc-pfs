# Description

This is the code repository for the manuscript "Reconstructing real-world metastatic lines of therapy enables progression risk stratification in breast cancer".
Every number cited in the manuscript can be found in `manuscript_data.txt` and `extended_data.txt`.

# Third-party code

Portions of this repository adapt code from [survival-intro](https://github.com/georgehc/survival-intro) by George H. Chen, used under the MIT License.
See `third_party_licenses/survival-intro/LICENSE` for details.

The above repository imports `dl_lib/pycox` and `dl_lib/torchtuples`, which are by Haavard Kvamme and are under a BSD 2-clause license.
The licenses of those two packages are included in their respective directory under `dl_lib`

# Data

- MSK-CHORD: the analysis uses the [msk_chord_2024](https://www.cbioportal.org/study/summary?id=msk_chord_2024) study as released on 2024-11-19 (cBioPortal datahub commit `c3b2da6168`). The current cBioPortal download is a later release (it has no `data_timeline_tumor_markers` file, and its treatment, sample and mutation files changed) and does not reproduce the cohort. The commands below fetch the 17 files the code reads from that commit, under the `.tsv` names the code expects, plus `brca_dx_1st_seq_OS.csv` from [msk-chord-figures-public](https://github.com/clinical-data-mining/msk-chord-figures-public):
  ```
  mkdir -p data/msk_chord_2024
  C=c3b2da6168ae72c3b9ced75912d9c5d105c3f70a
  for f in data_clinical_patient data_clinical_sample data_cna data_gene_panel_matrix data_mutations \
           data_timeline_cancer_presence data_timeline_diagnosis data_timeline_mmr data_timeline_pdl1 \
           data_timeline_performance_status data_timeline_progression data_timeline_radiation \
           data_timeline_specimen data_timeline_surgery data_timeline_treatment \
           data_timeline_tumor_markers data_timeline_tumor_sites; do
    curl -L -o data/msk_chord_2024/$f.tsv https://media.githubusercontent.com/media/cBioPortal/datahub/$C/public/msk_chord_2024/$f.txt
  done
  curl -L -o data/msk_chord_2024/brca_dx_1st_seq_OS.csv https://raw.githubusercontent.com/clinical-data-mining/msk-chord-figures-public/main/data/brca_dx_1st_seq_OS.csv
  ```
- AACR Project GENIE BPC Breast Cancer (BrCa) v1.0-public: download and place under `data/genie`. GENIE data is accessed controlled, and derived files are not distributed in this repo.

# Environment

`environment.txt` records Python 3.13.5 and the pinned packages. Install them with `pip install -r requirements.txt`.

# Running

1. Put the MSK-CHORD files in `data/msk_chord_2024/` and the GENIE BPC files in `data/genie/`.
2. Run the commands in `commands.txt` in order, from the repository root.
3. The last command, `python manuscript.py`, writes the results:
   - `manuscript_data.txt`: Tables 1-2 and the numbers behind Figures 2-5, in manuscript order; Figure 1 contains separate BioRender artwork
   - `extended_data.txt`: Extended Data Tables 1-28 and the numbers behind Extended Data Figures 1-6, in Extended Data order
   - the figures, in `manuscript/figures/`

`python print_tables.py t2 ed15` prints single tables (`--list` lists the keys).

