# Description

This is the code repository for the submission "Real-world EHR-derived progression-free survival across successive lines of therapy informs metastatic breast cancer risk stratification".

# Third-party code

Portions of this repository adapt code from [survival-intro](https://github.com/georgehc/survival-intro) by George H. Chen, used under the MIT License.
See `third_party_licenses/survival-intro/LICENSE` for details.

The above repository imports `dl_lib/pycox` and `dl_lib/torchtuples`, which are by Haavard Kvamme and are under a BSD 2-clause license.
The licenses of those two packages are included in their respective directory under `dl_lib`

# Environment

We use virtual environment to manage dependencies. Please install required packages using:
`pip install -r requirements.txt`.

# Data generation and analysis

## Raw data source

Raw data used in this study is released by MSK and available at https://www.cbioportal.org/study/summary?id=msk_chord_2024.
To reproduce results, please follow the link and download all source data and place them in `data/msk_chord_2024`.

## mLoT extraction and PFS labeling

Please run `python pipeline_overview.py` to run the mLoT extraction and PFS labeling pipeline. This will also generate the design matrix for machine learning modeling. Results will be saved under `data/`.

## Model training and evaluation

To train model and evaluate performance, simply run the commands in following order. This will run all experiments and save all results. Please note that the CPU-based nested CV experiments can be parallelized across models, so feel free to run them in parallel if you have the computational resources. The DL-based nested CV experiments are run sequentially.

```
python cpu_nested_cv.py --models coxph,gbsa,rsf
python cpu_nested_cv.py --no-treatment-masking --models coxph,gbsa,rsf
python dl_nested_cv.py --models deepsurv,deephit
python dl_nested_cv.py --no-treatment-masking --models deepsurv,deephit

python build_ablation_configs.py

python ablation.py --models deepsurv,deephit --runner dl
python ablation.py --models coxph,gbsa,rsf --runner cpu --max-parallel 32
```

# Data analysis

Simply run `python manuscript.py` to generate all results reported in the manuscript after all experiments are completed.
