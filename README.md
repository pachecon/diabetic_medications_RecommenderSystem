<p align="center">
  <a href="https://example.com/">
    <img src="./img/logo.png" alt="Logo" width="200" height="200">
  </a>
  <p style="text-align: justify">
    The project consists of making predictions if a patient with diabetes is readmitted to the hospital. In the case of readmission, a recommendation is given regarding the dosage of diabetic medications. If the insulin and metformin medications should be increased, decreased or stayed normal, considering also the results of blood glucose tests and other features. 
  </p>
</p>

## Table of contents

- [Table of contents](#table-of-contents)
- [Quick start](#quick-start)
  - [Main file `RS_diabetes.py`](#main-file-rsdiabetespy)
  - [Install libraries](#install-libraries)
  - [Run project](#run-project)
- [Description](#description)
- [What's included](#whats-included)
  - [Author](#author)
  - [Version](#version)

## Quick start

### Main file `RS_diabetes.py`

### Install libraries

- It is needed to install the libraries mentioned on requirements.txt
  ```
  pip3 install -r requirements.txt
  ```
- If there is an error for installing library pyFM, then please try:
  ```
  pip3 install git+https://github.com/coreylynch/pyFM
  ```
source: https://github.com/coreylynch/pyFM/blob/master/README.md

### Run project

- Run the main file
  ```
  python3 RS_diabetes.py type_of_recommender_system
  ```
- type_of_recommender_system:
  - baseline
    - item-based collaborative filter and matrix factorization are selected
  - camf
    - context-aware matrix factorization algorithms will run
  - fm
    - only factorization machine
  - all
    - All the recommender system algorithms will run

## Description

A complete explanation of the project can be fount on the report.

## What's included

File system architecture
*All folders are made by the make_dirs.py file

```text
main_folder/
└── data/
│   ├── diabetes_data_preprocessed.csv
│   ├── diabetic_data.csv
│   └── IDs_mapping.csv
└──src/
│   └──ml_model_src/
│   │  └──readmission_pred.py
│   │  └──pred_algo.py
│   │  └──split_data.py
│   │  └──plot_data.py
│   └──rs_model_src/
│   │  └──camf_src/
│   │  │  └──camf_c.py
│   │  │  └──camf_ci.py
│   │  │  └──camf_cu.py
│   │  └──camf.py
│   │  └──rs_baseline.py
│   │  └──matrix_factorization.py
│   │  └──rs_model_FM.py
│   │  └──rs_model.py
│   │  └──rs_modelCARS.py
│   │  └──feature_matrix_cl.py
│   │  └──utility_matrix_cl.py
│   │  └──plot_data_rs.py
│   └──make_dirs.py
│   └──utilities_data.py
MaLer/
│   └──Models/
│   │  └──log_reg/
│   │  └──random_forest/
│   │  └──svm/
│   │  └──metrics/
│   └──Plots/
│   │  └──compare_algos/
│   │  └──confusion_matrix/
│   │  └──imbalance_classes/
│   │  └──predict_test/
│   └──Predictions/
│   │  └──out/
RecSys/
│   └──feature_matrix/
│   └──out/
│   │  └──baseline
│   │  │  └──models
│   │  │  └──predictions
│   │  │  └──runtime
│   │  │  └──sim
│   │  └──CAMF
│   │  │  └──models
│   │  │  └──predictions
│   │  │  └──train
│   │  │  └──test
│   │  └──FM
│   │  │  └──models
│   │  │  └──resutls
│   │  │  └──train
│   │  └──MF
│   │  │  └──models
│   │  │  └──predictions
│   │  │  └──train
│   │  │  └──test
│   │  │  └──plot
│   │  └──test
│   └──utility_matrix/
img/
│
py3env/
```

### Author

<a href="https://github.com/pachecon" target="_blank">Arlette M. Perez Espinoza</a>

### Version

1.0.0
