import os

class MakeDirClass():
    def __init__(self):
        self.ml_directories()
        self.rs_directories()

    def ml_directories(self):
        if not os.path.exists("./MaLer/Predictions/out/"):
            os.makedirs("./MaLer/Predictions/out/")

        for folder in ['compare_algos', 'predict_test']:
            if not os.path.exists("./MaLer/Plots/"+folder+"/"):
                os.makedirs("./MaLer/Plots/"+folder+"/")
          
        for type_al in ['svm', 'random_forest', 'log_reg', 'metrics']:
            if not os.path.exists("./MaLer/Models/"+type_al+"/"):
                os.makedirs("./MaLer/Models/"+type_al+"/")
            if(type_al == 'metrics'):
                break
            if not os.path.exists("./MaLer/Plots/confusion_matrix/"+type_al+"/"):
                os.makedirs("./MaLer/Plots/confusion_matrix/"+type_al+"/")
            if not os.path.exists("./MaLer/Plots/imbalance_classes/SMOTE/"+type_al+"/"):
                os.makedirs("./MaLer/Plots/imbalance_classes/SMOTE/"+type_al+"/")

    def rs_directories(self):
        if not os.path.exists("./RecSys/data_behavior/"):
            os.makedirs("./RecSys/data_behavior/")
        if not os.path.exists("./RecSys/utility_matrix/"):
            os.makedirs("./RecSys/utility_matrix/")
        if not os.path.exists("./RecSys/feature_matrix/"):
            os.makedirs("./RecSys/feature_matrix/")
        if not os.path.exists("./RecSys/out/test/"):
            os.makedirs("./RecSys/out/test/")
        if not os.path.exists("./RecSys/out/MF/Train/"):
            os.makedirs("./RecSys/out/MF/Train/")
        if not os.path.exists("./RecSys/out/MF/test/"):
            os.makedirs("./RecSys/out/MF/test/")
        if not os.path.exists("./RecSys/out/MF/Plots/"):
            os.makedirs("./RecSys/out/MF/Plots/")
        if not os.path.exists("./RecSys/out/baseline/sim/"):
            os.makedirs("./RecSys/out/baseline/sim/")
        if not os.path.exists("./RecSys/out/baseline/runtime/"):
            os.makedirs("./RecSys/out/baseline/runtime/")
        if not os.path.exists("./RecSys/out/baseline/test/"):
            os.makedirs("./RecSys/out/baseline/test/")
        for type_rs in ['MF', 'baseline', 'CAMF']:
            if not os.path.exists("./RecSys/out/"+type_rs+"/predictions/"):
                os.makedirs("./RecSys/out/"+type_rs+"/predictions/")
            if not os.path.exists("./RecSys/out/"+type_rs+"/models/"):
                os.makedirs("./RecSys/out/"+type_rs+"/models/")
        for camf in ['CAMF_C', 'CAMF_CU', 'CAMF_CI']:
            if not os.path.exists("./RecSys/out/CAMF/train/Plots/"+camf+"/"):
                os.makedirs("./RecSys/out/CAMF/train/Plots/"+camf+"/")
            if not os.path.exists("./RecSys/out/CAMF/train/"+camf+"/"):
                os.makedirs("./RecSys/out/CAMF/train/"+camf+"/")
            if not os.path.exists("./RecSys/out/CAMF/models/"+camf+"/"):
                os.makedirs("./RecSys/out/CAMF/models/"+camf+"/")
            if not os.path.exists("./RecSys/out/CAMF/predictions/"+camf+"/"):
                os.makedirs("./RecSys/out/CAMF/predictions/"+camf+"/")
            if not os.path.exists("./RecSys/out/CAMF/test/"+camf+"/"):
                os.makedirs("./RecSys/out/CAMF/test/"+camf+"/")
        if not os.path.exists("./RecSys/out/FM/models/"):
            os.makedirs("./RecSys/out/FM/models/")
        if not os.path.exists("./RecSys/out/FM/results/"):
            os.makedirs("./RecSys/out/FM/results/")
        if not os.path.exists("./RecSys/out/FM/train/"):
            os.makedirs("./RecSys/out/FM/train/")
         