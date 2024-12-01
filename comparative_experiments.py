import time
import pickle
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from imblearn.under_sampling import RandomUnderSampler
import lightgbm
import catboost
import xgboost
import psutil
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset

# Setup configurations
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore")

# Global plot configuration
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# ---- Neural Network Model ----
class DNN(nn.Module):
    def __init__(self, input_size, output_size,alpha):
        super(DNN, self).__init__()
        self.input_size=input_size 
        self.output_size=output_size
        self.alpha=alpha
        a =input_size
        self.pipline = nn.Sequential(
             nn.Linear(input_size, 2*a),
             
             nn.Sigmoid(),
            nn.Linear(2*a, 2*a),
             nn.Sigmoid(),
            nn.Linear(2*a, 4*a),
             nn.Sigmoid(),
            nn.Linear(4*a, 4*a),
             nn.Sigmoid(),
            nn.Linear(4*a, output_size),
        ).cuda()
        self.focal_loss = FocalLoss(alpha=alpha).cuda()
        # self.focal_loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)  # 使用正态分布进行初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0) 

    def forward(self, x):
        x = self.pipline(x)

        return x

    def fit(self, inputs, labels, epochs=200):
        try:
            inputs = torch.Tensor(inputs).cuda()
            labels = torch.Tensor(labels).cuda().reshape(-1,1)
        except:
            pass
        train_data = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset=train_data, batch_size=10240, shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0
            for i, (inputs, labels) in enumerate(dataloader):
                outputs = self(inputs)
                loss = self.focal_loss(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
            print(f"Epoch: {epoch} Loss: {epoch_loss}")

    def predict(self, X):
        with torch.no_grad():
            if type(X)==pd.DataFrame:
                tensors = torch.tensor(X.values, dtype=torch.float).cuda()
            else:
                tensors = torch.tensor(X, dtype=torch.float).cuda()
            outputs = self(tensors)
            predictions = torch.round(torch.sigmoid(outputs))
        return predictions.cpu().numpy()
    def get_params(self, deep=True):
        return {"input_size":  self.input_size, 
                "output_size": self.output_size, 
                "alpha":       self.alpha
                }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# ---- Focal Loss ----
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.09, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (loss * alpha_t).mean()

# ---- Preprocessing Functions ----
def preprocess_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler

def preprocess_data_predict(X, scaler):
    return scaler.transform(X)

# ---- Model Building ----
def build_model(model_type, X_train, y_train):
    if model_type == 'LogisticRegression':
        model = LogisticRegression()
    elif model_type=='2GDNN-FL':
        model =  DNN(X_train.shape[1], 1, alpha=(len(y_train) - y_train.sum()) / len(y_train)).cuda()
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier()
    elif model_type == 'GBDT':
        model = GradientBoostingClassifier()
    elif model_type == 'KNN':
        model = KNeighborsClassifier()
    elif model_type == 'BayesianNetwork':
        model = GaussianNB()
    elif model_type == 'AdaBoost':
        model = AdaBoostClassifier()
    elif model_type == 'SVM':
        model = LinearSVC()
    elif model_type == 'XGBoost':
        model = xgboost.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == 'Stacking':
        estimators = [
            ('DNN', DNN(X_train.shape[1], 1, alpha=(len(y_train) - y_train.sum()) / len(y_train)).cuda()),
            ('CatBoost', catboost.CatBoostClassifier(silent=True)),
            ('RandomForest', RandomForestClassifier()),
            ('XGBoost', lightgbm.LGBMClassifier())
        ]
        final_estimator = LogisticRegression()
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    return model

# ---- Model Training ----
def train_model(model_type, X_train, y_train, resample_rate=(1, 3)):
    X_train, scaler = preprocess_data(X_train)
    resampler = RandomUnderSampler(sampling_strategy={1: resample_rate[0] * len(y_train.loc[y_train == 1]),
                                                      0: resample_rate[1] * len(y_train.loc[y_train == 1])})
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    model = build_model(model_type, X_resampled, y_resampled)
    return model, scaler

# ---- Model Prediction ----
def predict_model(model_type, X, model, scaler):
    X_scaled = preprocess_data_predict(X, scaler)
    return model.predict(X_scaled)

# ---- Main Execution ----
def main(X, y, n_splits=5):
    models = [
        # '2GDNN-FL'
        # 'Stacking',
        'SVM',
        'LogisticRegression',
        'DecisionTree',
        'GBDT',
        'KNN',
        'BayesianNetwork',
        'AdaBoost',
        'XGBoost'

    ]

    all_results = {}
    for model_type in models:
        print(f"Training model: {model_type}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        results = {'f1_score': [], 'matthews_corrcoef': [], 'roc_auc_score': []}
        memory_usage = []
        gpu_memory_usage = []
        training_time = []

        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            print(f"Fold {fold + 1}/{n_splits}")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 初始化内存统计
            torch.cuda.reset_peak_memory_stats()
            process = psutil.Process()

            start_time = time.time()

            model, scaler = train_model(model_type, X_train, y_train, resample_rate=(1, 3))
            y_pred = predict_model(model_type, X_test, model, scaler)

            end_time = time.time()
            training_time.append(end_time - start_time)
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_memory = process.memory_info().rss / (1024 * 1024)

            gpu_memory_usage.append(peak_gpu_memory)
            memory_usage.append(peak_memory)

            results['f1_score'].append(f1_score(y_test, y_pred, average='macro'))
            results['matthews_corrcoef'].append(matthews_corrcoef(y_test, y_pred))
            results['roc_auc_score'].append(roc_auc_score(y_test, y_pred))

            # 输出当前折的评价指标和性能开销
            print(f"Fold {fold + 1} Results:")
            print(f"F1 Score: {results['f1_score'][-1]:.4f}")
            print(f"Matthews Correlation Coefficient: {results['matthews_corrcoef'][-1]:.4f}")
            print(f"ROC AUC Score: {results['roc_auc_score'][-1]:.4f}")
            print(f"Peak Memory Usage (RAM): {peak_memory:.2f} MB")
            print(f"Peak Memory Usage (GPU): {peak_gpu_memory:.2f} MB")
            print(f"Training Time: {end_time - start_time:.2f} seconds")
            print("\n")

        mean_results = {k: np.mean(v) for k, v in results.items()}
        std_results = {k: np.std(v) for k, v in results.items()}
        final_results = {k: f"{mean_results[k]:.4f} ± {std_results[k]:.4f}" for k in mean_results.keys()}

        mean_memory = np.mean(memory_usage)
        std_memory = np.std(memory_usage)
        mean_gpu_memory = np.mean(gpu_memory_usage)
        std_gpu_memory = np.std(gpu_memory_usage)
        mean_time = np.mean(training_time)
        std_time = np.std(training_time)

        all_results[model_type] = {
            'results': final_results,
            'memory_usage': f"{mean_memory:.2f} ± {std_memory:.2f} MB",
            'gpu_memory_usage': f"{mean_gpu_memory:.2f} ± {std_gpu_memory:.2f} MB",
            'training_time': f"{mean_time:.2f} ± {std_time:.2f} seconds"
        }

    # Print all results
    print("Final Results:")
    for model_type, results in all_results.items():
        print(f"Model: {model_type}")
        for k, v in results['results'].items():
            print(f"{k}: {v}")
        print(f"Peak Memory Usage (RAM): {results['memory_usage']}")
        print(f"Peak Memory Usage (GPU): {results['gpu_memory_usage']}")
        print(f"Training Time: {results['training_time']}")
        print("\n")

if __name__ == "__main__":
    data = pd.read_csv(r'data\LLCP2022.csv', index_col=0)
    print(data.shape)
    target = '_MICHD'
    data = data.drop(['CVDINFR4', 'CVDCRHD4', '_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE', 'SEQNO', '_PSU', 'CTELENM1', 'PVTRESD1',
                      'RESPSLCT', 'SAFETIME', 'DIABAGE4', 'NUMPHON4', 'CPDEMO1C', 'BLDSTFIT', 'QSTVER', 'QSTLANG', '_METSTAT'], axis=1)

    missing_values_count = data.isnull().sum()

    columns_to_drop = missing_values_count[missing_values_count > 50000].index
    data.drop(columns=columns_to_drop, inplace=True)
    data.dropna(how='any', axis=0, inplace=True)
    print(data.shape)

    X = data.drop([target], axis=1)
    y = data[target].apply(lambda x: 2 - x)  # Convert binary target (1, 0) to (2, 1)

    main(X, y)
    pass