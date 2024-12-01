import csv
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
import warnings
warnings.filterwarnings("ignore")
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
            # print(f"Epoch: {epoch} Loss: {epoch_loss}")

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

    if model_type == '2GDNN':
        estimators = [
            ('DNN', DNN(X_train.shape[1], 1, alpha=(len(y_train) - y_train.sum()) / len(y_train)).cuda()),
            ('CatBoost', catboost.CatBoostClassifier(silent=True)),
            ('RandomForest', RandomForestClassifier()),
            ('XGBoost', lightgbm.LGBMClassifier())
        ]
        final_estimator = LogisticRegression()
        model = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    
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

def measure_performance(X, y, model_type, resample_rate=(1, 3)):
    X_train, scaler = preprocess_data(X)
    resampler = RandomUnderSampler(sampling_strategy={1: resample_rate[0] * len(y.loc[y == 1]),
                                                     0: resample_rate[1] * len(y.loc[y == 1])})
    X_resampled, y_resampled = resampler.fit_resample(X_train, y)
    
    # Measure memory and time usage
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    start_time = time.time()
    
    model = build_model(model_type, X_resampled, y_resampled)
    
    end_time = time.time()
    peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_memory = process.memory_info().rss / (1024 * 1024)
    
    training_time = end_time - start_time
    
    return training_time, peak_memory, peak_gpu_memory

# Function to plot the results
def plot_results(data_sizes, times, mem_usages):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 设置主轴（时间）
    color = 'tab:red'
    ax1.set_xlabel('Data Size', fontsize=14)
    ax1.set_ylabel('Time (seconds)', color=color, fontsize=14)
    ax1.plot(data_sizes, times, color=color, marker='o', linestyle='-', linewidth=2, label='Training Time')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 设置次轴（内存使用）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Memory Usage (MB)', color=color, fontsize=14)
    ax2.plot(data_sizes, mem_usages, color=color, marker='s', linestyle='--', linewidth=2, label='Memory Usage')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)


    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12)

    # 调整布局
    fig.tight_layout()

    # 显示图表
    plt.show()
    pass

# Main function adjusted to measure performance at different data sizes
def main(X, y):
    model_type = '2GDNN'
    times = []
    mem_usages = []
    data_sizes = []
    results = []

    for size in range(1000, 340000, 20000):
        if size == 0:
            continue  # Skip the case where dataset size is 0
        print(f"Training on dataset size: {size}")
        X_sampled, y_sampled = X.iloc[:size], y.iloc[:size]
        training_time, peak_memory, peak_gpu_memory = measure_performance(X_sampled, y_sampled, model_type)
        
        times.append(training_time)
        mem_usages.append(peak_memory + peak_gpu_memory)
        data_sizes.append(size)
        
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Peak Memory Usage (RAM): {peak_memory:.2f} MB")
        print(f"Peak Memory Usage (GPU): {peak_gpu_memory:.2f} MB")
        print("\n")

        results.append({
            'Data_Size': size,
            'Training_Time': training_time,
            'Total_Memory_Usage': peak_memory + peak_gpu_memory
        })
    with open('result.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Data_Size', 'Training_Time', 'Total_Memory_Usage'])
        writer.writeheader()
        for result in results:
            writer.writerow(result)
            
    plot_results(data_sizes, times, mem_usages)
    pass

if __name__ == "__main__":
    data = pd.read_csv(r'data\LLCP2022.csv', index_col=0)
    print(data.shape)
    target = '_MICHD'
    data = data.drop(['CVDINFR4', 'CVDCRHD4', '_STATE', 'FMONTH', 'IDATE', 'IMONTH', 'IDAY', 'IYEAR', 'DISPCODE', 'SEQNO', '_PSU', 'CTELENM1', 'PVTRESD1',
                      'RESPSLCT', 'SAFETIME', 'DIABAGE4', 'NUMPHON4', 'CPDEMO1C', 'BLDSTFIT', 'QSTVER', 'QSTLANG', '_METSTAT'], axis=1)

    # 计算每列中缺失值的数量
    missing_values_count = data.isnull().sum()

    # 找到缺失值数量大于10000的列
    columns_to_drop = missing_values_count[missing_values_count > 50000].index
    # 删除这些列
    data.drop(columns=columns_to_drop, inplace=True)
    data.dropna(how='any', axis=0, inplace=True)
    print(data.shape)

    X = data.drop([target], axis=1)
    y = data[target].apply(lambda x: 2 - x)  # Convert binary target (1, 0) to (2, 1)

    main(X, y)