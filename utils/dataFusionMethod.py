from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# KPCA 算法实现
def KPCA_func(data):
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # 初始化KPCA
    kpca = KernelPCA(n_components=1, kernel='rbf', gamma=15)

    # 应用KPCA
    X_kpca = kpca.fit_transform(X_scaled)
    return X_kpca

def PCA_func(data):
    # 标准化数据
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data)

    # 初始化PCA
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_std)
    return X_pca

def WA_func(data):
    
    return np.mean(data, axis=1, keepdims=True)