import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

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
    # 计算协方差矩阵
    data_T = data.T
    covMatrix = calCovMatrix(data_T)

    # 计算相关系数
    correlationMatrix = calCorrelationMatrix(covMatrix)
    row_sums = np.sum(correlationMatrix, axis=1)
    weights = (row_sums)/(np.sum(row_sums))
    data_out = np.matmul(data, weights)
    return data_out.reshape(-1, 1)


def EVA_func(data):
    # 计算协方差矩阵
    data_T = data.T
    covMatrix = calCovMatrix(data_T)

    # 计算相关系数
    correlationMatrix = calCorrelationMatrix(covMatrix)
    row_sums = np.sum(correlationMatrix, axis=1)
    max_index = np.argmax(row_sums)
    return data[:,max_index].reshape(-1, 1)

# 建立文件夹
def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

# 计算协方差矩阵
def calCovMatrix(data):
    return np.cov(data)

# 计算相关系数
def calCorrelationMatrix(covMatrix):
    correlationMatrix = np.zeros((len(covMatrix),len(covMatrix[0])))
    for i in range(len(covMatrix)):
        for j in range(len(covMatrix[0])):
            correlationMatrix[i,j] = covMatrix[i,j] / (np.sqrt(covMatrix[i,i])*np.sqrt(covMatrix[j,j]))
    correlationMatrix = np.round(correlationMatrix, 2)
    return correlationMatrix

# 聚类
def clusterMethod(cluster_num, correlationMatrix):
    # 将相关性矩阵转换为距离矩阵
    distanceMatrix = 1 - np.abs(correlationMatrix)

    # 将距离矩阵转换为 condensed 形式
    condensedDistanceMatrix = squareform(distanceMatrix)

    # 使用层次聚类
    Z = linkage(condensedDistanceMatrix, method='ward')
    clusters = fcluster(Z, t=cluster_num, criterion='maxclust')

    # 根据聚类结果将数据分群
    afterClusterEmgIndex = []
    for i in range(cluster_num):
        emgIndexGroup = []
        for j in range(len(clusters)):
            if clusters[j] == i+1:
                emgIndexGroup.append(j+1)
        afterClusterEmgIndex.append(emgIndexGroup)

    return afterClusterEmgIndex



# 制作数据集
def make_datasets(data_Dir, peopleList, exp_class, cluster_num, fusionMethod, windowLength, stepLength, delta_T, motionL):
    allEmgData = []
    allAngleData = []
    dataSegFlags = []

    for people_index in peopleList:
        expList = os.listdir(os.path.join(data_Dir, people_index, exp_class))
        for exp_index in expList:
            dataDirPath = os.path.join(data_Dir, people_index, exp_class, exp_index)
            emgDataPath = os.path.join(dataDirPath, "emgFilt.csv")
            angleDataPath = os.path.join(dataDirPath, "humanPositions.csv")
            emgDataArray = pd.read_csv(emgDataPath).to_numpy()
            angleDataArray = pd.read_csv(angleDataPath).to_numpy()
            emgDataArray = emgDataArray[:,1:]
            angleDataArray = angleDataArray[:,motionL]*180/np.pi
            allEmgData.append(emgDataArray)
            allAngleData.append(angleDataArray)
            dataSegFlags.append(len(emgDataArray))

    # 将所有 emgDataArray 垂直拼接在一起
    if allEmgData:  # 确保列表不为空
        final_emg_data = np.concatenate(allEmgData, axis=0)
    else:
        print("没有找到有效的 emgDataArray，无法拼接。")

    # 将所有 angleDataArray 垂直拼接在一起
    if allEmgData:  # 确保列表不为空
        final_angle_data = np.concatenate(allAngleData, axis=0)
    else:
        print("没有找到有效的 allEmgData，无法拼接。")


    # 计算协方差矩阵
    final_emg_data_T = final_emg_data.T
    covMatrix = calCovMatrix(final_emg_data_T)

    # 计算相关系数
    correlationMatrix = calCorrelationMatrix(covMatrix)

    # 聚类并分群
    afterClusterEmgIndex = clusterMethod(cluster_num, correlationMatrix)

    # 数据融合

    if fusionMethod != "NNWA":
        mergedEmgData = []
        for i in afterClusterEmgIndex:
            subEmgData = final_emg_data[:,np.array(i)-1]

            if len(np.array(i)) != 1:
                if fusionMethod == "KPCA":
                    subMergedEmgData = KPCA_func(subEmgData)

                elif fusionMethod == "PCA":
                    subMergedEmgData = PCA_func(subEmgData)

                elif fusionMethod == "WA":
                    subMergedEmgData = WA_func(subEmgData)
                elif fusionMethod == "EVA":
                    subMergedEmgData = EVA_func(subEmgData)
            else:
                subMergedEmgData = subEmgData
            
            mergedEmgData.append(subMergedEmgData)
        finalMergedEmgData = np.concatenate(mergedEmgData, axis=1)
    else:
        finalMergedEmgData = final_emg_data

    # 开始制作数据集
    Length = windowLength + delta_T # 单个数据长度
    finalMergedEmgData_T = finalMergedEmgData.T
    finalAngleData_T = final_angle_data.T
    
    # 定义空列表
    emgList = []
    angleList = []

    startIndex = 0
    for dataFlag in dataSegFlags:
        singleExpEmgData = finalMergedEmgData_T[:, startIndex:startIndex+dataFlag]
        singleExpAngleData = finalAngleData_T[:,startIndex:startIndex+dataFlag]
        startIndex += dataFlag
        length = np.floor((len(singleExpEmgData[0])-Length/stepLength))
        for j in range(int(length)):
            semgSample = singleExpEmgData[:, stepLength*j:(windowLength+stepLength*j)]
            angleSample = singleExpAngleData[:, Length+stepLength*j]
            emgList.append(semgSample)
            angleList.append(angleSample)

    emg = np.array(emgList)
    angle = np.array(angleList)
    
    return emg, angle, afterClusterEmgIndex


def NormalizedArray(arr):
    arr_normalized = (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))
    return arr_normalized

def returnIndex(array):
     # 计算每一行的和
    row_sums = np.sum(array, axis=1)
    # 按照行和对原数组进行排序
    # 使用 argsort 获取行和的排序索引
    sorted_indices = np.argsort(row_sums)
    return sorted_indices

def calMeans(rmse, mae, r2, allFlag=True, thor=2):
    if allFlag:
        sumArray = NormalizedArray(rmse) + NormalizedArray(mae) - NormalizedArray(r2)

        sorted_indices = returnIndex(sumArray)
        sorted_mae = mae[sorted_indices]
        sorted_rmse = rmse[sorted_indices]
        sorted_r2 = r2[sorted_indices]
    else:
        sorted_mae = mae[returnIndex(mae)]
        sorted_rmse = rmse[returnIndex(rmse)]
        sorted_r2 = r2[returnIndex(r2)]


    if len(rmse) > 6:
        T = thor
    else:
        T = 1
    rmse_new, mae_new, r2_new = np.mean(sorted_rmse[:-T], axis=0), np.mean(sorted_mae[:-T], axis=0), np.mean(sorted_r2[:-T], axis=0)
    return rmse_new, mae_new, r2_new


def plot_func(save_Dir, mean_Value, Xlist, Xlabel, Ylabel="value", Xleft=0, Xright=256, name="RMSE.png"):

    plt.figure()
    png_path = os.path.join(save_Dir, name)
    plt.plot(Xlist, mean_Value[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
    plt.plot(Xlist, mean_Value[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
    plt.plot(Xlist, mean_Value[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.grid()
    plt.legend()
    plt.title(name[:-4])
    plt.xlim(Xleft, Xright)
    plt.savefig(png_path, dpi=300, bbox_inches='tight') 

def plot_func_ticks(save_Dir, mean_Value, Xlist, Xlabel, x_ticks, Ylabel="value", Xleft=0, Xright=256, name="RMSE.png"):

    plt.figure()
    png_path = os.path.join(save_Dir, name)
    plt.plot(Xlist, mean_Value[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
    plt.plot(Xlist, mean_Value[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
    plt.plot(Xlist, mean_Value[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.grid()
    plt.legend()
    plt.title("RMSE")
    plt.xlim(Xleft, Xright)
    plt.xticks(ticks=Xlist, labels=x_ticks)  # 设置 x 轴刻度标签
    plt.savefig(png_path, dpi=300, bbox_inches='tight') 
