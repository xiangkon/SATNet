import os
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from utils.dataFusionMethod import *

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
def make_datasets(data_Dir, peopleList, exp_class, cluster_num, fusionMethod, windowLength, stepLength, delta_T):
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
            angleDataArray = angleDataArray[:,[1,2,4]]*180/np.pi
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
            if fusionMethod == "KPCA":
                subMergedEmgData = KPCA_func(subEmgData)

            elif fusionMethod == "PCA":
                subMergedEmgData = PCA_func(subEmgData)

            elif fusionMethod == "WA":
                subMergedEmgData = WA_func(subEmgData)
            
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

