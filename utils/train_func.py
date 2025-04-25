import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from Models.SATNet import SATNet
from Models.SeEANet import SeEANet
from Models.MyoNet import MyoNet
from Models.SATNet_E import SATNet_E
from Models.SATNet_N import SATNet_N
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from utils.tools import make_datasets,make_dir
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import csv
# 计算性能指标
import matplotlib.pyplot as plt
from Models.SATNet import SATNet
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

def train_func(epochs=2000,modelName="SAT", train_num=5,
    data_Dir="/home/admin123/SATData/data",
    save_Dir = "nn",
    peopleList=['S01'], exp_class="MJ",
    cluster_num=6, fusionMethod="PCA", 
    windowLength=256, stepLength=1, delta_T=20, train_ratio=0.8, PreNum=3,
    formatted_time = '0', batch_size = 512):

    if modelName == "SAT_N":
        fusionMethod = "NNWA"
    


    # 解析数据集
    emg_data, angle_data, afterClusterEmgIndex = make_datasets(data_Dir, peopleList, exp_class, cluster_num, fusionMethod, 
                                        windowLength, stepLength, delta_T)
    
    semgData = torch.tensor(emg_data, dtype=torch.float32)
    angleData = torch.tensor(angle_data, dtype=torch.float32)
    print(f"训练参数：网络：{modelName}, 簇:{cluster_num}, 间隔时间: {delta_T}, 数据融合算法: {fusionMethod}, 窗口大小：{windowLength}")
    print("semg 数据形状为：", semgData.shape)
    print("angle 数据形状为：", angleData.shape)
    dataset = TensorDataset(semgData, angleData)

    # 定义划分比例
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    rmse, mae, r2 = [[] for _ in range(train_num)], [[] for _ in range(train_num)], [[] for _ in range(train_num)]
    
    checkpoint_save_Dir = os.path.join(save_Dir, modelName, formatted_time,f"{exp_class}_{fusionMethod}_{delta_T}_{windowLength}_{cluster_num}")
    make_dir(checkpoint_save_Dir)

    for train_index in range(train_num):

        print(f"开始第{train_index+1}次训练！！！")
        # 数据集分割
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        # 数据加载
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if modelName != "SAT_N":
            # 初始化模型、优化器和损失函数
            if modelName == "SAT":
                model = SATNet(PreNum, cluster_num=cluster_num)
            elif modelName == "SeEANet":
                model = SeEANet(PreNum, cluster_num=cluster_num)
            elif modelName == "MyoNet":
                model = MyoNet(PreNum, cluster_num=cluster_num)
            elif modelName == "SAT_E":
                model = SATNet_E(PreNum, cluster_num=cluster_num)
        else:
            model = SATNet_N(PreNum, IndexL=afterClusterEmgIndex)

        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        criterion = nn.MSELoss()
        criterion = criterion.cuda()
        checkpoint_save_path = os.path.join(checkpoint_save_Dir, "best_%d.pth"%(train_index+1))

        best_loss = float('inf')
        train_running_loss_ls = []
        test_running_loss_ls = []
        for epoch in range(1, epochs+1):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step()
            train_running_loss_ls.append(running_loss)
            
            # 测试模型
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
            test_running_loss_ls.append(test_loss)

            if epoch % 100 == 0:

                print('Epoch %d, train_idnex=%d, Train Loss: %.5f, Test Loss: %.5f'%(epoch, train_index+1, running_loss, test_loss*train_size/test_size))

            if running_loss < best_loss:
                best_loss = running_loss
                torch.save(model.state_dict(), checkpoint_save_path)


        file_path = os.path.join(checkpoint_save_Dir, "output_%d.csv"%(train_index+1))
        # 将两个列表写入到 CSV 文件
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头（可选）
            writer.writerow(["train_loss", "test_loss"])
            
            # 写入数据
            for item1, item2 in zip(train_running_loss_ls, test_running_loss_ls):
                writer.writerow([item1, item2])
        print(f"loss csv 文件已保存到: {file_path}")

        # 绘制 loss 曲线
        png_path  = os.path.join(checkpoint_save_Dir, "loss_%d.png"%(train_index+1))
        plt.figure()
        plt.plot(np.arange(len(train_running_loss_ls))+1, train_running_loss_ls, label="train loss")
        plt.plot(np.arange(len(test_running_loss_ls))+1, test_running_loss_ls, label="test loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss")
        plt.legend()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"loss png 文件已保存到: {png_path}")


        # 初始化模型结构
        # 初始化模型、优化器和损失函数
        if modelName == "SAT":
            model_eval = SATNet(PreNum, cluster_num=cluster_num)
        elif modelName == "SeEANet":
            model_eval = SeEANet(PreNum, cluster_num=cluster_num)
        elif modelName == "MyoNet":
            model_eval = MyoNet(PreNum, cluster_num=cluster_num)

        model_eval.load_state_dict(torch.load(checkpoint_save_path))

        model_eval = model
        model_eval.eval()
        truthAnglesList = []
        preAnglesList = []
        for inputs, labels in test_loader:
            truthAnglesList.append(labels)
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model_eval(inputs)
            preAnglesList.append(outputs.cpu().detach().numpy())

        truthAngles = np.concatenate(truthAnglesList, axis=0)
        preAngles = np.concatenate(preAnglesList, axis=0)

        tru_angle_1 = truthAngles[:,0].T
        tru_angle_2 = truthAngles[:,1].T
        tru_angle_3 = truthAngles[:,2].T

        pre_angle_1 = preAngles[:,0].T
        pre_angle_2 = preAngles[:,1].T
        pre_angle_3 = preAngles[:,2].T

        plt.figure()
        plt.hist(np.abs(tru_angle_1-pre_angle_1), bins=100, alpha=0.8, label="elv_angle")
        plt.hist(np.abs(tru_angle_2-pre_angle_2), bins=100, alpha=0.8, label="shoulder_elv")
        plt.hist(np.abs(tru_angle_3-pre_angle_3), bins=100, alpha=0.8, label="elbow_flexion")
        plt.xlim((0, 2))
        plt.legend()
        plt.title('Histogram Example')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

        # 显示网格线
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(checkpoint_save_Dir, "test_error_%d.png"%(train_index+1)), dpi=300, bbox_inches='tight')
        plt.close()

        rmse[train_index].append(np.sqrt(mean_squared_error(tru_angle_1, pre_angle_1)))
        rmse[train_index].append(np.sqrt(mean_squared_error(tru_angle_2, pre_angle_2)))
        rmse[train_index].append(np.sqrt(mean_squared_error(tru_angle_3, pre_angle_3)))

        mae[train_index].append(mean_absolute_error(tru_angle_1, pre_angle_1))
        mae[train_index].append(mean_absolute_error(tru_angle_2, pre_angle_2))
        mae[train_index].append(mean_absolute_error(tru_angle_3, pre_angle_3))

        r2[train_index].append(r2_score(tru_angle_1, pre_angle_1))
        r2[train_index].append(r2_score(tru_angle_2, pre_angle_2))
        r2[train_index].append(r2_score(tru_angle_3, pre_angle_3))
    
    # 将数据存储到字典中
    data_dict = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }

    json_path = os.path.join(checkpoint_save_Dir, "metrics.json")
    # 将字典保存为 JSON 文件
    with open(json_path, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)

    print(f"metrics.json 文件已保存到: {json_path}")

    return rmse, mae, r2


