# 验证 delta_T
from utils.train_func import train_func
from datetime import datetime
from tqdm import tqdm
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from utils.tools import make_dir

def train():
    modelList = ['SeEANet', 'SAT', 'MyoNet']
    data_dict = {}
    # delta_Tlist = [1, 10, 15, 20, 25, 30, 40, 50, 60]
    delta_Tlist = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90 ,100, 120, 140, 160, 180, 200, 230, 260, 300]

    # people_List = ['S01', 'S02', "S03"]
    people_List = ['S01']

    rmseL = []
    maeL = []
    r2L = []
    formatted_time = datetime.now().strftime("%m-%d-%H:%M:%S")
    make_dir(os.path.join('/home/admin123/SATData/Run', formatted_time))
    print("开始训练")
    for delta_T in tqdm(delta_Tlist, desc="Processing", position=0):
        rmse, mae, r2 = train_func(modelName='SeEANet', epochs=1500, train_num=5, delta_T=delta_T,
                                formatted_time = formatted_time, peopleList=people_List, batch_size=256)
        rmse = np.array(rmse)
        rmse_m = np.mean(rmse,axis=0)
        rmse_s = np.std(rmse, axis=0)
        rmse_d = np.max(rmse, axis=0)-np.min(rmse, axis=0)
        rmseL.append(rmse_m)

        mae = np.array(mae)
        mae_m = np.mean(mae,axis=0)
        mae_s = np.std(mae, axis=0)
        mae_d = np.max(mae, axis=0)-np.min(mae, axis=0)
        maeL.append(mae_m)

        r2 = np.array(r2)
        r2_m = np.mean(r2,axis=0)
        r2_s = np.std(r2, axis=0)
        r2_d = np.max(r2, axis=0)-np.min(r2, axis=0)
        r2L.append(r2_m)

        data_dict[f"rmse_{delta_T}"] = [list(rmse_m), list(rmse_s), list(rmse_d)]
        data_dict[f"mae_{delta_T}"] = [list(mae_m), list(mae_s), list(mae_d)]
        data_dict[f"r2_{delta_T}"] = [list(r2_m), list(r2_s), list(r2_d)]

    # 指定 JSON 文件路径
    file_path = os.path.join('/home/admin123/SATData/Run/', formatted_time, "data.json")

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)

    print(f"字典已成功保存到 {file_path}")


    rmseL = np.array(rmseL)
    maeL = np.array(maeL)
    r2L = np.array(r2L)


    plt.figure()
    png_path = os.path.join('/home/admin123/SATData/Run', formatted_time, "RMSE.png")
    plt.plot(delta_Tlist, rmseL[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
    plt.plot(delta_Tlist, rmseL[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
    plt.plot(delta_Tlist, rmseL[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
    plt.xlabel("dleta_T")
    plt.ylabel("value")
    plt.grid()
    plt.legend()
    plt.title("RMSE")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    # plt.scatter(x, y, color='r', label='Points')  # 单独绘制点 


    plt.figure()
    png_path = os.path.join('/home/admin123/SATData/Run', formatted_time, "MAE.png")
    plt.plot(delta_Tlist, maeL[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
    plt.plot(delta_Tlist, maeL[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
    plt.plot(delta_Tlist, maeL[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
    plt.xlabel("dleta_T")
    plt.ylabel("value")
    plt.title("MAE")
    plt.grid()
    plt.legend()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')


    plt.figure()
    png_path = os.path.join('/home/admin123/SATData/Run', formatted_time, "R2.png")
    plt.plot(delta_Tlist, r2L[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
    plt.plot(delta_Tlist, r2L[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
    plt.plot(delta_Tlist, r2L[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
    plt.xlabel("dleta_T")
    plt.ylabel("value")
    plt.title("R2")
    plt.grid()
    plt.legend()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
