# 多个文件夹

import json
import numpy as np
import os
import matplotlib.pyplot as plt

delta_Tlist = []
# 自定义排序函数：提取倒数第二个数字
def sort_key(folder_name):
    # 假设文件夹名称格式为 MJ_PCA_1_256，提取倒数第二个数字
    parts = folder_name.split('_')  # 按下划线分割
    if len(parts) >= 2 and parts[-2].isdigit():  # 确保倒数第二个部分是数字
        delta_Tlist.append(int(parts[-2]))
        return int(parts[-2])  # 返回倒数第二个数字作为排序键
    else:
        return float('inf')  # 如果不符合格式，放在最后


parent_folderL = ['/home/admin123/SATData/Run/SeEANet/04-15-10:59:08',
                  '/home/admin123/SATData/Run/SeEANet/04-15-10:59:08（复件）']  
sorted_subfolderL = []
for parent_folder in parent_folderL:
    subfolders = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]
# 对子文件夹进行排序
    sorted_subfolders = sorted(subfolders, key=sort_key)
    sorted_subfolderL.append(sorted_subfolders)

mean_rmseL = []
mean_maeL = []
mean_r2L = []
delta_Tlist = sorted(delta_Tlist)
# 打印排序后的子文件夹列表
print("排序后的子文件夹列表：")
for folder in sorted_subfolders:
    file_path = os.path.join(parent_folder, folder, "metrics.json")
    # 打开并读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 将 JSON 数据加载为 Python 对象

    rmse = np.array(data['rmse'])
    rmse_m = np.mean(rmse, axis=0)
    mean_rmseL.append(rmse_m)

    mae = np.array(data['mae'])
    mae_m = np.mean(mae,axis=0)
    mean_maeL.append(mae_m)

    r2 = np.array(data['r2'])
    r2_m = np.mean(r2,axis=0)
    mean_r2L.append(r2_m)


mean_maeL = np.array(mean_maeL)
mean_rmseL = np.array(mean_rmseL)
mean_r2L = np.array(mean_r2L)


plt.figure()
# png_path = os.path.join('/home/admin123/SATData/Run', formatted_time, "RMSE.png")
plt.plot(delta_Tlist, mean_rmseL[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
plt.plot(delta_Tlist, mean_rmseL[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
plt.plot(delta_Tlist, mean_rmseL[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
plt.xlabel("dleta_T")
plt.ylabel("value")
plt.grid()
plt.legend()
plt.title("RMSE")
# plt.savefig(png_path, dpi=300, bbox_inches='tight') 


plt.figure()
# png_path = os.path.join('/home/admin123/SATData/Run', formatted_time, "MAE.png")
plt.plot(delta_Tlist, mean_maeL[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
plt.plot(delta_Tlist, mean_maeL[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
plt.plot(delta_Tlist, mean_maeL[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
plt.xlabel("dleta_T")
plt.ylabel("value")
plt.title("MAE")
plt.grid()
plt.legend()
# plt.savefig(png_path, dpi=300, bbox_inches='tight')


plt.figure()
# png_path = os.path.join('/home/admin123/SATData/Run', formatted_time, "R2.png")
plt.plot(delta_Tlist, mean_r2L[:,0], marker='o', linestyle='-', label='elv_angle')  # 线条和点
plt.plot(delta_Tlist, mean_r2L[:,1], marker='o', linestyle='-', label='shoulder_elv')  # 线条和点
plt.plot(delta_Tlist, mean_r2L[:,2], marker='o', linestyle='-', label='elbow_flexion')  # 线条和点
plt.xlabel("dleta_T")
plt.ylabel("value")
plt.title("R2")
plt.grid()
plt.legend()
# plt.savefig(png_path, dpi=300, bbox_inches='tight')