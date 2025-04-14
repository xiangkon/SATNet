from utils.train_func import train_func
from datetime import datetime


rmse, mae, r2 = train_func(epochs=5, train_num=5, formatted_time = datetime.now().strftime("%m-%d-%H:%M:%S"))