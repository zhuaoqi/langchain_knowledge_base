'''
Descripttion: 
Author: zhuaoqi
Date: 2025-03-28 10:02:43
LastEditors: zhuaoqi
LastEditTime: 2025-04-18 09:56:13
'''
from modelscope import snapshot_download

smodel_dir = snapshot_download(model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",cache_dir="D:/models")