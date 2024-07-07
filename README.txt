-- wo_train: 不进行训练，需要指定 reload_path 加载历史模型
-- wo_inference: 默认在几个测试数据集上推理模型结果，并存在 workdir 下指定文件夹
-- wo_score： 默认读取workdir 下几个测试数据集上模型推理结果计算数据集模型分数，保存在 workdir 下 txt 文件中。


# 模型的加载：
1. 指定 reload_path = ""
2. 默认加载 work_dir 中最后模型（用于断点续训），没有则不加载 [问题：为了记录时间，加入了日期，如果不在同一天训练完，就会有问题]



用法：

1. 加载模型，获取模型运行在四个数据集上的预测结果，并计算数据集的分数（添加--wo_score可跳过计算分数）
例如：
python main.py
--config=/data/qmengyu/02-Results/01-ScanPath/360_logs/11-29-new-lr=5e-6/config.py # 指定config文件
--work_dir="/data/qmengyu/02-Results/01-ScanPath/360_logs/11-29-new-lr=5e-6/" # 指定工作路径，数据集预测结果会保存在该目录下
--device="cuda:0"
--wo_train
--options
"val_batch_size=1"
"reload_path=/data/qmengyu/02-Results/01-ScanPath/360_logs/11-29-new-lr=5e-6/checkpoint/ep110.pth.tar" # 加载模型的路径

运行结果：
四个数据集的预测结果会保存在 work_dir 下的 seq_results_best_model，测试指标分数保存在这个路径底下的 txt 文件中





# TODO



