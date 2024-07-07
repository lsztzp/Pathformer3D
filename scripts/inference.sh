#python main.py \
#  --config=/data/qmengyu/02-Results/01-ScanPath/360_logs/11-29-new-lr=5e-6/config.py\
#  --work_dir="/data/qmengyu/02-Results/01-ScanPath/360_logs/11-29-new-lr=5e-6/"\
#  --device="cuda:0"\
#  --wo_train\
#  --wo_score\
#  --options "val_batch_size=1" "reload_path=/data/qmengyu/02-Results/01-ScanPath/360_logs/11-29-new-lr=5e-6/checkpoint/ep110.pth.tar"

#python main.py \
#  --config=/data/qmengyu/02-Results/01-ScanPath/360_logs/12-02-lr=lr=5e-5_no_warm_up/config.py\
#  --work_dir="/data/qmengyu/02-Results/01-ScanPath/360_logs/12-02-lr=lr=5e-5_no_warm_up/"\
#  --device="cuda:1"\
#  --wo_train\
#  --options "val_batch_size=1" "reload_path=/data/qmengyu/02-Results/01-ScanPath/360_logs/12-02-lr=lr=5e-5_no_warm_up/checkpoint/ep11.pth.tar"
#

#python main.py \
#  --config=/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-two_datasets/config.py\
#  --work_dir="/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-two_datasets/"\
#  --device="cuda:2"\
#  --wo_train\
#  --options "val_batch_size=1" "reload_path=/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-two_datasets/checkpoint/ep14.pth.tar"

#python main.py \
#  --config=/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-salient360_new_lr/config.py\
#  --work_dir="/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-salient360_new_lr/"\
#  --device="cuda:2"\
#  --wo_train\
#  --options "val_batch_size=1" "reload_path=/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-salient360_new_lr/checkpoint/checkpoint.pth.tar"

#python main.py \
#  --config=/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-merge_train/config.py\
#  --work_dir="/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-merge_train/"\
#  --device="cuda:2"\
#  --wo_train\
#  --options "val_batch_size=1" "reload_path=/data/lyt/02-Results/01-ScanPath/360_test_logs/01-04-merge_train/checkpoint/ep16.pth.tar"

python main.py \
  --config=/data/lyt/02-Results/01-ScanPath/360_ablation_logs/01-28-enc_dec_3/config.py\
  --work_dir="/data/lyt/02-Results/01-ScanPath/360_test_logs/01-05-num_gauss=8/"\
  --device="cuda:3"\
  --wo_train\
  --options "val_batch_size=1" "reload_path=/data/lyt/02-Results/01-ScanPath/360_test_logs/01-05-num_gauss=8/checkpoint/ep10.pth.tar"

