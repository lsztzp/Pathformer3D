python main.py --work_dir="enc_dec_3" --device="cuda:2" --options "enc_n_layers=3" "dec_n_layers=3"

#num_gausss=(6 7)
#for num_gauss in "${num_gausss[@]}"
#do
#  python main.py --work_dir="num_gauss=$num_gauss" --device="cuda:0" --options "num_gauss=$num_gauss" "num_epoch=40"
#done

#python main.py --work_dir="merge_train" --device="cuda:2" --options "train_dataset=merge"

#d_models=(64  128  192 256)
#
#for d_model in "${d_models[@]}"
#do
#  python main.py --work_dir="d_model=$d_model " --device="cuda:2" --options "d_model=$d_model"
#done

#feature_dims=(64  192  576 960)
#
#for feature_dim in "${feature_dims[@]}"
#do
#  python main.py --work_dir="feature_dim=$feature_dim " --device="cuda:2" --options "feature_dim=$feature_dim"
#done
#

#python main.py --work_dir="salient360-train_dataset" --device="cuda:2" --config="config.py" --options "train_dataset=salient360"



#num_gausss=(1 3 5 7 10 15)
#
#for num in "${num_gausss[@]}"
#do
#  python main.py --work_dir="num_gauss=$num" --device="cuda:2" --options "num_gausss=$num"
#done
#
#
#
#dropouts=(0. 0.3 0.5 0.7 0.9)
#
#for dropout in "${dropouts[@]}"
#do
#  python main.py --work_dir="dropout=$dropout" --device="cuda:2" --options "dropout=$dropout"
#done
#

