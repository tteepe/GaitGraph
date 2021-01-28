cd ../src || exit

python3 train.py casia-b ../data/casia-b_pose_train_valid.csv \
                 --valid_data_path ../data/casia-b_pose_test.csv \
                 --batch_size 128 \
                 --batch_size_validation 256 \
                 --learning_rate 1e-6 \
                 --embedding_layer_size 128 \
                 --weight_path ../save/supcon_casia-b_models/<INSERT NAME>/ckpt_epoch_best.pth \
                 --epochs 50 \
                 --temp 0.01 \
                 --weight_decay 1e-5 \
                 --shuffle \
                 --exp_name shuffle_fine

