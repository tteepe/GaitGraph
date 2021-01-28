cd ../src || exit

python3 train.py casia-b ../data/casia-b_pose_train_valid.csv \
                 --valid_data_path ../data/casia-b_pose_test.csv \
                 --batch_size 128 \
                 --batch_size_validation 256 \
                 --embedding_layer_size 128 \
                 --epochs 500 \
                 --learning_rate 1e-4\
                 --weight_decay 1e-4 \
                 --temp 0.01 \
                 --shuffle \
                 --exp_name shuffle
