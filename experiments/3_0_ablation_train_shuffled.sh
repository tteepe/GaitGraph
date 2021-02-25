cd ../src || exit

python3 train.py casia-b ../data/casia-b_pose_train_valid.csv \
                 --valid_data_path ../data/casia-b_pose_test.csv \
                 --batch_size 128 \
                 --batch_size_validation 256 \
                 --embedding_layer_size 128 \
                 --epochs 300 \
                 --learning_rate 1e-2 \
                 --temp 0.01 \
                 --network_name resgcn-n39-r8 \
                 --shuffle \
                 --exp_name shuffle
