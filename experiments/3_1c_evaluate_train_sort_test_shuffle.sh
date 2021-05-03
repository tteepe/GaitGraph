cd ../src || exit

python3 evaluate.py casia-b \
                 ../save/casia-b_models/<INSERT NAME>/ckpt_epoch_best.pth \
                 ../data/casia-b_pose_test.csv \
                 --network_name resgcn-n39-r8 \
                 --shuffle
