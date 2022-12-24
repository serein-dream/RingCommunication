python train.py --data_dir ./data/ \
                --dataset mnist \
                --device cpu \
                --world_size 3 \
                --batch_size 150 \
                --lr 0.01 \
                --lr_gamma 0.9 \
                --epoch 200 \
                --backend gloo
