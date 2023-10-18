version=$(date "+%Y.%m.%d_%H.%M.%S")
MODEL='baseberta'

if [ ! -d "../log/" ];then
    mkdir ../log
else
    echo "log文件夹已经存在"
fi

NUM=1
for lr in 1e-05
do
    for seed in 500
    do
        FILENAME=${lr}_${seed}_${version}
        python3 ../train.py \
            -pretrain_model_name='bert-base-cased' \
            -data_dir='' \
            -train_file='../data/data_train_2.json' \
            -dev_file='../data/data_dev_2.json' \
            -test_file='../data/data_dev_2.json' \
            -label_file='../data/event_labels.txt' \
            -max_seq_length=512 \
            -log_file='../log/'$FILENAME'.log' \
            -do_train=True \
            -do_predict=False \
            -train_batch_size=8 \
            -dev_batch_size=8 \
            -test_batch_size=8 \
            -epoch_nums=16 \
            -gpu=3 \
            -save_result=False \
            -continue_checkpoint=False \
            -lr=${lr} \
            -checkpoint=10 \
            -cur_batch=19000 \
            -load_checkpoint=False\
            -save_dir='../checkpoints/'$FILENAME \
            -performance_file='../performance/'$FILENAME'.eval' \
            -save_step=1000 \
            -seed=${seed} \
            -fp16=True
    done
done

