version=$(date "+%Y.%m.%d_%H.%M.%S")
MODEL='baseberta'

if [ ! -d "../log/" ];then

    mkdir ../log
else

    echo  "log文件夹已经存在\n"
fi

if [ ! -d "../trash/" ];then

    mkdir ../trash
else

    echo  "trash文件夹已经存在\n"
fi

if [ ! -d "../logs/" ];then

    mkdir ../logs
else

    echo  "logs文件夹已经存在\n"
fi

if [ ! -d "../checkpoints/" ];then

    mkdir ../checkpoints
else

    echo  "checkpoints文件夹已经存在\n"
fi

if [ ! -d "../performance/" ];then

    mkdir ../performance
else

    echo  "performance文件夹已经存在\n"
fi

echo "Do you want drill [y/n]?"

read yn    
if [ "$yn" = "y" ]||[ "$yn" = "Y" ];then
    echo  "script is running for drill...\n"
    echo  "related file will be in trash...\n"
    log_dir_o='../trash/'
    logs_dir_o='../trash/'
    check_dir_o='../trash/'
    performance_dir='../trash/'
    pin='1.json'
    change_batch_size=1
    change_dev_size=1
    change_epoch=1
else
    echo  "script is running for training...\n"
    log_dir_o='../log/'
    logs_dir_o='../logs/'
    check_dir_o='../checkpoints/'
    performance_dir='../performance/'
    pin='_newq01.json'
    change_batch_size=8
    change_dev_size=8
    change_epoch=16
fi


for lr in 1e-05
do
    for seed in 521
    do
        FILENAME=${lr}_${seed}_${version}
        python3 ../train.py \
            -pretrain_model_name='bert-large-cased' \
            -data_dir='' \
            -train_file='../data/train'${pin} \
            -dev_file='../data/dev'${pin} \
            -test_file='../data/test'${pin} \
            -label_file='../data/event_labels.txt' \
            -max_seq_length=128 \
            -log_file=${log_dir_o}${FILENAME}'.log' \
            -do_train=True \
            -do_predict=False \
            -train_batch_size=$change_batch_size \
            -dev_batch_size=$change_dev_size \
            -test_batch_size=$change_dev_size \
            -epoch_nums=$change_epoch \
            -gpu=6  \
            -save_result=False \
            -continue_checkpoint=False \
            -lr=${lr} \
            -checkpoint=10 \
            -cur_batch=19000 \
            -load_checkpoint=False\
            -save_dir=${check_dir_o}${FILENAME} \
            -performance_file=${performance_dir}${FILENAME}'.eval' \
            -save_step=1000 \
            -seed=${seed} \
            -fp16=True 
    done
#      -logs_file=${logs_dir_o}
  # let i++
done


