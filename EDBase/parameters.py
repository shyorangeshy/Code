import argparse


parser = argparse.ArgumentParser()
# add_arguments
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

# ---- model setting ----
parser.add_argument('-pretrain_model_name',type=str, default='bert-base-cased')
parser.add_argument('-features_size', type=int, default=256)
parser.add_argument('-MODE',type=int,default=1)

# ---- file path ----
parser.add_argument('-data_dir',type=str, default='')
parser.add_argument('-train_file',type=str, default='./data/train.json')
parser.add_argument('-dev_file',type=str, default='./data/dev.json')
parser.add_argument('-test_file',type=str, default='./data/test.json')
parser.add_argument('-label_file',type=str, default='./data/event_labels.txt')


# ---- schema setting ----
parser.add_argument('-do_train',type=boolean_string, default=True)
parser.add_argument('-do_dev',type=boolean_string, default=False)
parser.add_argument('-do_predict',type=boolean_string, default=False)
parser.add_argument('-save_result',type=boolean_string, default=False)

# ---- task setting ----
parser.add_argument(
    '-event_nums',type=int,default=34
)
# --- env setting ----
parser.add_argument('-gpu',type=int,default=0)
parser.add_argument('-seed',type=int,default=-1)
parser.add_argument('-log_file',type=str,default='./logs/log.txt')
parser.add_argument('-save_dir',type=str,default='./checkpoints')
parser.add_argument('-base_dir',type=str,default='./checkpoints',help="the base model path")
parser.add_argument('-prefix',type=str,default='bert')
parser.add_argument('-suffix',type=str,default='.base')
parser.add_argument('-fp16',type=boolean_string, default=False)
parser.add_argument('-fptype',type=str, default='O1')

# ---- training setting ----
parser.add_argument('-train_batch_size', type=int, default=16)
parser.add_argument('-dev_batch_size', type=int, default=1)
parser.add_argument('-test_batch_size', type=int, default=1)
parser.add_argument('-max_seq_length', type=int, default=128)
parser.add_argument('-epoch_nums', type=int, default=8)
parser.add_argument('-warmup_rate', type=float, default=0.1)
parser.add_argument('-lr', type=float, default=2e-5)
parser.add_argument('-save_step', type=int, default=1000)

# continue是为了读取schelar optimizer
parser.add_argument('-continue_checkpoint',type=boolean_string, default=False)
parser.add_argument('-checkpoint',type=int, default=0)
parser.add_argument('-cur_batch',type=int,default=0)
parser.add_argument('-load_checkpoint',type=boolean_string, default=False)


# ---- context_model setting ----
parser.add_argument('-performance_file',type=str,default='performance/CotRoberta.txt')

args = parser.parse_args()
