import sys

sys.path.append("..")


import numpy as np
import sklearn
import os
from tqdm import tqdm
from math import ceil
import shutil
import random
import time
import json
from logger import Logger, EvalLogger
import fitlog
import torch
# print('\nmmmmdevice',torch.cuda.is_available())
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertModel,BertTokenizer,AlbertTokenizer,RobertaTokenizer,XLNetTokenizer,BertTokenizerFast
from transformers import TrainingArguments
from transformers import Trainer
# load:data\model\args
from processing import load_dataset
from model.BERTED import myBert
from parameters import args
# model Performance Testing
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score



fitlog.set_log_dir("../logs/")
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)  # 自动记录超参，超参必须作为全局变量

torch.set_printoptions(edgeitems=10)
myLogger = Logger(name=args.pretrain_model_name+'.log', save_file=args.log_file)
evalLogger = EvalLogger(args.performance_file)
logger = myLogger.get_logger()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
# print device
#device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


#mycode_start
def mycollect(batch):
    #print('\nppp',batch)
    batch_re={}
    c=[]
    p=[]
    uy=torch.tensor([1,3])
    a_uy=type(uy)
    f=[ i.shape[-1] if isinstance(i,a_uy) and i.shape !=torch.Size([]) else -1 for i in batch[0].values()]
    names = locals()
    for i in range(len(batch[0])):
        c.append(True)
        names['x%s' % i] = []
    for j in range(len(batch)):
        i=batch[j]
        d=[]
        count=0
        for a,b in i.items():
            if isinstance(b,a_uy) :
                if b.shape !=torch.Size([]):
                    d.append(b.shape[-1])
                else:
                    d.append(-1)
            else:
                d.append(-1)
            if j==0:
                p.append(a)
                names['x%s' % count]=[b]
            else:
                names['x%s' % count].append(b)
            count=count+1
        
        c=[ False if f[i]==-1 else  c[i] and f[i]==d[i] for i in range(len(f))]
        f=d
    for i in range(len(c)):
        if c[i]==True and p[i]!='com_input_ids' and p[i]!='com_attention_mask' and p[i]!='com_token_type_ids' and p[i]!='dep_tree':
            z=names['x%s' % i]
            batch_re[p[i]]=torch.cat([torch.unsqueeze(i,dim=0) for i in z],0)
        else:
            if p[i]=='act_lens':
                batch_re[p[i]]=torch.tensor([len(names['x%s' % i])])
            elif p[i]=='common_sense':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='sentence':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='com_input_ids':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='com_attention_mask':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='com_token_type_ids':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='dep_tree':
                batch_re[p[i]]=names['x%s' % i]
            elif p[i]=='pos_info':
                batch_re[p[i]]=names['x%s' % i]
            
    return batch_re

#mycode_end
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# torch.backends.cudnn.enabled = False


if args.seed > -1:
    seed_torch(args.seed)


def train(model):
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    # print logger
    evalLogger.write('\n\n---- train stage ----\n')
    print('---- train stage ----')
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    evalLogger.write(localtime + '\n')

    best_res = tuple()

    # processing data
    tokenizer = select_tokenizer(args)

    train_dataset = load_dataset(args, tokenizer, "train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler = train_sampler, batch_size = args.train_batch_size,collate_fn=mycollect)

    dev_dataset = load_dataset(args, tokenizer, "dev")
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler = dev_sampler, batch_size = args.dev_batch_size,collate_fn=mycollect)

    test_dataset = load_dataset(args, tokenizer, "test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler = test_sampler, batch_size = args.test_batch_size,collate_fn=mycollect)


    batch_nums = len(train_dataloader)
    print("batch_nums", batch_nums)
    total_train_steps = batch_nums * args.epoch_nums

    logger.debug('total_train_steps:%d', total_train_steps)
    print('total_train_steps: ', total_train_steps)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_steps = ceil(total_train_steps * args.warmup_rate)

    logger.debug('warmup_step:%d', warmup_steps)
    print('warm_up_steps: ', warmup_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps
    )

    # apex
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fptype)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # 读取断点 optimizer、scheduler
    if args.continue_checkpoint:
        checkpoint_dir = args.save_dir + "/checkpoint-" + str(args.checkpoint) + '-' + str(args.cur_batch)
        logger.debug('Load checkpoint:optimizer.pt & scheduler.pt from %s', checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, "optimizer.pt")):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(checkpoint_dir, "scheduler.pt")))
        checkpoint = args.checkpoint + 1
    else:
        checkpoint = 0

    best_f1 = 0.0
    best_epoch_num = 0
    global_step = 0
    
    logging_loss = 0
    for epoch in range(checkpoint, args.epoch_nums):
        total_loss = 0
        model.train()
        localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(localtime)
        print('epoch {} / {}'.format(epoch + 1, args.epoch_nums))
        logger.debug('epoch {} / {}'.format(epoch + 1, args.epoch_nums))
        evalLogger.write('epoch-' + str(epoch + 1) + '\n')

        for step, batch in tqdm(enumerate(train_dataloader),desc = "Training", total = batch_nums, ncols=50):
            model.zero_grad()
            optimizer.zero_grad()
            global_step += 1
            
            inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "token_type_ids": batch["token_type_ids"].to(device),
                    "labels": batch["labels"].to(device),
                    "device":device,
                    "com_input_ids": batch["com_input_ids"],
                    "com_attention_mask": batch["com_attention_mask"],
                    "com_token_type_ids": batch["com_token_type_ids"],
                    "merge_index": batch["merge_index"]
                }
            #'com_input_ids''com_token_type_ids''com_token_type_ids'
            # pdb.set_trace()
            outputs = model(**inputs)
            batch_loss = outputs[0]
            total_loss += batch_loss.item()

            
            if args.fp16:
                with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                batch_loss.backward()
            optimizer.step()
            scheduler.step()
            if global_step % args.save_step == 0 or ((2*(step+1))%batch_nums == 0 and step+1!=batch_nums):
                new_eva, ori_eva = dev(model, dev_dataloader, 'dev',
                                       str(epoch) + '-' + str(global_step))
                fitlog.add_loss(total_loss - logging_loss, name='Train_loss', step=global_step)
                fitlog.add_metric(ori_eva[-1], name="dev_origin F1", step=global_step)
                fitlog.add_metric(new_eva[-1], name="dev_new F1", step=global_step)
                logging_loss = total_loss
                if ori_eva[-1] > best_f1 and epoch>5:

                    best_f1 = ori_eva[-1]
                    best_epoch_num = epoch

                    # output_dir = args.save_dir + '/best-checkpoint/checkpoint-'+str(epoch)+'-'+str(global_step)
                    # # 删除历史模型
                    # filelist = os.listdir(args.save_dir+'/best-checkpoint')
                    # for f in filelist:
                    #     filepath = os.path.join(args.save_dir, f)
                    #     if os.path.isdir(filepath):
                    #         shutil.rmtree(filepath,True)

                    # # 保存模型
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    # model_to_save = (model.module if hasattr(model, "module") else model)
                    # model_to_save.save_pretrained(output_dir)
                    # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    # logger.debug("Saving model checkpoint to %s", output_dir)
                    # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    # logger.debug("Saving optimizer and scheduler states to %s", output_dir)

                    save_model(model, optimizer, scheduler, epoch, global_step, 'best')

                    new_eva, ori_eva = dev(model, test_dataloader, 'test',
                                           str(epoch) + '-' + str(global_step))
                    fitlog.add_metric(ori_eva[-1], name="test_origin F1", step=global_step)
                    fitlog.add_metric(new_eva[-1], name="test_new F1", step=global_step)
                    best_res = (str(epoch) + '-' + str(global_step), [ori_eva, new_eva])
                else:
                    new_eva, ori_eva = dev(model, test_dataloader, 'test',
                                           str(epoch) + '-' + str(global_step))
                    fitlog.add_metric(ori_eva[-1], name="test_origin F1", step=global_step)
                    fitlog.add_metric(new_eva[-1], name="test_new F1", step=global_step)
                model.train()
        each_sentences_loss = total_loss
        print('---- train loss:{:.5f}'.format(each_sentences_loss))
        evalLogger.write('---- train loss:{:.5f}\n'.format(each_sentences_loss))

        new_eva, ori_eva = dev(model, dev_dataloader, 'dev',
                               str(epoch) + '-' + str(global_step))
        if ori_eva[-1] > best_f1 and epoch>5:

            best_f1 = ori_eva[-1]
            best_epoch_num = epoch

            # output_dir = args.save_dir + '/best-checkpoint/checkpoint-'+str(epoch)+'-'+str(global_step)

            # # 删除历史模型
            # filelist = os.listdir(args.save_dir+'/best-checkpoint')
            # for f in filelist:
            #     filepath = os.path.join(args.save_dir, f)
            #     if os.path.isdir(filepath):
            #         shutil.rmtree(filepath,True)

            # # 保存模型
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # model_to_save = (model.module if hasattr(model, "module") else model)
            # model_to_save.save_pretrained(output_dir)
            # torch.save(args, os.path.join(output_dir, "training_args.bin"))
            # logger.debug("Saving model checkpoint to %s", output_dir)
            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            # logger.debug("Saving optimizer and scheduler states to %s", output_dir)

            save_model(model, optimizer, scheduler, epoch, global_step, 'best')

            new_eva, ori_eva = dev(model, test_dataloader, 'test',
                                   str(epoch) + '-' + str(global_step))
            best_res = (str(epoch) + '-' + str(global_step), [ori_eva, new_eva])
        else:
            new_eva, ori_eva = dev(model, test_dataloader, 'test',
                                   str(epoch) + '-' + str(global_step))
        # save epoch model
        save_model(model, optimizer, scheduler, epoch)

    fitlog.add_best_metric({"Dev": {"F1": best_f1, "Epoch": best_epoch_num}})
    fitlog.finish()

    evalLogger.write('!!!!!!!!!\nBEST PERFORMER' + str(best_res) + '\n\n')
    print('!!!!!!!!!\nBEST PERFORMER' + str(best_res) + '\n\n')


def dev(model, dataloader, prefix='', suffix=''):
    print('\n')
    print('--------------- ' + prefix + ':' + suffix + '---------------')
    # logger.debug('--------------- ' + prefix + ':' + suffix + '---------------')
    # evalLogger.write('---- ' + prefix + ':' + suffix + ' ----')
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(localtime)

    model.eval()
    total_gt_labels = list()
    total_pt_labels = list()
    total_merge_index = list()
    batch_nums = len(dataloader)
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
            
            inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "token_type_ids": batch["token_type_ids"].to(device),
                    "labels": batch["labels"].to(device),
                    "device":device,
                    "com_input_ids": batch["com_input_ids"],
                    "com_attention_mask": batch["com_attention_mask"],
                    "com_token_type_ids": batch["com_token_type_ids"],
                    "merge_index": batch["merge_index"]
                }
            
            with torch.no_grad():
                outputs = model(**inputs)
                batch_loss = outputs[0]
                total_loss += batch_loss.item()
      
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                gt_labels = inputs["labels"].detach().cpu().numpy()
                merge_index = batch['merge_index'].detach().cpu().numpy()
                attention_mask = inputs["attention_mask"].detach().cpu().numpy()
                act_lens = batch["act_lens"].detach().cpu().numpy()


                # 原版是用attention mask取得，由于cls和sep也不需要计算进去，所以这么取的gt_labels >0的结点
                distill_boolean = gt_labels >= 0
                total_gt_labels.extend(gt_labels[distill_boolean])
                # logits.shape: batch_size * max_seq_len * hidden_size
                predicts = np.argmax(logits, axis=-1)
                total_pt_labels.extend(predicts[distill_boolean])

                # process original labels
                total_merge_index.extend(merge_index[distill_boolean])

    print('total_bpe_words:', len(total_pt_labels))

    if args.save_result:
        with open('../log/logits_res', 'w', encoding='utf8') as fout:
            fout.write(str(total_pt_labels[:2000]) + '\n')
            fout.write(str(total_gt_labels[:2000]) + '\n')
            fout.write(str(total_merge_index[:2000]) + '\n')

    print('pt_labels', len(total_pt_labels))
    assert len(total_pt_labels) == len(total_gt_labels)
    assert len(total_pt_labels) == len(total_merge_index)

    new_p, new_r, new_f1 = evaluate_score(total_gt_labels, total_pt_labels, prefix='bpe_labels')

    ori_gt_labels, ori_pt_labels = merge_strategy(total_gt_labels, total_pt_labels, total_merge_index)
    # logger.debug('Reconstruct ori_label success!')
    # print('Reconstruct ori_label success!')
    ori_p, ori_r, ori_f1 = evaluate_score(ori_gt_labels, ori_pt_labels, prefix='ori_labels')

    each_sentences_loss = total_loss
    print(prefix + ' loss:{:.5f}'.format(each_sentences_loss))
    evalLogger.write('---- {} loss:{:.5f}\n'.format(prefix, each_sentences_loss))

    return [[new_p, new_r, new_f1], [ori_p, ori_r, ori_f1]]


def merge_strategy(gt_labels, pt_labels, merge_index):
    ori_gt_labels = list()
    ori_pt_labels = list()
    flag = -1
    
    # 取被切分成的几个部分的token的最左边的一份
    for i in range(len(merge_index)):
        if merge_index[i] < 0 or flag == merge_index[i]: # 注意这里是merge[i]而不是i
            continue
        else:
            ori_gt_labels.append(gt_labels[i])
            ori_pt_labels.append(pt_labels[i])
            flag = merge_index[i]
    
    assert len(ori_gt_labels) == len(ori_pt_labels)
    return ori_gt_labels, ori_pt_labels


def save_model(model, optimizer, scheduler, epoch, global_step=None, typ='epoch'):
    if typ == 'best':
        cor_dir = args.save_dir + '/best-checkpoint'
        output_dir = cor_dir + '/checkpoint-' + str(epoch) + '-' + str(global_step)
    elif typ == 'epoch':
        cor_dir = args.save_dir + '/epoch-checkpoint'
        output_dir = cor_dir + '/checkpoint-' + str(epoch)

    if not os.path.exists(cor_dir):
        os.makedirs(cor_dir)


    # 删除历史模型
    filelist = os.listdir(cor_dir)
    for f in filelist:
        filepath = os.path.join(cor_dir, f)
        if os.path.isdir(filepath):
            shutil.rmtree(filepath, True)

    # 保存模型
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = (model.module if hasattr(model, "module") else model)
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.debug("Saving model checkpoint to %s", output_dir)
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.debug("Saving optimizer and scheduler states to %s", output_dir)

    arg_dict = args.__dict__
    with open(os.path.join(output_dir,"args.json"),'w',encoding='utf8') as f:
        json.dump(arg_dict,f,indent=2,ensure_ascii=False)


def evaluate_score(gt_labels, pt_labels, prefix=''):
    label_types = [i for i in range(1, args.event_nums)]
    find_trigger = np.count_nonzero(pt_labels)
    gt_trigger = np.count_nonzero(gt_labels)
    evalLogger.write('gt_trigger_nums: {} \t find_trigger_nums: {}\n'.format(gt_trigger, find_trigger))
    logger.debug('gt_trigger_nums: {} \t find_trigger_nums: {}'.format(gt_trigger, find_trigger))
    # print('gt_triggers_num: ',gt_trigger)
    # print('find_triggers_num: ',find_trigger)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_labels,
        pt_labels,
        labels=label_types,
        average='micro'
    )
    print(prefix + '\t', end='')
    print("precision {:g}, recall {:g}, f1 {:g}".format(precision, recall, f1))
    evalLogger.write(prefix + '\t')
    evalLogger.write("precision {:g}, recall {:g}, f1 {:g}\n".format(precision, recall, f1))
    logger.debug(prefix)
    logger.debug("precision {:g}, recall {:g}, f1 {:g}".format(precision, recall, f1))
    return precision, recall, f1



def select_tokenizer(args):
    if "albert" in args.pretrain_model_name:
        return AlbertTokenizer.from_pretrained(args.pretrain_model_name)
    elif "roberta" in args.pretrain_model_name:
        return RobertaTokenizer.from_pretrained(args.pretrain_model_name)
    elif "bert" in args.pretrain_model_name:
        return BertTokenizerFast.from_pretrained(args.pretrain_model_name)
    elif "xlnet" in args.pretrain_model_name:
        return XLNetTokenizer.from_pretrained(args.pretrain_model_name)




def main():

    logger.info(args)
    # checkpoint_dir = args.save_dir + "/checkpoint-" + str(args.checkpoint) + '-' + str(args.cur_batch)
    checkpoint_dir = args.base_dir + "/checkpoint-" + str(args.checkpoint) + '-' + str(args.cur_batch)
    # print('check_point_dir',checkpoint_dir)
    model = myBert.from_pretrained( \
        args.pretrain_model_name if not args.load_checkpoint else checkpoint_dir, \
        num_labels=args.event_nums).to(device)
    # print(dir(model))
    if args.do_train:
        train(model)


if __name__ == '__main__':
    main()
