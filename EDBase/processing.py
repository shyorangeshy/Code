import torch
import os
from torch._C import FunctionSchema
from torch.utils.data import TensorDataset
import stanza
from tqdm import tqdm
from itertools import islice
import pdb
import json
import nltk
import random
class ACEExample(object):
    def __init__(self, context, labels,com,com_num,choice) -> None:
        self.context = context
        self.labels = labels
        self.com=com
        self.com_num=com_num
        self.choice=choice
class ACEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, merge_index, act_lens,com_encodings) -> None:
        self.encodings = encodings
        self.labels = labels
        self.merge_index = merge_index
        self.act_lens = act_lens
        self.com_encodings=com_encodings
        # self.com_sen_num=com_sen_num
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['merge_index'] = torch.tensor(self.merge_index[idx])
        item['act_lens'] = torch.tensor(self.act_lens[idx])
        for key, val in self.com_encodings.items():
             item[key]=val[idx]
        return item
    def __len__(self):
        return len(self.labels)

class ACEProcessor():
    def read_example(self, file_name):
        """
        @description  : read source data and pack each context with its label
        ---------
        @param  :
            file_name: the path/name of source data
        -------
        @Returns  :
            examples：[ACEExample(context1, labels1),ACEExample(context2, labels2),...]
        -------
        """
        
        
        with open(file_name,'r',encoding="utf8") as fin:
            examples = []
            _corpus = json.load(fin)
            for traget in _corpus:
                context, label_str,com,com_num,choice = traget[0], traget[1],traget[-3],traget[-2],traget[-1]
                labels = label_str.split(" ")
                assert len(context.split(" ")) == len(labels)
                example = ACEExample(context, labels,com,com_num,choice)
                examples.append(example)

        return examples

    def convert_examples_to_features(self, tokenizer, examples, label_to_id, max_seq_length,args):
        """
        @description  :
            convert the origin examples to pretrain model's(or other) input forms
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        rel_all_B=[" is capable of "," causes "," is a "," is a manner of "," is motivated by "," receives action of "]
        encodings = {}
        encodings["input_ids"] = []
        encodings["attention_mask"] = []
        encodings["token_type_ids"] = []
        labels = []
        act_lens = []
        merge_index = []
#mycode_start
        com_encodings = {}
        com_encodings["com_input_ids"] = []
        com_encodings["com_attention_mask"] = []
        com_encodings["com_token_type_ids"] = []
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
#mycode_end
        for example in tqdm(examples, total = len(examples), desc = "ACE processing"):
            # *** same as encode_plus, encode just return ids, encode_plus return all information
            # pdb.set_trace()
            # 
# mycode_start  
            com_input_ids=[]
            com_attention_mask=[]
            com_token_type_ids=[]
            com_input_ids1=[]
            com_attention_mask1=[]
            com_token_type_ids1=[]
            #pos_select=['FW','JJ','JJR','JJS','NN','NNS','NNP','NNPS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']
            context=example.context.split(" ")
            context_len=len(context)
            for k in range(context_len):
                word_know=example.com[k].split(" ")
                word_know_num=example.com_num[k].split(" ")
                #word_know_choice=int(example.choice.split(" ")[k][1:-1])
                word_know_fin="[CLS]"
                len_word_know=len(word_know)
                if len_word_know==1:
                    word_know_rand=0
                else:
                    word_know_rand=random.randint(1,len_word_know-1)
                if len_word_know==1 or word_know_rand==0:
                    word_know_fin=word_know_fin+" "+word_know[0]
                else:  
                    word_know_fin=word_know_fin+" "+word_know[0]+rel_all_B[int(word_know_num[word_know_rand])]+word_know[word_know_rand]+ " [SEP]"
                tokenized_inputs1 = tokenizer(
                    word_know_fin.split(" "), # transformer有Tokenizer 和 TokenizerFast 两类，其中Fast在输入分词好的结果时，会返回一个
                    add_special_tokens=True,
                    padding = 'max_length', 
                    truncation=True,
                    max_length=16,
                    is_split_into_words = True)
            # ***
                com_input_ids.append(tokenized_inputs1["input_ids"])
                com_attention_mask.append(tokenized_inputs1["attention_mask"])
                com_token_type_ids.append(tokenized_inputs1["token_type_ids"])
                
            
#mycode_end
            tokenized_inputs = tokenizer(
                example.context.split(" "), # transformer有Tokenizer 和 TokenizerFast 两类，其中Fast在输入分词好的结果时，会返回一个
                add_special_tokens=True,
                padding = 'max_length', 
                truncation=True,
                max_length=max_seq_length,
                is_split_into_words = True)

            label = example.labels
            word_ids = tokenized_inputs.word_ids() # 为了对齐token
            # previous_word_idx = None
            label_ids = []
            #pos_expadding=[]
            real_len = 0
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    #pos_expadding.append(42)
                # We set the label for the first token of each word.
                # elif word_idx != previous_word_idx:
                #     label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    real_len += 1
                    label_ids.append(label_to_id[label[word_idx]])
            merge = [-1 if i == None else i for i in word_ids]
            for i in merge:
                 if i!=-1:
                    com_input_ids1.append(com_input_ids[i])
                    com_attention_mask1.append(com_attention_mask[i])
                    com_token_type_ids1.append(com_token_type_ids[i])

            encodings["input_ids"].append(tokenized_inputs["input_ids"])
            encodings["attention_mask"].append(tokenized_inputs["attention_mask"])
            encodings["token_type_ids"].append(tokenized_inputs["token_type_ids"])
            com_encodings["com_input_ids"].append(com_input_ids1)
            com_encodings["com_attention_mask"].append(com_attention_mask1)
            com_encodings["com_token_type_ids"].append(com_token_type_ids1)
            labels.append(label_ids)
            act_lens.append(real_len)
            merge_index.append(merge)
        dataset = ACEDataset(encodings, labels, merge_index, act_lens,com_encodings)
        return dataset

def load_label_fix(filePath, data_type='txt'):
	if data_type == 'json':
		with open(filePath, 'r', encoding='utf8') as fin:
			dat = json.load(fin)
		return dat
	elif data_type == 'txt':
		labeldict = {}
		with open(filePath, 'r', encoding='utf8') as fin:
			for ind, line in enumerate(fin.readlines()):
                # key: event type; value: id
				labeldict[line.strip()] = ind
		return labeldict


def load_dataset(args, tokenizer, mode):
    processor = ACEProcessor()
    
    if mode == "train":
        file_path = os.path.join(args.data_dir, args.train_file)
    elif mode == "dev" or mode == "valid":
        file_path = os.path.join(args.data_dir, args.dev_file)
    else:
        file_path = os.path.join(args.data_dir, args.test_file)
    label_path = os.path.join(args.data_dir, args.label_file)

    examples = processor.read_example(file_path)
    label_to_id = load_label_fix(label_path)
    return processor.convert_examples_to_features(tokenizer, examples, label_to_id, args.max_seq_length,args)
