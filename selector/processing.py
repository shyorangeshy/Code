import torch
import os
from torch._C import FunctionSchema
from torch.utils.data import TensorDataset

from tqdm import tqdm
from itertools import islice
import pdb
import json

class ACEExample(object):
    def __init__(self, context, labels,com,rel,pos) -> None:
        self.context = context
        self.labels = labels
        self.com=com
        self.rel=rel
        self.pos=pos

class ACEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, merge_index, act_lens) -> None:
        self.encodings = encodings
        self.labels = labels
        self.merge_index = merge_index
        self.act_lens = act_lens
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['merge_index'] = torch.tensor(self.merge_index[idx])
        item['act_lens'] = torch.tensor(self.act_lens[idx])
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
                context, com,rel,label_pos,label_str = traget[0], traget[1], traget[2],traget[-2], traget[-1]
                # com_fragment = com.split(" ")
                # rel_fragment=rel.split(" ")
                assert len(com) == len(rel)
                example = ACEExample(context,label_str,com,rel,label_pos)
                examples.append(example)

        return examples

    def convert_examples_to_features(self, tokenizer, examples, label_to_id, max_seq_length):
        """
        @description  :
            convert the origin examples to pretrain model's(or other) input forms
        ---------
        @param  :
        -------
        @Returns  :
        -------
        """
        
        
        encodings = {}
        encodings["input_ids"] = []
        encodings["attention_mask"] = []
        encodings["token_type_ids"] = []
        labels = []
        act_lens = []
        merge_index = []
        rel_all_B=[" is capable of "," causes "," is a "," is a manner of "," is motivated by "," receives action of "]
        for example in tqdm(examples, total = len(examples), desc = "ACE processing"):
            # *** same as encode_plus, encode just return ids, encode_plus return all information
            # pdb.set_trace()
            # 
            chat_labels= example.labels.split(" ")
            chat_pos=example.pos.split(" ")
            for jj in range(len(chat_labels)):
                word_pos=int(chat_pos[jj])
                word_jj=example.context.split(" ")[word_pos]
                label_pos=int(chat_labels[jj])
                choices=["\""+example.com[word_pos].split(" ")[0]+" "+rel_all_B[int(example.rel[word_pos].split(" ")[i])]+example.com[word_pos].split(" ")[i]+"\"" for i in range(1,len(example.com[word_pos].split(" ")))]
                str_choices=" or ".join(choices)
                prompt="Given the sentence \""+example.context+"\", the sense of "+word_jj+" in the sentence is  "+str_choices+" ?"
                #print('\nhhhh',prompt)
                tokenized_inputs = tokenizer(
                    prompt.split(" "), # transformer有Tokenizer 和 TokenizerFast 两类，其中Fast在输入分词好的结果时，会返回一个
                    add_special_tokens=True,
                    padding = 'max_length', 
                    truncation=True,
                    max_length=max_seq_length,
                    is_split_into_words = True)
                # ***

                
                # align word level labels to tokens
                
                #label = example.labels[-1]
                word_ids = tokenized_inputs.word_ids() # 为了对齐token
                # previous_word_idx = None
                #label_ids = []
                real_len = 0
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        pass
                        # label_ids.append(-100)
                    # We set the label for the first token of each word.
                    # elif word_idx != previous_word_idx:
                    #     label_ids.append(label_to_id[label[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        real_len += 1
                        #label_ids.append(label_to_id[label[word_idx]])
                    # previous_word_idx = word_idx
                merge = [-1 if i == None else i for i in word_ids]
                # print(merge)
                # torch.where(word_ids == None, torch.full_like(word_ids, -1), word_ids)
                # pdb.set_trace()
                encodings["input_ids"].append(tokenized_inputs["input_ids"])
                encodings["attention_mask"].append(tokenized_inputs["attention_mask"])
                encodings["token_type_ids"].append(tokenized_inputs["token_type_ids"])
                labels.append(label_pos)
                act_lens.append(real_len)
                merge_index.append(merge)


        dataset = ACEDataset(encodings, labels, merge_index, act_lens)
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
    return processor.convert_examples_to_features(tokenizer, examples, label_to_id, args.max_seq_length)

