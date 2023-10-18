import torch.nn.functional as F
from transformers import BertConfig
from transformers import BertModel,BertLayer
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import numpy as np

class myBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(2*config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            label_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            device=None,
            com_input_ids=None,
            com_attention_mask=None,
            com_token_type_ids=None,
            merge_index=None
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #print('\nkkkk',attention_mask)
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        batch_size,seq_len,embed_size=sequence_output.shape
        finV=True
        for i in range(batch_size):
            outputs1 = self.bert(
                torch.tensor(com_input_ids[i]).to(device),
                attention_mask=torch.tensor(com_attention_mask[i]).to(device),
                token_type_ids=torch.tensor(com_token_type_ids[i]).to(device),
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            a=torch.randn(1,embed_size).to(device)
            b=outputs1[0][:,:1,:]
            b=torch.squeeze(b,dim=1).to(device)
            z,_=b.shape
            c=torch.randn(seq_len-z-1,embed_size).to(device)
            b=torch.cat([a,b,c],dim=0) 
            b=torch.unsqueeze(b,0)
            if finV:
                comse=b
                
                finV=False
            else:
                comse=torch.cat((comse,b),0)
        # seqnew=self.mapnew(torch.cat((sequence_output,comse),-1))
        # seqnorm=self.LayerNorm(seqnew)
        # seqdrop=self.dropout(seqnorm)
        # layer_mask=self.get_extended_attention_mask(attention_mask, input_ids.size())
        # layer_outputs = self.trans(
        #             seqdrop,
        #             layer_mask,
        #             head_mask=None,
        #             encoder_hidden_states=None,
        #             encoder_attention_mask=None,
        #             past_key_value=None,
        #             output_attentions=False,
        #         )
        #print('\nmmmm',layer_outputs[0].shape)
        # aa_bi=self.biaffine(self.dropout(sequence_output),self.dropout(comse))
        # #print('\noooo',aa_bi)
        # aa_bi_x=torch.squeeze(torch.softmax(aa_bi, dim=1),-1)
        # aa_bi_y=torch.squeeze(torch.softmax(aa_bi, dim=2),-1)

        # seq_new=torch.einsum('bij,bjk->bik',aa_bi_x,comse)+torch.einsum('bij,bik->bjk',aa_bi_y,sequence_output)
        #print('\nuuuu',seq_new.shape)+torch.einsum('bij,bjk->bik',(1-aa_bi_x),sequence_output)
        logits = self.classifier(torch.cat((sequence_output,comse),-1))
        # logit_seq=self.cls_task(seq_new)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                # active_logit_seq=logit_seq.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # dot_product = np.dot(A, B)
        # norm_A = np.linalg.norm(A)
        # norm_B = np.linalg.norm(B)
        # return dot_product / (norm_A * norm_B)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
