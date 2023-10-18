from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn


class myBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.num_labels = 48
        self.classifier = nn.Linear(config.hidden_size, 48)
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
            device=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        sequence_output = outputs[0][:,:1,:]
        
        logits = self.classifier(sequence_output)
        #kkk torch.Size([8, 1, 768]) torch.Size([8, 1, 48])
        # print('\nkkk',sequence_output.shape,logits.shape)
        # print('\nuuuu',labels.shape)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                #active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                #print('\nqqq',attention_mask.shape,active_loss.shape,'\n',active_logits.shape)
                #qqq torch.Size([8, 512]) torch.Size([4096]) torch.Size([8, 48])
                # active_labels = torch.where(
                #     active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                # )
                loss = loss_fct(active_logits, labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
