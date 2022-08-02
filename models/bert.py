import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer
# from .layers import AMSoftmax
import re

from cocolm.modeling_cocolm import COCOLMModel
from cocolm.configuration_cocolm import COCOLMConfig
# import lele
from models.layers import MLPLayers
from models.post import OrdinalRegressionLoss


class AbstractBert(nn.Module):
    def __init__(self, config):
        super(AbstractBert, self).__init__()
        self.num_class = 4
        self.hidden_size = 768
        self.dropout = nn.Dropout(p=config.dropout)
        self.isDropout = True if 0 < config.dropout < 1 else False
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        raise NotImplementedError

    def bert_output(self, input):
        with torch.no_grad():
            output = self.bert_model(input_ids=input.input_ids,
                                     token_type_ids=input.token_type_ids,
                                     attention_mask=input.attention_mask,
                                     output_hidden_states=True)
        return output[0], output[1], output[2]

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.num_class = 4
        self.hidden_size = 768
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(
            config.bertmodel, num_labels=4)

    def forward(self, input):
        loss, prediction = self.bert_model(input_ids=input.input_ids,
                                           token_type_ids=input.token_type_ids,
                                           attention_mask=input.attention_mask,
                                           labels=input.esci_label).values()
        return loss, prediction

    def calculate_loss(self, input):
        loss, prediction = self.forward(input)
        return loss

    def predict(self, input):
        loss, prediction = self.forward(input)
        return prediction


class DeBERTaLastLayer(nn.Module):
    def __init__(self, config, num_class=4):
        super(DeBERTaLastLayer, self).__init__()
        self.n_classes = num_class
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 4)
        self.loss = nn.CrossEntropyLoss()

        for name, parm in self.bert_model.named_parameters():
            if name.startswith('encoder.layer'):
                num = re.search(r"encoder.layer.([0-9]*).", name).group(1)
                if int(num) < config.fixed_layer :
                    parm.requires_grad = False
            if name.startswith('embeddings'):
                parm.requires_grad = False

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 output_hidden_states=True)
        sequence_output, hidden_states = output[0], output[1]
        output = self.dropout(hidden_states[-1][:, 0, :])
        logits = self.classifier(output)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class DeBERTa(nn.Module):
    def __init__(self, config, num_class=4):
        super().__init__()
        self.n_classes = num_class
        self.noise_lambda = 0.2
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = MLPLayers([768, 256, 4])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 output_hidden_states=True)
        sequence_output, hidden_states = output[0], output[1]
        output = self.dropout(hidden_states[-1][:, 0, :])
        logits = self.classifier(output)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class DeBERTaLarge(nn.Module):
    def __init__(self, config, num_class=4):
        super().__init__()
        self.n_classes = num_class
        self.noise_lambda = 0.2
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = MLPLayers([1024, 256, 4])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 output_hidden_states=True)
        sequence_output, hidden_states = output[0], output[1]
        output = self.dropout(hidden_states[-1][:, 0, :])
        logits = self.classifier(output)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class DeBERTaProb(nn.Module):
    def __init__(self, config, num_class=4):
        super().__init__()
        self.n_classes = num_class
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 1)
        # for name, parm in self.bert_model.named_parameters():
        #     if not name.startswith('encoder.layer.11') and not name.startswith('pooler'):
        #         parm.requires_grad = False
        self.loss = nn.CrossEntropyLoss()
        self.ord_reg = OrdinalRegressionLoss(4)

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 output_hidden_states=True)
        sequence_output, hidden_states = output[0], output[1]
        output = self.dropout(hidden_states[-1][:, 0, :])
        logits = self.classifier(output)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss, likelihoods = self.ord_reg(logits, input.esci_label)
        # import pdb; pdb.set_trace()
        return loss

    def predict(self, input):
        logits = self.forward(input)
        loss, likelihoods = self.ord_reg(logits, input.esci_label)
        return likelihoods


class DeBERTaLSTM(nn.Module):
    def __init__(self, config, num_class=4):
        super().__init__()
        self.n_classes = num_class
        self.noise_lambda = 0.2
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 4)

        # for name, parm in self.bert_model.named_parameters():
        #     if not name.startswith('encoder.layer.11') and not name.startswith('pooler'):
        #         parm.requires_grad = False
        self.loss = nn.CrossEntropyLoss()
        RNN = getattr(nn, config.rnn_type)
        self.seq_encoder = RNN(768, int(768 / 2), 2, dropout=0.1, bidirectional=True,
                               batch_first=True)
        self.pooling = lele.layers.Pooling('linear_attention', 768)
        # lele.freeze(lele.get_word_embeddings(self.bert_model))

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 output_hidden_states=True)
        sequence_output, hidden_states = output[0], output[1]
        output, _ = self.seq_encoder(sequence_output)
        output = self.dropout(self.pooling(output, input.attention_mask))
        logits = self.classifier(output)
        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class COCOLM(nn.Module):
    def __init__(self, config, num_class=4):
        super().__init__()
        self.n_classes = num_class
        self.bert_config = COCOLMConfig.from_pretrained(config.bertmodel)
        self.bert_config.need_pooler = True
        self.bert_model = COCOLMModel.from_pretrained(config.bertmodel, config=self.bert_config)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(1024, 4)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask)

        sequence_output, pooler_output = output[0], output[1]
        output = self.dropout(pooler_output)
        logits = self.classifier(output)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class RoBERTa(nn.Module):
    def __init__(self, config, num_class=4):
        super().__init__()
        self.n_classes = num_class
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(1024, 4)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 # position_ids=torch.zeros_like(input.input_ids),
                                 # token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 output_hidden_states=True)

        pooler_output = output[1]
        output = self.dropout(pooler_output)
        logits = self.classifier(output)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class DeBERTaCLSAM(nn.Module):
    def __init__(self, config, num_class=4):
        super(DeBERTaCLSAM, self).__init__()
        self.n_classes = num_class
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        self.dropout = nn.Dropout(p=0.1)
        self.amsoftmax = AMSoftmax(768, n_classes=4)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input):
        # [CLS] t-shirt [SEP] nike t-shirt [SEP]
        # 6 * 768
        output = self.bert_model(input_ids=input.input_ids,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 output_hidden_states=True)
        sequence_output, hidden_states = output[0], output[1]
        output = hidden_states[-1][:, 0, :]
        # TODO

        #######
        output = self.dropout(output)
        logits = self.amsoftmax(output, input.esci_label)


class DTDeBERTa(nn.Module):
    def __init__(self, config, num_class=4):
        super(DTDeBERTa, self).__init__()
        self.n_classes = num_class
        self.noise_lambda = 0.2
        # self.bert_config = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 4)
        self.loss = nn.CrossEntropyLoss()
        self.qp_mlp_layer = nn.Linear(768, 768)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.trm_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, input):
        query_output = self.bert_model(input_ids=input.query_input_ids,
                                       token_type_ids=input.query_token_type_ids,
                                       attention_mask=input.query_attention_mask,
                                       output_hidden_states=True)
        query_sequence_output, query_hidden_states = query_output[0], query_output[1]

        product_output = self.bert_model(input_ids=input.product_input_ids,
                                         token_type_ids=input.product_token_type_ids,
                                         attention_mask=input.product_attention_mask,
                                         output_hidden_states=True)
        product_sequence_output, product_hidden_states = product_output[0], product_output[1]

        concat_in = torch.cat([query_hidden_states[-1][:, 0, :], product_hidden_states[-1][:, 0, :]], dim=1)
        concat_in = query_hidden_states[-1][:, 0, :] + product_hidden_states[-1][:, 0, :]
        # att_mask = torch.cat([input.query_attention_mask, input.product_attention_mask], dim=1)
        # dt_output = self.trm_encoder(concat_in,
        #                              src_key_padding_mask=(torch.ones_like(att_mask) -
        #                                                    att_mask).bool())
        dt_output = self.qp_mlp_layer(concat_in)
        output = self.dropout(dt_output)
        logits = self.classifier(output)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class DeBERTaBrand(DeBERTa):
    def __init__(self, config, num_class=4):
        super().__init__(config, num_class=num_class)

    def forward(self, input):
        word_embeddings = self.bert_model.embeddings.word_embeddings(input.input_ids)
        brand_embedding = torch.einsum("bh,d->bhd", [input.brand_flag, self.is_brand_embedding])
        output = self.bert_model(input_ids=None,
                                 token_type_ids=input.token_type_ids,
                                 attention_mask=input.attention_mask,
                                 inputs_embeds=word_embeddings + brand_embedding,
                                 output_hidden_states=True)

        sequence_output, hidden_states = output[0], output[1]
        output = self.dropout(hidden_states[-1][:, 0, :])
        logits = self.classifier(output)

        return logits


class MPNet(nn.Module):
    def __init__(self, config, num_class=4):
        super(MPNet, self).__init__()
        self.n_classes = num_class
        self.bert_model = AutoModel.from_pretrained(
            config.bertmodel)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(768, 4)
        self.loss = nn.CrossEntropyLoss()

    def mean_pooling(self, model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input):
        output = self.bert_model(input_ids=input.input_ids,
                                 attention_mask=input.attention_mask)
        sentence_embeddings = self.mean_pooling(output, input.attention_mask)
        logits = self.classifier(sentence_embeddings)

        return logits

    def calculate_loss(self, input):
        logits = self.forward(input)
        loss = self.loss(logits, input.esci_label)
        return loss

    def predict(self, input):
        logits = self.forward(input)
        return logits


class BertForClass(AbstractBert):
    def __init__(self, config):
        super(BertForClass, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        if self.isDropout:
            concat_out = self.dropout(concat_out)
        logit = self.classifier(concat_out)
        return logit


class BertForClass_MultiDropout(AbstractBert):
    def __init__(self, config):
        super(BertForClass_MultiDropout, self).__init__(config)
        self.multi_drop = 5
        self.multi_dropouts = nn.ModuleList(
            [nn.Dropout(self.dropout) for _ in range(self.multi_drop)])
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        for j, dropout in enumerate(self.multi_dropouts):
            if j == 0:
                logit = self.classifier(dropout(concat_out)) / self.multi_drop
            else:
                logit += self.classifier(dropout(concat_out)) / self.multi_drop

        return logit


class BertLastTwoCls(AbstractBert):
    def __init__(self, config):
        super(BertLastTwoCls, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 3, 4)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        output = torch.cat(
            (pooler_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0]), dim=1)
        output = self.dropout(output)
        logits = self.classifier(output)

        return logits


class BertLastTwoClsPooler(AbstractBert):
    def __init__(self, config):
        super(BertLastTwoClsPooler, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 3, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        output = torch.cat(
            (pooler_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0]), dim=1)
        if self.isDropout:
            output = self.dropout(output)
        logit = self.classifier(output)

        return logit


class BertLastTwoEmbeddings(AbstractBert):
    def __init__(self, config):
        super(BertLastTwoEmbeddings, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 2, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)

        output = torch.cat(
            (hidden_states1, hidden_states2), dim=1)
        if self.isDropout:
            output = self.dropout(output)
        logit = self.classifier(output)

        return logit


class BertLastTwoEmbeddingsPooler(AbstractBert):
    def __init__(self, config):
        super(BertLastTwoEmbeddingsPooler, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 3, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)

        output = torch.cat(
            (pooler_output, hidden_states1, hidden_states2), dim=1)
        if self.isDropout:
            output = self.dropout(output)
        logit = self.classifier(output)

        return logit


class BertLastFourCls(AbstractBert):
    def __init__(self, config):
        super(BertLastFourCls, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 4, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        output = torch.cat(
            (hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), dim=1)
        if self.isDropout:
            output = self.dropout(output)
        logit = self.classifier(output)

        return logit


class BertLastFourClsPooler(AbstractBert):
    def __init__(self, config):
        super(BertLastFourClsPooler, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 5, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)

        output = torch.cat(
            (pooler_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0],
             hidden_states[-4][:, 0]), dim=1)
        if self.isDropout:
            output = self.dropout(output)
        logit = self.classifier(output)

        return logit


class BertLastFourEmbeddings(AbstractBert):
    def __init__(self, config):
        super(BertLastFourEmbeddings, self).__init__(config)
        self.classifier = nn.Linear(self.hidden_size * 4, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)
        hidden_states3 = torch.mean(hidden_states[-3], dim=1)
        hidden_states4 = torch.mean(hidden_states[-4], dim=1)
        output = torch.cat(
            (hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1)
        if self.isDropout:
            output = self.dropout(output)
        logit = self.classifier(output)

        return logit


class BertLastFourEmbeddingsPooler(AbstractBert):
    def __init__(self, config):
        super(BertLastFourEmbeddingsPooler, self).__init__(config)
        self.classifier = nn.Linear(
            self.bert_self.hidden_size * 5, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        hidden_states1 = torch.mean(hidden_states[-1], dim=1)
        hidden_states2 = torch.mean(hidden_states[-2], dim=1)
        hidden_states3 = torch.mean(hidden_states[-3], dim=1)
        hidden_states4 = torch.mean(hidden_states[-4], dim=1)
        output = torch.cat(
            (pooler_output, hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1)
        if self.isDropout:
            output = self.dropout(output)
        logit = self.classifier(output)

        return logit


class BertDynCls(AbstractBert):
    def __init__(self, config):
        super(BertDynCls, self).__init__(config)
        self.dynWeight = nn.Linear(self.hidden_size, 1)
        self.dense = nn.Linear(self.hidden_size, 512)
        self.classifier = nn.Linear(512, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        batch_size = pooler_output.shape[0]

        hid_avg_list = None
        weight_list = None
        for i, hidden in enumerate(hidden_states):
            hid_avg = hidden_states[-(i + 1)][0]
            weight = self.dynWeight(hid_avg).repeat(1, self.hidden_size)
            if hid_avg_list is None:
                hid_avg_list = hid_avg
            else:
                hid_avg_list = torch.cat((hid_avg_list, hid_avg), dim=1)

            if weight_list is None:
                weight_list = hid_avg
            else:
                weight_list = torch.cat((weight_list, weight), dim=1)

        concat_out = weight_list.mul_(hid_avg_list)
        concat_out = concat_out.reshape(batch_size, -1, self.hidden_size)
        concat_out = torch.sum(concat_out, dim=1)

        if self.isDropout:
            concat_out = self.dropout(concat_out)
        concat_out = self.dense(concat_out)
        logit = self.classifier(concat_out)

        return logit


class BertDynEmbeddings(AbstractBert):
    def __init__(self, config):
        super(BertDynEmbeddings, self).__init__(config)
        self.dynWeight = nn.Linear(self.hidden_size, 1)
        self.dense = nn.Linear(self.hidden_size, 512)
        self.classifier = nn.Linear(512, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)

        batch_size = pooler_output.shape[0]

        hid_avg_list = None
        weight_list = None
        for i, hidden in enumerate(hidden_states):
            hid_avg = torch.mean(hidden_states[-(i + 1)], dim=1)
            weight = self.dynWeight(hid_avg).repeat(1, self.hidden_size)
            if hid_avg_list is None:
                hid_avg_list = hid_avg
            else:
                hid_avg_list = torch.cat((hid_avg_list, hid_avg), dim=1)

            if weight_list is None:
                weight_list = hid_avg
            else:
                weight_list = torch.cat((weight_list, weight), dim=1)

        concat_out = weight_list.mul_(hid_avg_list)
        concat_out = concat_out.reshape(batch_size, -1, self.hidden_size)
        concat_out = torch.sum(concat_out, dim=1)

        if self.isDropout:
            concat_out = self.dropout(concat_out)

        concat_out = self.dense(concat_out)
        logit = self.classifier(concat_out)

        return logit


class BertRNN(AbstractBert):

    def __init__(self, config):
        super(BertRNN, self).__init__(config)
        self.rnn_type = "gru"
        self.bidirectional = True
        self.hidden_dim = 768
        self.n_layers = 2
        self.batch_first = True
        self.drop_out = 0.1
        self.num_directions = 1 if not self.bidirectional else 2

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.hidden_size,
                               hidden_size=self.hidden_dim // 2,
                               num_layers=self.n_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first,
                               dropout=self.drop_out)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.hidden_size,
                              hidden_size=self.hidden_dim // 2,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)
        else:
            self.rnn = nn.RNN(self.hidden_size,
                              hidden_size=self.hidden_dim // 2,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)

        self.fc_rnn = nn.Linear(
            self.hidden_dim * self.num_directions, self.num_class)

    def forward(self, input):

        sequence_output, pooler_output, hidden_states = self.bert_output(input)

        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(sequence_output)
        else:
            output, (hidden, cell) = self.rnn(sequence_output)

        # output = [ batch size, sent len, hidden_dim * bidirectional]
        batch_size, max_seq_len, hidden_dim = output.shape
        hidden = torch.transpose(hidden, 1, 0)
        hidden = torch.mean(torch.reshape(
            hidden, [batch_size, -1, hidden_dim]), dim=1)
        output = torch.sum(output, dim=1)
        fc_input = self.dropout(output + hidden)

        # output = torch.mean(output, dim=1)
        # fc_input = self.dropout(output)
        out = self.fc_rnn(fc_input)

        return out


class BertCNN(AbstractBert):

    def __init__(self, config):
        super(BertCNN, self).__init__(config)
        self.num_filters = 100
        self.hidden_size = self.hidden_size
        self.filter_sizes = {3, 4, 5}
        self.drop_out = 0.5

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.hidden_size)) for k in self.filter_sizes])

        self.fc_cnn = nn.Linear(
            self.num_filters * len(self.filter_sizes), self.num_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        sequence_output = self.dropout(sequence_output)
        out = sequence_output.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out


class BertRCNN(AbstractBert):
    def __init__(self, config):
        super(BertRCNN, self).__init__(config)
        self.rnn_type = "lstm"
        self.bidirectional = True
        self.hidden_dim = 256
        self.n_layers = 2
        self.batch_first = True
        self.drop_out = 0.5

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.hidden_size,
                               self.hidden_dim,
                               num_layers=self.n_layers,
                               bidirectional=self.bidirectional,
                               batch_first=self.batch_first,
                               dropout=self.drop_out)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.hidden_size,
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)
        else:
            self.rnn = nn.RNN(self.hidden_size,
                              hidden_size=self.hidden_dim,
                              num_layers=self.n_layers,
                              bidirectional=self.bidirectional,
                              batch_first=self.batch_first,
                              dropout=self.drop_out)

        # self.maxpool = nn.MaxPool1d()

        self.fc = nn.Linear(self.hidden_dim * self.n_layers, self.num_class)

    def forward(self, input):
        sequence_output, pooler_output, hidden_states = self.bert_output(input)
        sentence_len = sequence_output.shape[1]
        pooler_output = pooler_output.unsqueeze(
            dim=1).repeat(1, sentence_len, 1)
        bert_sentence = sequence_output + pooler_output

        self.rnn.flatten_parameters()
        if self.rnn_type in ['rnn', 'gru']:
            output, hidden = self.rnn(bert_sentence)
        else:
            output, (hidden, cell) = self.rnn(bert_sentence)

        batch_size, max_seq_len, hidden_dim = output.shape
        out = torch.transpose(output.relu(), 1, 2)

        out = F.max_pool1d(out, max_seq_len).squeeze()
        out = self.fc(out)

        return out

# class XLNet(nn.Module):
#
#     def __init__(self, config):
#         super(XLNet, self).__init__()
#         self.xlnet = XLNetModel.from_pretrained(config.model_path)
#
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.fc = nn.Linear(self.xlnet.d_model, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output = self.xlnet(input_ids=input_ids, token_type_ids=segment_ids,
#                                      attention_mask=input_masks)
#         sequence_output = torch.sum(sequence_output[0], dim=1)
#         if self.isDropout:
#             sequence_output = self.dropout(sequence_output)
#         out = self.fc(sequence_output)
#         return out
#
#
# class ElectraClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(self.hidden_size, self.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.out_proj = nn.Linear(self.hidden_size, config.num_labels)
#
#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x
#
# class Electra(nn.Module):
#
#     def __init__(self, config):
#         super(Electra, self).__init__()
#         self.electra = ElectraModel.from_pretrained(config.model_path)
#
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.electra_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json)
#         self.electra_config.num_labels = config.num_class
#         self.fc = ElectraClassificationHead(self.electra_config)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         discriminator_hidden_states = self.electra(input_ids=input_ids, token_type_ids=segment_ids,
#                                      attention_mask=input_masks)
#
#         sequence_output = discriminator_hidden_states[0]
#         out = self.fc(sequence_output)
#         return out

# MODEL_CLASSES = {
#     'bertforclass': BertForClass,
#     'bertlasttwocls': BertLastTwoCls,
#     'bertlasttwoclspooler': BertLastTwoClsPooler,
#     'bertlasttwoembeddings': BertLastTwoEmbeddings,
#     'bertlasttwoembeddingspooler': BertLastTwoEmbeddingsPooler,
#     'bertlastfourcls': BertLastFourCls,
#     'bertlastfourclspooler': BertLastFourClsPooler,
#     'bertlastfourembeddings': BertLastFourEmbeddings,
#     'bertlastfourembeddingspooler': BertLastFourEmbeddingsPooler,
#     'bertdyncls': BertDynCls,
#     'bertdynembeddings': BertDynEmbeddings,
#     'bertrnn': BertRNN,
#     'bertcnn': BertCNN,
#     'bertrcnn': BertRCNN,
#     # 'xlnet': XLNet,
#     # 'electra': Electra,
# }
