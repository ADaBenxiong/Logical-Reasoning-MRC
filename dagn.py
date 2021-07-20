
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
from transformers import BertPreTrainedModel, BertModel, RobertaModel
import collections
import json

from utils import punctuations, Vocab_text

# BERT模型词表预处理
# vocab = tokenizer.vocab     #字典类型表示{token:id}
# ids_to_tokens = tokenizer.ids_to_tokens #列表类型表示[id]
# def load_vocab(vocab_file):
#     vocab = collections.OrderedDict()
#     with open(vocab_file, 'r', encoding = 'utf-8') as reader:
#         tokens = reader.readlines()
#     for index, token in enumerate(tokens):
#         token = token.rstrip("\n")
#         vocab[token] = index
#     return vocab

#RoBERTa模型词表预处理
def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, 'r', encoding = 'utf-8') as reader:
        encoder = json.load(reader)
    vocab = encoder
    return vocab

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def replace_masked_values(tensor, mask, replace_with):

    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill((1 - mask).bool(), replace_with)

class FFNLayer(nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer_norm = True):
        super(FFNLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        if layer_norm:
            self.ln = nn.LayerNorm(intermediate_dim)
        else:
            self.ln = None
        self.dropout_func = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, input):
        inter = self.fc1(self.dropout_func(input))
        inter_act = gelu(inter)
        if self.ln:
            inter_act = self.ln(inter_act)
        return self.fc2(inter_act)

class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)

class ArgumentGCN(nn.Module):

    def __init__(self, node_dim, iteration_steps=1):  # 1024， 0， 2
        super(ArgumentGCN, self).__init__()

        self.node_dim = node_dim  # 1024
        self.iteration_steps = iteration_steps  # 2

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)  # 1024 -> 1

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)  # 1024 -> 1024

        self._node_fc_punctuation = torch.nn.Linear(node_dim, node_dim, bias=False)  # 1024 -> 1024

    def forward(self,
                node,  # （4， 16， 1024）
                node_mask,  # （4， 16）
                punctuation_graph,  # （4， 16， 16）
        ):
        node_len = node.size(1)  # 分成node_len个片段

        # 单位矩阵
        diagmat = torch.diagflat(torch.ones(node.size(1), dtype=torch.long, device=node.device))
        # diagmat是一个node.size(1) * node.size(1)维度的向量
        # 四个单位矩阵
        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)
        # diagmat是一个node.size(0)【值为4】* node.size(1) * node.size(1)维度的向量
        # （4， 1， 16）             #（4， 16， 1）

        # print(node_mask[0]) #tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], device='cuda:0')
        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)
        # print(dd_graph[0])  #减去单位矩阵

        graph_punctuation = dd_graph * punctuation_graph
        # print(graph_punctuation[0]) #和punctuation_graph矩阵相同

        # 计算每个节点的度数（4， 16）
        node_neighbor_num = graph_punctuation.sum(-1)
        # print(node_neighbor_num[0]) #tensor([1., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0.], device='cuda:0')
        # 计算每个节点度数的mask
        node_neighbor_num_mask = (node_neighbor_num >= 1).long()
        # print(node_neighbor_num_mask[0])    #tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], device='cuda:0')
        node_neighbor_num = replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)
        # print(node_neighbor_num[0]) #tensor([1., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1.], device='cuda:0')

        all_weight = []
        # 两步操作
        for step in range(self.iteration_steps):
            d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)
            all_weight.append(d_node_weight)
            self_node_info = self._self_node_fc(node)

            node_info_punctuation = self._node_fc_punctuation(node)
            node_weight = replace_masked_values(
                d_node_weight.unsqueeze(1).expand(-1, node_len, -1),
                graph_punctuation,
                0
            )
            node_info_punctuation = torch.matmul(node_weight, node_info_punctuation)
            agg_node_info = node_info_punctuation / node_neighbor_num.unsqueeze(-1)

            node = F.relu(self_node_info + agg_node_info)

        all_weight = [weight.unsqueeze(1) for weight in all_weight]
        all_weight = torch.cat(all_weight, dim=1)

        return node, all_weight

class GCN(nn.Module):
    def __init__(self, batch_size, node_dim, hidden_size_1, hidden_size_2):
        super(GCN, self).__init__()

        self.weight1 = nn.Parameter(torch.FloatTensor(batch_size, node_dim, hidden_size_1))
        var1 = 2./(self.weight1.size(1) + self.weight1.size(0))
        self.weight1.data.normal_(0, var1)

        self.weight2 = nn.Parameter(torch.FloatTensor(batch_size, hidden_size_1, hidden_size_2))
        var2 = 2./(self.weight2.size(1) + self.weight2.size(0))
        self.weight2.data.normal_(0, var2)

        # self.bias1 = nn.Parameter(torch.FloatTensor(hidden_size_1))
        # self.bias1.data.normal_(0, var1)
        # self.bias2 = nn.Parameter(torch.FloatTensor(hidden_size_2))
        # self.bias2.data.normal_(0, var2)

        self.fc = nn.Linear(hidden_size_2, node_dim)

    def forward(self,
                X,      #输入的数据
                A_hat):

        #A_hat = torch.tensor(A_hat, requires_grad=False).float()
        A_hat = A_hat.detach().float()

        X = torch.bmm(X, self.weight1)
        X = F.relu(torch.bmm(A_hat, X))

        X = torch.bmm(X, self.weight2)
        X = F.relu(torch.bmm(A_hat, X))
        # for i in range(X.size(0)):
        #
        #     print(X[i].shape)
        #     print(self.weight1.shape)
        #     ans1 = torch.mm(X[i], self.weight1)
        #     # X[i] = torch.mm(X[i], self.weight1)
        #     # X[i] = X[i] + self.bias1
        #     print(A_hat[i].shape)
        #     # print(X[i].shape)
        #     gcn_1 = F.relu(torch.mm(A_hat[i], ans1))
        #     # print(X[i].shape)
        #
        #     ans2= torch.mm(gcn_1, self.weight2)
        #     # X[i] = X[i] + self.bias2
        #     gcn_2 = F.relu(torch.mm(A_hat[i], ans2))
        #
        #     gcn_final = self.fc(gcn_2)
        #     X[i] = gcn_final

        return X

class DAGN_GCN(BertPreTrainedModel):
    def __init__(self,
                 config,
                 vocab_file=None,
                 max_rel_id=4,
                 gnn_version='GCN',
                 use_gcn=True,
                 use_pool=False,
                 matrix_size=32,
                 gcn_steps=2
                 ):
        super().__init__(config)
        self.max_rel_id = max_rel_id
        self.vocab_file = vocab_file
        self.use_gcn = use_gcn
        self.use_pool = use_pool
        self.gnn_version = gnn_version

        self.hidden_size = config.hidden_size
        self.hidden_size_1 = self.hidden_size * 4
        self.hidden_size_2 = self.hidden_size
        self.dropout_prob = config.hidden_dropout_prob
        self.roberta = RobertaModel(config)

        if self.use_pool:
            self.dropout = nn.Dropout(self.dropout_prob)
            self.classifier = nn.Linear(self.hidden_size, 1)

        if self.use_gcn:

            assert self.vocab_file
            self.vocab = load_vocab(self.vocab_file)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
            self._iteration_steps = gcn_steps
            self.matrix_size = matrix_size

            print("GCN")
            if gnn_version == "GCN":
                self.GCN = GCN(batch_size=4, node_dim=self.hidden_size, hidden_size_1=self.hidden_size_1, hidden_size_2=self.hidden_size_2)
                print(self.GCN)
            else:
                print("gnn_version == {}".format(gnn_version))
                raise Exception()

            self._gcn_prj_ln = nn.LayerNorm(self.hidden_size)
            self._gcn_enc = ResidualGRU(self.hidden_size, self.dropout_prob, 2)

            self._proj_span_num = FFNLayer(self.hidden_size, self.hidden_size, 1, self.dropout_prob)

        self.init_weights()

    #输入三个参数大小相同（输入数据、masked数据、分隔符矩阵数据）
    def split_into_spans(self, sequence_output, attention_mask, masked_punctuations):

        # embed_size = sequence_output.size(-1)
        device = sequence_output.device
        #分割后的片段向量
        encoded_spans = []
        #分割后的片段masked值
        span_masks = []
        #分割符的位置
        truncated_edges = []
        #分割后片段的位置
        node_in_seq_indices = []

        #记录每次分割后的片段数
        edges = []

        #求span_masks
        for num in range(masked_punctuations.size(0)):
            edges.append((masked_punctuations[num] > 0).sum().item())
        max_ = max(edges)
        max_size = self.matrix_size

        for num in range(masked_punctuations.size(0)):
            masked_list = []
            for i in range(min(edges[num], max_size)):
                masked_list.append(1)
            for i in range(max_size - min(edges[num], max_size)):
                masked_list.append(0)
            span_masks.append(masked_list)
        span_masks = torch.Tensor(span_masks)
        span_masks = span_masks.to(device).long()

        #求truncated_edges
        for i in range(masked_punctuations.size(0)):
            idx_edges = []
            for j in range(masked_punctuations.size(1)):
                if masked_punctuations[i][j] > 0:
                    idx_edges.append(j)
                    if len(idx_edges) == max_size:
                        break
            if len(idx_edges) < max_size:
                for j in range(max_size - len(idx_edges)):
                    idx_edges.append(0)
            truncated_edges.append(idx_edges)

        #求node_in_seq_indices
        for index in range(len(truncated_edges)):
            example = []
            start_ = 0
            for figure in range(len(truncated_edges[index])):
                number = []
                for k in range(start_, truncated_edges[index][figure]):
                    number.append(k)
                start_ = truncated_edges[index][figure] + 1
                example.append(number)
                if len(example) == max_size:
                    break
            if len(example) < max_size:
                for figure in range(max_size - len(example)):
                    example.append([0])
            node_in_seq_indices.append(example)

        for i in range(len(node_in_seq_indices)):
            encoded_span = []
            spans = node_in_seq_indices[i]
            for j in spans:
                ans_add = 0
                ans_add_ = np.zeros(self.hidden_size)
                ans_add_ = ans_add_.tolist()
                if len(j) == 1 and j[0] == 0:
                    encoded_span.append(torch.zeros(1024))
                else:
                    for k in j:
                        ans_add += sequence_output[i][k]
                    if len(j) > 0:
                        ans_add = (ans_add / len(j))
                        ans_add_ = [i.item() for i in ans_add]
                    # ans_add = (ans_add / max(len(j), 1))    #有问题
                    # ans_add_ = [i.item() for i in ans_add]  #有问题
                    encoded_span.append(ans_add_)
            if len(spans) < max_:
                for j in range(max_ - len(spans)):
                    encoded_span.append(torch.zeros(1024))
            encoded_spans.append(encoded_span)
        encoded_spans = torch.Tensor(encoded_spans)
        encoded_spans = encoded_spans.to(device).float()

        return encoded_spans, span_masks, truncated_edges, node_in_seq_indices

    def get_adjacency_matrix(self, edges, n_nodes, device):
        batch_size = len(edges)
        punct_graph = torch.zeros((batch_size, n_nodes, n_nodes))


        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if i != len(sample_edges) - 1:
                    punct_graph[b, i, i+1] = 1
                    punct_graph[b, i+1, i] = 1

        A_hat = []
        for i in range(batch_size):
            node_neighbor_num = punct_graph[i].sum(-1)
            degrees = []
            for j in range(len(node_neighbor_num)):
                degrees.append(node_neighbor_num[j] ** (-0.5))
            degrees = np.diag(degrees)
            degrees = torch.Tensor(degrees)
            hat_ = degrees @ punct_graph[i] @ degrees
            A_hat.append(hat_)
        # A_hat = torch.tensor(A_hat).float()
        A_hat = torch.tensor([item.cpu().detach().numpy() for item in A_hat]).float()

        return punct_graph.to(device), A_hat.to(device)

    #indices: 4, node: (4, 11, 1024), size: (4, 256, 1024)
    def get_gcn_info_vector(self, indices, node, size, device):
        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
                if len(ids) == 1 and ids[0] == 0:
                    continue
                gcn_info_vec[b, ids] = emb

        return gcn_info_vec

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids=None,
                labels=4
                ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        bert_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )

        sequence_output = bert_outputs[0]    #矩阵维度（4， 256， 1024） 【4 = batch_size * 4】
        pooled_output = bert_outputs[1]      #矩阵维度（4， 1024）

        if self.use_gcn:
            ids_punctuations = [self.vocab[idx] for idx in punctuations]
            masked_punctuations = torch.zeros(input_ids.size())
            for x in range(masked_punctuations.size(0)):
                for y in range(masked_punctuations.size(1)):
                    if input_ids[x][y] in ids_punctuations:
                        masked_punctuations[x][y] = self.max_rel_id

            #（batch_size * 4, 32, 1024）, (batch_size * 4, 32, 1024), (batch_size * 4, 32), (batch_size * 4, 32, list)
            encoded_spans, span_mask, edges, node_in_seq_indices = self.split_into_spans(sequence_output, attention_mask, masked_punctuations)

            #(batch_size * 4, 32, 32), (batch_size * 4, 32, 32)正则化度矩阵
            punctuation_graph, A_hat= self.get_adjacency_matrix(edges, n_nodes=encoded_spans.size(1), device=encoded_spans.device)
            node = self.GCN(encoded_spans, A_hat)

            # 单词分解后的和 sequence_output同大小的矩阵
            gcn_info_vec = self.get_gcn_info_vector(node_in_seq_indices, node, size=sequence_output.size(), device=sequence_output.device)

            gcn_updated_sequence_output = self._gcn_enc(self._gcn_prj_ln(sequence_output + gcn_info_vec))

            gcn_logits = self._proj_span_num(gcn_updated_sequence_output[:, 0])

        if self.use_pool:
            pooled_output = self.dropout(pooled_output)
            baseline_logits = self.classifier(pooled_output)

        if self.use_gcn and self.use_pool:
            logits = gcn_logits + baseline_logits
        elif self.use_gcn:
            logits = gcn_logits
        elif self.use_pool:
            logits = baseline_logits
        else:
            raise Exception

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss, ) + outputs
        # print(outputs)
        # print(labels)
        # print(np.argmax(outputs[1].detach().cpu().numpy(), axis=1))
        return outputs

class DAGN_Modify(BertPreTrainedModel):
    def __init__(self,
                 config,
                 vocab_file=None,
                 max_rel_id=4,
                 gnn_version='GCN',
                 use_gcn=True,
                 use_pool=False,
                 gcn_steps=2,
                 ):
        super().__init__(config)

        self.max_rel_id = max_rel_id

        self.vocab_file = vocab_file
        self.use_gcn = use_gcn
        self.use_pool = use_pool
        self.gnn_version = gnn_version
        self.gcn_steps = gcn_steps


        self.hidden_size = config.hidden_size
        self.dropout_prob = config.hidden_dropout_prob
        self.roberta = RobertaModel(config)

        if self.use_pool:
            self.dropout = nn.Dropout(self.dropout_prob)
            self.classifier = nn.Linear(self.hidden_size, 1)
        if self.use_gcn:

            assert self.vocab_file
            self.vocab = load_vocab(self.vocab_file)
            self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
            self._iteration_steps = gcn_steps

            if gnn_version == "GCN":
                self._gcn = ArgumentGCN(node_dim=self.hidden_size, iteration_steps=gcn_steps)
            else:
                print("gnn_version == {}".format(gnn_version))
                raise Exception()

            self._gcn_prj_ln = nn.LayerNorm(self.hidden_size)
            self._gcn_enc = ResidualGRU(self.hidden_size, self.dropout_prob, 2)

            self._proj_span_num = FFNLayer(self.hidden_size, self.hidden_size, 1, self.dropout_prob)

        self.init_weights()

    #输入三个参数大小相同（输入数据、masked数据、分隔符矩阵数据）
    def split_into_spans(self, sequence_output, attention_mask, masked_punctuations):

        # embed_size = sequence_output.size(-1)
        device = sequence_output.device
        #分割后的片段向量
        encoded_spans = []
        #分割后的片段masked值
        span_masks = []
        #分割符的位置
        truncated_edges = []
        #分割后片段的位置
        node_in_seq_indices = []

        #记录每次分割后的片段数
        edges = []

        #求span_masks
        for num in range(masked_punctuations.size(0)):
            edges.append((masked_punctuations[num] > 0).sum().item())
        max_ = max(edges)

        for num in range(masked_punctuations.size(0)):
            masked_list = []
            for i in range(edges[num]):
                masked_list.append(1)
            for i in range(max_ - edges[num]):
                masked_list.append(0)
            span_masks.append(masked_list)
        span_masks = torch.Tensor(span_masks)
        span_masks = span_masks.to(device).long()

        #求truncated_edges
        for i in range(masked_punctuations.size(0)):
            idx_edges = []
            for j in range(masked_punctuations.size(1)):
                if masked_punctuations[i][j] > 0:
                    idx_edges.append(j)
            truncated_edges.append(idx_edges)

        #求node_in_seq_indices
        for index in range(len(truncated_edges)):
            example = []
            start_ = 0
            for figure in range(len(truncated_edges[index])):
                number = []
                for k in range(start_, truncated_edges[index][figure]):
                    number.append(k)
                start_ = truncated_edges[index][figure] + 1
                example.append(number)
            node_in_seq_indices.append(example)

        for i in range(len(node_in_seq_indices)):
            encoded_span = []
            spans = node_in_seq_indices[i]
            for j in spans:
                ans_add = 0
                ans_add_ = np.zeros(self.hidden_size)
                ans_add_ = ans_add_.tolist()
                for k in j:
                    ans_add += sequence_output[i][k]
                if len(j) > 0:
                    ans_add = (ans_add / len(j))
                    ans_add_ = [i.item() for i in ans_add]
                # ans_add = (ans_add / max(len(j), 1))    #有问题
                # ans_add_ = [i.item() for i in ans_add]  #有问题
                encoded_span.append(ans_add_)
            if len(spans) < max_:
                for j in range(max_ - len(spans)):
                    encoded_span.append(torch.zeros(1024))
            encoded_spans.append(encoded_span)
        encoded_spans = torch.Tensor(encoded_spans)
        encoded_spans = encoded_spans.to(device).float()

        return encoded_spans, span_masks, truncated_edges, node_in_seq_indices

    def get_adjacency_matrix(self, edges, n_nodes, device):
        batch_size = len(edges)
        punct_graph = torch.zeros((batch_size, n_nodes, n_nodes))

        for b, sample_edges in enumerate(edges):
            for i, edge_value in enumerate(sample_edges):
                if i != len(sample_edges) - 1:
                    punct_graph[b, i, i+1] = 1
                    punct_graph[b, i+1, i] = 1

        return punct_graph.to(device)

    #indices: 4, node: (4, 11, 1024), size: (4, 256, 1024)
    def get_gcn_info_vector(self, indices, node, size, device):
        batch_size = size[0]
        gcn_info_vec = torch.zeros(size=size, dtype=torch.float, device=device)

        for b in range(batch_size):
            for ids, emb in zip(indices[b], node[b]):
                gcn_info_vec[b, ids] = emb

        return gcn_info_vec


    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids = None,
                labels = 4
                ):

        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        bert_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=None
        )
        sequence_output = bert_outputs[0]    #矩阵维度（4， 256， 1024） 【4 = batch_size * 4】
        pooled_output = bert_outputs[1]      #矩阵维度（4， 1024）

        if self.use_gcn:
            # vocab, masked_punctuations = Vocab_text()
            ids_punctuations = [self.vocab[idx] for idx in punctuations]    #特殊标记符号
            masked_punctuations = torch.zeros(input_ids.size())
            for x in range(masked_punctuations.size(0)):
                for y in range(masked_punctuations.size(1)):
                    if input_ids[x][y] in ids_punctuations:
                        masked_punctuations[x][y] = self.max_rel_id

            encoded_spans, span_mask, edges, node_in_seq_indices = self.split_into_spans(sequence_output, attention_mask, masked_punctuations)

            punctuation_graph = self.get_adjacency_matrix(edges, n_nodes=encoded_spans.size(1), device=encoded_spans.device)

            #经过邻居传递后的节点信息 node <- （4， 15， 1024）， node_weight <- (4, 2, 15)
            node, node_weight = self._gcn(node=encoded_spans, node_mask=span_mask, punctuation_graph=punctuation_graph)

            #单词分解后的和 sequence_output同大小的矩阵
            gcn_info_vec = self.get_gcn_info_vector(node_in_seq_indices, node, size=sequence_output.size(), device=sequence_output.device)

            # print(gcn_info_vec.shape)
            gcn_updated_sequence_output = self._gcn_enc(self._gcn_prj_ln(sequence_output + gcn_info_vec))
            #gcn_updated_sequence_output = self._gcn_enc(self._gcn_prj_ln(sequence_output))
            # print(gcn_updated_sequence_output.shape)
            #
            # sequence_h2_weight = self._proj_sequence_h(gcn_updated_sequence_output).squeeze(-1)
            # print(sequence_h2_weight.shape)
            # print(gcn_updated_sequence_output[:, 0].shape)  #(4, 1024)

            gcn_logits = self._proj_span_num(gcn_updated_sequence_output[:, 0])
            # print(gcn_logits.shape)

        if self.use_pool:
            pooled_output = self.dropout(pooled_output)
            baseline_logits = self.classifier(pooled_output)

        if self.use_gcn and self.use_pool:
            logits = gcn_logits + baseline_logits
        elif self.use_gcn:
            logits = gcn_logits
        elif self.use_pool:
            logits = baseline_logits
        else:
            raise Exception

        reshaped_logits = logits.squeeze(-1).view(-1, num_choices)
        outputs = (reshaped_logits, )

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss, ) + outputs
        # print(outputs)
        # print(labels)
        # print(np.argmax(outputs[1].detach().cpu().numpy(), axis=1))
        return outputs


# 该网络结构为正常的多项选择模型
class DAGN_(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()
        '''
        #配置模型的参数矩阵
        print(self.classifier.weight.data)
        print(self.classifier.bias.data)

        self.init_weights()
        print(self.classifier.weight.data)
        print(self.classifier.bias.data)
        self.classifier.weight.data.normal_(mean = 0.0, std = 1.0)
        self.classifier.bias.data.zero_()
        print(self.classifier.weight.data)
        print(self.classifier.bias.data)
        '''

    def forward(
            self,
            input_ids = None,       #输入维度（[2, 4, 128]）, ([batch_size, 选项数， 文本长度])
            attention_mask = None,  #输入维度（[2, 4, 128]）, ([batch_size, 选项数， 文本长度])
            token_type_ids = None,  #输入维度（[2, 4, 128]）, ([batch_size, 选项数， 文本长度])
            labels = None,          #输入维度（[2]）, ([batch_size])
    ):

        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        output = (reshaped_logits, ) + outputs[2:]
        return ((loss, ) + output) if loss is not None else output