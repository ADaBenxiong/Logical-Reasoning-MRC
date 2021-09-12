
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np
from transformers import BertPreTrainedModel, RobertaModel
import collections
import json

def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, 'r', encoding = 'utf-8') as reader:
        encoder = json.load(reader)
    vocab = encoder
    return vocab

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def replace_masked_values(tensor, mask, replace_with):

    return tensor.masked_fill((1 - mask).bool(), replace_with)

class ArgumentGCN(nn.Module):
    def __init__(self, node_dim, iteration_steps = 1):
        super(ArgumentGCN, self).__init__()

        self.node_dim = node_dim    #1024
        self.iteration_steps = iteration_steps  #1

        self._node_weight_fc = torch.nn.Linear(node_dim, 1, bias=True)

        self._self_node_fc = torch.nn.Linear(node_dim, node_dim, bias=True)

        self._node_fc_argument = torch.nn.Linear(node_dim, node_dim, bias=False)

        self.fc_1 = torch.nn.Linear(node_dim, 1, bias = True)

    #(4, 128, 1024)
    def forward(self, node, node_mask, punctuation_graph):

        node_len = node.size(1) #divide into node_len segments

        diagmat = torch.diagflat(torch.ones(node.size(1), dtype = torch.long, device = node.device))

        diagmat = diagmat.unsqueeze(0).expand(node.size(0), -1, -1)

        dd_graph = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)

        punct_graph = dd_graph * punctuation_graph

        node_neighbor_num = punct_graph.sum(-1)

        node_neighbor_num_mask = (node_neighbor_num >= 1).long()

        node_neighbor_num = replace_masked_values(node_neighbor_num.float(), node_neighbor_num_mask, 1)

        for step in range(self.iteration_steps):
            #(4, 128)
            d_node_weight = torch.sigmoid(self._node_weight_fc(node)).squeeze(-1)
            #(4, 128, 1024)
            self_node_info = self._self_node_fc(node)

            #(4, 128, 1024)
            node_info_argument = self._node_fc_argument(node)
            #(4, 128)
            node_weight = replace_masked_values(d_node_weight.unsqueeze(1).expand(-1, node_len, -1), punctuation_graph, 0)
            node_info_argument = torch.matmul(node_weight, node_info_argument)

            agg_node_info = node_info_argument / node_neighbor_num.unsqueeze(-1)

            node = F.relu(self_node_info + agg_node_info)

        return node

class AttentionScore(nn.Module):
    """
    correlation_func = 1, sij = x1^Tx2
    correlation_func = 2, sij = (Wx1)D(Wx2)
    correlation_func = 3, sij = Relu(Wx1)DRelu(Wx2)
    correlation_func = 4, sij = x1^TWx2
    correlation_func = 5, sij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, input_size, hidden_size, correlation_func=1, do_similarity=False):
        super(AttentionScore, self).__init__()
        self.correlation_func = correlation_func
        self.hidden_size = hidden_size

        self.dropout = nn.Dropout(0.1)

        if correlation_func == 2 or correlation_func == 3:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)
            if do_similarity:
                self.diagonal = nn.Parameter(torch.ones(1, 1, 1) / (hidden_size ** 0.5), requires_grad=False)
            else:
                self.diagonal = nn.Parameter(torch.ones(1, 1, hidden_size), requires_grad=True)

        if correlation_func == 4:
            self.linear = nn.Linear(input_size, input_size, bias=False)

        if correlation_func == 5:
            self.linear = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, x1, x2):
        '''
        Input:
        x1: batch x word_num1 x dim
        x2: batch x word_num2 x dim
        Output:
        scores: batch x word_num1 x word_num2
        '''

        x1_rep = x1
        x2_rep = x2
        batch = x1_rep.size(0)
        word_num1 = x1_rep.size(1)
        word_num2 = x2_rep.size(1)
        dim = x1_rep.size(2)
        if self.correlation_func == 2 or self.correlation_func == 3:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            if self.correlation_func == 3:
                x1_rep = F.relu(x1_rep)
                x2_rep = F.relu(x2_rep)
            x1_rep = x1_rep * self.diagonal.expand_as(x1_rep)
            # x1_rep is (Wx1)D or Relu(Wx1)D
            # x1_rep: batch x word_num1 x dim (corr=1) or hidden_size (corr=2,3)

        if self.correlation_func == 4:
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, dim)  # Wx2

        if self.correlation_func == 5:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)

        scores = x1_rep.bmm(x2_rep.transpose(1, 2))
        return scores

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, correlation_func=1, do_similarity=False):
        super(Attention, self).__init__()
        self.scoring = AttentionScore(input_size, hidden_size, correlation_func, do_similarity)

    def forward(self, x1, x2, x2_mask, x3=None, drop_diagonal=False):
        '''
        For each word in x1, get its attended linear combination of x3 (if none, x2),
         using scores calculated between x1 and x2.
        Input:
         x1: batch x word_num1 x dim
         x2: batch x word_num2 x dim
         x2_mask: batch x word_num2
         x3 (if not None) : batch x word_num2 x dim_3
        Output:
         attended: batch x word_num1 x dim_3
        '''
        batch = x1.size(0)
        word_num1 = x1.size(1)
        word_num2 = x2.size(1)

        if x3 is None:
            x3 = x2

        scores = self.scoring(x1, x2)

        # scores: batch x word_num1 x word_num2
        empty_mask = x2_mask.eq(0).unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(empty_mask.data, -100)    #问题

        if drop_diagonal:
            assert (scores.size(1) == scores.size(2))
            diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
            scores.data.masked_fill_(diag_mask, -float('inf'))

        # softmax
        alpha_flat = F.softmax(scores.view(-1, x2.size(1)), dim=1)
        alpha = alpha_flat.view(-1, x1.size(1), x2.size(1))
        # alpha: batch x word_num1 x word_num2

        attended = alpha.bmm(x3)
        # attended: batch x word_num1 x dim_3

        return attended

class EIGN(BertPreTrainedModel):
    def __init__(self,
                 config,
                 max_graph_size=128,
                 use_gcn=False,
                 use_pool=False):
        super().__init__(config)
        self.use_gcn = use_gcn
        self.use_pool = use_pool
        self.hidden_size = config.hidden_size
        self.dropout_prob = config.hidden_dropout_prob
        self.max_graph_size = max_graph_size

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self._gcn_a = ArgumentGCN(node_dim=self.hidden_size, iteration_steps=2)
        self._gcn_b = ArgumentGCN(node_dim=self.hidden_size, iteration_steps=2)

        self.fc = torch.nn.Linear(self.max_graph_size * 2, 1, bias = True)

        self.attention_1 = Attention(self.hidden_size, self.hidden_size, 3, True)
        self.attention_2 = Attention(self.hidden_size, self.hidden_size, 3, True)

        self.init_weights()

    def forward(self,
                input_ids = None,
                attention_mask = None,
                token_type_ids = None,
                graph_nodes_a = None,
                graph_nodes_b = None,
                graph_edges_a = None,
                graph_edges_b = None,
                a_mask = None,
                b_mask = None,
                labels = None
    ):
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        graph_nodes_a = graph_nodes_a.view(-1, graph_nodes_a.size(-2), graph_nodes_a.size(-1)) if graph_nodes_a is not None else None   #(4 * batch_size, 64, 3)
        graph_nodes_b = graph_nodes_b.view(-1, graph_nodes_b.size(-2), graph_nodes_b.size(-1)) if graph_nodes_b is not None else None
        graph_edges_a = graph_edges_a.view(-1, graph_edges_a.size(-2), graph_edges_a.size(-1)) if graph_edges_a is not None else None   #(4 * batch_size, 64, 2)
        graph_edges_b = graph_edges_b.view(-1, graph_edges_b.size(-2), graph_edges_b.size(-1)) if graph_edges_b is not None else None
        a_mask = a_mask.view(-1, a_mask.size(-1)) if a_mask is not None else None
        b_mask = b_mask.view(-1, b_mask.size(-1)) if b_mask is not None else None

        outputs = self.roberta(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = None
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        Length = input_ids.size(1)
        max_length = self.max_graph_size

        if self.use_gcn:

            encoded_spans_a = []    #encoded of node
            encoded_spans_b = []

            masked_spans_a = []     #masked of node
            masked_spans_b = []

            item_graph_a = torch.zeros((input_ids.size(0), max_length, max_length))
            item_graph_b = torch.zeros((input_ids.size(0), max_length, max_length))

            for num in range(input_ids.size(0)):
                len_nodes_a = (graph_nodes_a[num][:, 0] >= 0).sum()
                len_nodes_b = (graph_nodes_b[num][:, 0] >= 0).sum()
                len_edges_a = (graph_edges_a[num][:, 0] >= 0).sum()
                len_edges_b = (graph_edges_b[num][:, 0] >= 0).sum()

                item_spans_a = []
                item_spans_b = []
                for item in range(min(max_length, len_nodes_a.item())):
                    if graph_nodes_a[num][item][1] < Length and graph_nodes_a[num][item][2] < Length:
                        span = sequence_output[num, graph_nodes_a[num][item][1]: graph_nodes_a[num][item][2]]
                        item_spans_a.append(span.sum(0) / span.size(0))
                    else:
                        pad_embed = torch.zeros(self.hidden_size, dtype=sequence_output.dtype, device=sequence_output.device)
                        item_spans_a.append(pad_embed)
                encoded_spans_a.append(item_spans_a)
                masked_span = [1] * len(item_spans_a) + [0] * (max_length - len(item_spans_a))
                masked_spans_a.append(masked_span)

                for item in range(min(max_length, len_nodes_b.item())):
                    if graph_nodes_b[num][item][1] < Length and graph_nodes_b[num][item][2] < Length:
                        span = sequence_output[num, graph_nodes_b[num][item][1]: graph_nodes_b[num][item][2]]
                        item_spans_b.append(span.sum(0) / span.size(0))
                    else:
                        pad_embed = torch.zeros(self.hidden_size, dtype=sequence_output.dtype, device=sequence_output.device)
                        item_spans_b.append(pad_embed)

                encoded_spans_b.append(item_spans_b)
                masked_span = [1] * len(item_spans_b) + [0] * (max_length - len(item_spans_b))
                masked_spans_b.append(masked_span)

                node_convert_a = {}
                for item in range(min(max_length, len_nodes_a.item())):
                    node_convert_a[graph_nodes_a[num][item][0].item()] = item

                for item in range(len_edges_a.item()):
                    u = graph_edges_a[num][item][0].item()
                    v = graph_edges_a[num][item][1].item()
                    if u in node_convert_a.keys() and v in node_convert_a.keys():
                        item_graph_a[num][node_convert_a[u]][node_convert_a[v]] = 1

                node_convert_b = {}
                for item in range(min(max_length, len_nodes_b.item())):
                    node_convert_b[graph_nodes_b[num][item][0].item()] = item

                for item in range(len_edges_b.item()):
                    u = graph_edges_b[num][item][0].item()
                    v = graph_edges_b[num][item][1].item()
                    if u in node_convert_b.keys() and v in node_convert_b.keys():
                        item_graph_b[num][node_convert_b[u]][node_convert_b[v]] = 1

            #process of node
            pad_embed = torch.zeros(self.hidden_size, dtype=sequence_output.dtype, device=sequence_output.device)
            encoded_spans_a = [spans + [pad_embed] * (max_length - len(spans)) for spans in encoded_spans_a]
            encoded_spans_a = [torch.stack(lst, dim=0) for lst in encoded_spans_a]
            encoded_spans_a = torch.stack(encoded_spans_a, dim=0)
            encoded_spans_a = encoded_spans_a.to(sequence_output.device).float()
            masked_spans_a = [torch.Tensor(lst) for lst in masked_spans_a]
            masked_spans_a = torch.stack(masked_spans_a, dim=0)
            masked_spans_a = masked_spans_a.to(sequence_output.device)

            encoded_spans_b = [spans + [pad_embed] * (max_length - len(spans)) for spans in encoded_spans_b]
            encoded_spans_b = [torch.stack(lst, dim=0) for lst in encoded_spans_b]
            encoded_spans_b = torch.stack(encoded_spans_b, dim=0)
            encoded_spans_b = encoded_spans_b.to(sequence_output.device).float()
            masked_spans_b = [torch.Tensor(lst) for lst in masked_spans_b]
            masked_spans_b = torch.stack(masked_spans_b, dim=0)
            masked_spans_b = masked_spans_b.to(sequence_output.device)

            #process of graph
            item_graph_a = item_graph_a.to(sequence_output.device)
            item_graph_b = item_graph_b.to(sequence_output.device)

            #(4, 128, 1024)
            node_a_raw = self._gcn_a(node=encoded_spans_a, node_mask=masked_spans_a, punctuation_graph=item_graph_a)
            node_b_raw = self._gcn_b(node=encoded_spans_b, node_mask=masked_spans_b, punctuation_graph=item_graph_b)

            #bi-directional attention mechanism
            node_a_to_b = self.attention_1(node_a_raw, node_b_raw, masked_spans_b)
            node_b_to_a = self.attention_2(node_b_raw, node_a_raw, masked_spans_a)

            node = torch.cat((node_a_to_b[:, :, 0], node_b_to_a[:, :, 0]), dim=1)

            node_1 = self.fc(node)
            gcn_logits = node_1.view(-1, num_choices)

        if self.use_pool:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            reshaped_logits = logits.view(-1, num_choices)

        if self.use_pool and not self.use_gcn:
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(reshaped_logits, labels)

            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        elif self.use_gcn and not self.use_pool:
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(gcn_logits, labels)

            output = (gcn_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        elif self.use_gcn and self.use_pool:
            loss = None
            logits = reshaped_logits + gcn_logits
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)

            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


