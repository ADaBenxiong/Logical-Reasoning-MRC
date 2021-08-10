import csv
import json
import logging
import os
from typing import List
import tqdm
import collections

from transformers import PreTrainedTokenizer, BertTokenizer

logger = logging.getLogger(__name__)

punctuations = [',', '.', ';', ':', '?', '!']
separator = ['CLS', '[SEP]']

import spacy
import re
import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS
import re
import networkx as nx
nlp = spacy.load('en_core_web_sm')

# #获取bert模型的词表
# def Vocab_text():
#     tokenizer = BertTokenizer.from_pretrained(
#         args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
#         do_lower_case=args.do_lower_case,
#         cache_dir=args.cache_dir if args.cache_dir else None,
#     )
#     vocab = tokenizer.vocab  # 字典类型表示{token:id}
#     ids_to_tokens = tokenizer.ids_to_tokens  # 列表类型表示[id]
#     return vocab, ids_to_tokens
def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, 'r', encoding = 'utf-8') as reader:
        encoder = json.load(reader)
    vocab = encoder
    return vocab

def read_explict(explict_arg_file):
    with open(explict_arg_file, 'r', encoding = 'utf-8') as reader:
        text = reader.read()
    text = json.loads(text)
    return [idx for idx in text]

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label

# class InputFeatures(object):
#     def __init__(self, example_id, choices_features, label):
#         self.example_id = example_id
#         self.choices_features = [
#             {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "compose_unit": compose_unit, "graph_a": graph_a, "graph_b": graph_b, "a_mask": a_mask, "b_mask": b_mask}
#             for input_ids, input_mask, segment_ids, compose_unit, graph_a, graph_b, a_mask, b_mask in choices_features
#         ]
#         self.label = label

class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "graph_nodes_a": graph_nodes_a, "graph_nodes_b": graph_nodes_b, "graph_edges_a": graph_edges_a, "graph_edges_b": graph_edges_b, "a_mask": a_mask, "b_mask": b_mask}
            for input_ids, input_mask, segment_ids, graph_nodes_a, graph_nodes_b, graph_edges_a, graph_edges_b, a_mask, b_mask in choices_features
        ]
        self.label = label
# class InputFeatures(object):
#     def __init__(self, example_id, choices_features, label):
#         self.example_id = example_id
#         self.choices_features = [
#             {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "span_masks": span_masks, "node_in_seq": node_in_seq}
#             for input_ids, input_mask, segment_ids, span_masks, node_in_seq in choices_features
#         ]
#         self.label = label

class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class ReClorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""
        examples = []
        for d in lines:
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label
                    )
                )
        return examples

#max_rel_id = 4
def find_token_idx(token:str, bpe_token:list, start_idx:int):
    len_all = len(bpe_token)
    # print(start_idx)
    # print(len_all)
    for i in range(start_idx, min(len_all, start_idx + 3)): #最多向后找3个(例如 cannot can not)
        len_start = len(bpe_token[i])
        if token[:len_start].lower() == bpe_token[i]:
            for j in range(min(len_all - i, 10)):   #最多由10个拼接而成
                token_start = "".join(bpe_token[i:i+j])
                if token.lower() == token_start:
                    return (i, i+j), i+j

    return (start_idx, start_idx + 1), start_idx


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):

        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            text_a = text_a.lower().replace("cannot", "can not").replace("<b>", "").replace("</b>", "").replace("%.", "% .").replace(')', ') ').replace('""', '" "').replace('":', '" :')
            text_b = text_b.lower().replace("cannot", "can not").replace("<b>", "").replace("</b>", "").replace("%.", "% .").replace(')', ') ').replace('""', '" "').replace('":', '" :')
            text_a = text_a.replace("  ", " ").replace("__", "")
            text_b = text_b.replace("  ", " ").replace("__", "")
            bpe_tokens_a = tokenizer.tokenize(text_a)

            bpe_tokens_b = tokenizer.tokenize(text_b)

            bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                         bpe_tokens_b + [tokenizer.eos_token]

            a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))

            b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (
                        max_length - len(bpe_tokens))

            a_mask = a_mask[:max_length]

            b_mask = b_mask[:max_length]

            assert isinstance(bpe_tokens, list)
            bare_tokens = [token[1:] if "Ġ" in token else token for token in bpe_tokens]
            # print(bare_tokens)  # 去除单词特殊符号G
            # print(len(bare_tokens))

            #分别刻画文章的图结构和答案的图结构
            start_ = 0
            count_ = 0
            compose_a = {}  #{编号：（文本、文本主干、开始结束的位置）}
            idx_a = {}      #{唯一特征：编号}
            List_a = []     #[唯一特征]
            compose_b = {}
            idx_b = {}
            List_b = []

            doc_a = nlp(text_a)
            for token in doc_a:
                # print(text_a)
                # print(token.text)
                # print(bare_tokens)
                # print(len(bare_tokens))
                # print(start_)
                # try:
                start_end, k = find_token_idx(token.text, bare_tokens, start_)
                # except TypeError:
                #     print(text_a)
                #     print(token.text)
                #     print(bare_tokens)
                #     print(len(bare_tokens))
                #     print(start_)
                start_ = k
                compose_a[count_] = (token.text, token.lemma_, start_end)
                idx_a[token.idx] = count_
                List_a.append(token.idx)
                count_ += 1

            #count_ = 0
            doc_b = nlp(text_b)
            for token in doc_b:
                start_end, k = find_token_idx(token.text, bare_tokens, start_)

                start_ = k
                compose_b[count_] = (token.text, token.lemma_, start_end)
                idx_b[token.idx] = count_
                List_b.append(token.idx)
                count_ += 1


            pattern1 = re.compile('.*subj')
            pattern2 = re.compile('.*obj')

            context_edges_context = []  #超图的结构
            for sent in doc_a.sents:
                edges = []
                for token in sent:
                    for child in token.children:
                        edges.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[child.idx])))

                graph = nx.Graph(edges)

                noun1 = []  #主语
                noun2 = []  #宾语
                for token in sent:
                    if token.pos_ == 'NOUN' and pattern1.match(token.dep_):
                        noun1.append(token.idx)
                        if token.text.lower() not in STOP_WORDS and token.head.text.lower() not in STOP_WORDS:
                            context_edges_context.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[token.head.idx])))
                    if token.pos_ == 'NOUN' and pattern2.match(token.dep_):
                        noun2.append(token.idx)
                        if token.text.lower() not in STOP_WORDS and token.head.text.lower() not in STOP_WORDS:
                            context_edges_context.append(('{0}'.format(idx_a[token.head.idx]), '{0}'.format(idx_a[token.idx])))

                for i in noun1:
                    for j in noun2:
                        entity1 = idx_a[i]
                        entity2 = idx_a[j]
                        path = nx.shortest_path(graph, source=str(entity1), target=str(entity2))
                        # print(nx.shortest_path(graph, source=str(entity1), target=str(entity2)))
                        stop_path = []
                        for num in path:
                            if compose_a[int(num)][0].lower() not in STOP_WORDS:
                                stop_path.append(num)
                        for num in range(len(stop_path) - 1):
                            context_edges_context.append(('{0}'.format(stop_path[num]), '{0}'.format(stop_path[num + 1])))

            super_graph_context = nx.DiGraph(context_edges_context)
            for i in super_graph_context.nodes():
                for j in super_graph_context.nodes():
                    if compose_a[int(i)][1] == compose_a[int(j)][1] and i != j:
                        super_graph_context.add_edge(i, j)

            context_edges_answer = []  # 超图的结构
            for sent in doc_b.sents:
                edges = []
                for token in sent:
                    for child in token.children:
                        edges.append(('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[child.idx])))

                graph = nx.Graph(edges)

                noun1 = []  # 主语
                noun2 = []  # 宾语
                for token in sent:
                    if token.pos_ == 'NOUN' and pattern1.match(token.dep_):
                        noun1.append(token.idx)
                        if token.text.lower() not in STOP_WORDS and token.head.text.lower() not in STOP_WORDS:
                            context_edges_answer.append(
                                ('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[token.head.idx])))
                    if token.pos_ == 'NOUN' and pattern2.match(token.dep_):
                        noun2.append(token.idx)
                        if token.text.lower() not in STOP_WORDS and token.head.text.lower() not in STOP_WORDS:
                            context_edges_answer.append(
                                ('{0}'.format(idx_b[token.head.idx]), '{0}'.format(idx_b[token.idx])))

                for i in noun1:
                    for j in noun2:
                        entity1 = idx_b[i]
                        entity2 = idx_b[j]
                        path = nx.shortest_path(graph, source=str(entity1), target=str(entity2))
                        # print(nx.shortest_path(graph, source=str(entity1), target=str(entity2)))
                        stop_path = []
                        for num in path:
                            if compose_b[int(num)][0].lower() not in STOP_WORDS:
                                stop_path.append(num)
                        for num in range(len(stop_path) - 1):
                            context_edges_answer.append(
                                ('{0}'.format(stop_path[num]), '{0}'.format(stop_path[num + 1])))

            super_graph_answer = nx.DiGraph(context_edges_answer)
            for i in super_graph_answer.nodes():
                for j in super_graph_answer.nodes():
                    if compose_b[int(i)][1] == compose_b[int(j)][1] and i != j:
                        super_graph_answer.add_edge(i, j)

            input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
            attention_mask = [1] * len(input_ids)
            token_type_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)

            padding = [0] * (max_length - len(input_ids))
            input_ids += padding
            attention_mask += padding
            token_type_ids += padding

            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

            # print(super_graph_context)
            # print(super_graph_context.nodes())
            # print(super_graph_context.edges())
            # print(super_graph_answer)
            # print(super_graph_answer.nodes())
            # print(super_graph_answer.edges())

            graph_nodes_a = []
            graph_nodes_b = []
            for i in super_graph_context.nodes():
                assert compose_a[int(i)][2][0] != compose_a[int(i)][2][1]
                graph_nodes_a.append((int(i), compose_a[int(i)][2][0], compose_a[int(i)][2][1]))
            for i in super_graph_answer.nodes():
                assert compose_b[int(i)][2][0] != compose_b[int(i)][2][1]
                graph_nodes_b.append((int(i), compose_b[int(i)][2][0], compose_b[int(i)][2][1]))

            graph_nodes_a = graph_nodes_a + [(-1, -1, -1)] * (max_length - len(graph_nodes_a))

            graph_nodes_b = graph_nodes_b + [(-1, -1, -1)] * (max_length - len(graph_nodes_b))


            graph_edges_a = []
            graph_edges_b = []
            for i in super_graph_context.edges():
                graph_edges_a.append((int(i[0]), int(i[1])))
            for i in super_graph_answer.edges():
                graph_edges_b.append((int(i[0]), int(i[1])))

            graph_edges_a = graph_edges_a + [(-1, -1)] * (max_length - len(graph_edges_a))
            graph_edges_b = graph_edges_b + [(-1, -1)] * (max_length - len(graph_edges_b))

            graph_nodes_a = graph_nodes_a[:max_length]
            graph_nodes_b = graph_nodes_b[:max_length]
            graph_edges_a = graph_edges_a[:max_length]
            graph_edges_b = graph_edges_b[:max_length]

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            assert len(graph_nodes_a) == max_length
            assert len(graph_nodes_b) == max_length
            assert len(graph_edges_a) == max_length
            assert len(graph_edges_b) == max_length
            assert len(a_mask) == max_length
            assert len(b_mask) == max_length

            choices_features.append(
                (input_ids, attention_mask, token_type_ids, graph_nodes_a, graph_nodes_b, graph_edges_a, graph_edges_b, a_mask, b_mask))

        label = label_map[example.label]

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label, ))

    return features






















