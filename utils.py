import json
import logging
import os
from typing import List
import tqdm
import glob

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

class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "graph_nodes_a": graph_nodes_a, "graph_nodes_b": graph_nodes_b, "graph_edges_a": graph_edges_a, "graph_edges_b": graph_edges_b, "a_mask": a_mask, "b_mask": b_mask}
            for input_ids, input_mask, segment_ids, graph_nodes_a, graph_nodes_b, graph_edges_a, graph_edges_b, a_mask, b_mask in choices_features
        ]
        self.label = label

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

class LogiQAProcessor(DataProcessor):
    """ Processor for the LogiQA data set. """

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Train.txt")), "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Eval.txt")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_txt(os.path.join(data_dir, "Test.txt")), "test")

    def get_labels(self):
        return [0, 1, 2, 3]

    def _read_txt(self, input_file):
        with open(input_file, "r") as f:
            lines = f.readlines()
        return lines

    def _create_examples(self, lines, type):
        """ LogiQA: each 8 lines is one data point.
                The first line is blank line;
                The second is right choice;
                The third is context;
                The fourth is question;
                The remaining four lines are four options.
        """
        label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
        assert len(lines) % 8 ==0, 'len(lines)={}'.format(len(lines))
        n_examples = int(len(lines) / 8)
        examples = []
        # for i, line in enumerate(examples):
        for i in range(n_examples):
            label_str = lines[i*8+1].strip()
            context = lines[i*8+2].strip()
            question = lines[i*8+3].strip()
            answers = lines[i*8+4 : i*8+8]

            examples.append(
                InputExample(
                    example_id = " ",  # no example_id in LogiQA.
                    question = question,
                    contexts = [context, context, context, context],
                    endings = [item.strip()[2:] for item in answers],
                    label = label_map[label_str]
                )
            )
        assert len(examples) == n_examples
        return examples

class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples

def find_token_idx(token:str, bpe_token:list, start_idx:int):
    len_all = len(bpe_token)
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
    # len_a_ave = 0
    # len_a_max = 0
    # len_b_ave = 0
    # len_b_max = 0
    # len_a_min = 256
    # len_b_min = 256
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):

        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context

            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            # Preprocess the 'EXCEPT' in questions
            text_b.replace("EXCEPT:", ". EXCEPT .")
            # Preprocess the data
            text_a.replace(":", ".")
            text_b.replace(":", ".")
            text_a = text_a.lower().replace("cannot", "can not").replace("<b>", "").replace("</b>", "").replace("%.", "% .").replace(')', ') ').replace('""', '" "').replace('":', '" :')
            text_b = text_b.lower().replace("cannot", "can not").replace("<b>", "").replace("</b>", "").replace("%.", "% .").replace(')', ') ').replace('""', '" "').replace('":', '" :')
            text_a = text_a.replace("  ", " ").replace("__", "")
            text_b = text_b.replace("  ", " ").replace("__", "")
            text_a = text_a.replace("which of", "which one of").replace("each of", "each one of")
            text_b = text_b.replace("which of", "which one of").replace("each of", "each one of")

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
            # Remove special symbols 'Ġ'from words

            # Construct the graph structure of the article and the graph structure of the answer respectively
            start_ = 0
            count_ = 0
            compose_a = {}  #{id: (context, context trunk, (start, end)) }
            idx_a = {}      #{unique feature: id}
            List_a = []     #[unique feature]
            compose_b = {}
            idx_b = {}
            List_b = []

            doc_a = nlp(text_a)
            for token in doc_a:
                start_end, k = find_token_idx(token.text, bare_tokens, start_)
                start_ = k
                compose_a[count_] = (token.text, token.lemma_, start_end)
                idx_a[token.idx] = count_
                List_a.append(token.idx)
                count_ += 1

            doc_b = nlp(text_b)
            for token in doc_b:
                start_end, k = find_token_idx(token.text, bare_tokens, start_)
                start_ = k
                compose_b[count_] = (token.text, token.lemma_, start_end)
                idx_b[token.idx] = count_
                List_b.append(token.idx)
                count_ += 1

            stop_words_alter = []

            pattern1 = re.compile('.*subj') #subject
            pattern2 = re.compile('.*obj')  #object
            pattern3 = re.compile('.*mod')  #adjunct word
            pattern4 = re.compile('poss.*') #relationship

            context_edges_context = []  #the edges of context supergraph
            doc_a_root = []
            for sent in doc_a.sents:
                edges = []
                for token in sent:
                    if token.dep_ == 'ROOT':
                        doc_a_root.append(idx_a[token.idx])
                    for child in token.children:
                        edges.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[child.idx])))

                graph = nx.Graph(edges)

                noun1 = []  #subject
                noun2 = []  #object
                noun = ['NOUN', 'NUM', 'PRONP', 'PRON']
                for count in range(len(sent)):
                    token = sent[count]
                    #subject
                    if token.pos_ in noun and pattern1.match(token.dep_):
                        noun1.append(token.idx) # add all subjects
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                                context_edges_context.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[token.head.idx])))
                    #object
                    if token.pos_ in noun and pattern2.match(token.dep_):
                        noun2.append(token.idx) # add all objects
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_context.append(('{0}'.format(idx_a[token.head.idx]), '{0}'.format(idx_a[token.idx])))
                    if token.pos_ in noun and (token.head.pos_ == 'AUX'  and token.dep_ == 'attr'):
                        noun2.append(token.idx)
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_context.append(('{0}'.format(idx_a[token.head.idx]), '{0}'.format(idx_a[token.idx])))
                    #adjunct
                    if token.head.pos_ in noun and (token.dep_ == 'compound' or pattern3.match(token.dep_)):
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_context.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[token.head.idx])))
                    #relationship
                    if token.pos_ in noun and pattern4.match(token.dep_):
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_context.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[token.head.idx])))
                    #negative word
                    if token.dep_ == 'neg':
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_context.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[token.head.idx])))
                    #conjunction
                    if token.dep_ == 'conj':
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_context.append(('{0}'.format(idx_a[token.idx]), '{0}'.format(idx_a[token.head.idx])))

                for i in noun1:
                    for j in noun2:
                        entity1 = idx_a[i]
                        entity2 = idx_a[j]
                        path = nx.shortest_path(graph, source=str(entity1), target=str(entity2))
                        stop_path = []
                        for num in path:
                            if compose_a[int(num)][0].lower() not in stop_words_alter:
                                stop_path.append(num)
                        for num in range(len(stop_path) - 1):
                            context_edges_context.append(('{0}'.format(stop_path[num]), '{0}'.format(stop_path[num + 1])))

            for num in range(len(doc_a_root) - 1):
                context_edges_context.append(('{0}'.format(doc_a_root[num]), "{0}".format(doc_a_root[num + 1])))

            #the struct of context supergraph
            super_graph_context = nx.Graph(context_edges_context)
            for i in super_graph_context.nodes():
                for j in super_graph_context.nodes():
                    if compose_a[int(i)][1] == compose_a[int(j)][1] and i != j and compose_a[int(i)][1] not in STOP_WORDS:
                        super_graph_context.add_edge(i, j)

            context_edges_answer = []  #the edges of answer supergraph
            doc_b_root = []
            for sent in doc_b.sents:
                edges = []
                for token in sent:
                    if token.dep_ == 'ROOT':
                        doc_b_root.append(idx_b[token.idx])
                    for child in token.children:
                        edges.append(('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[child.idx])))

                graph = nx.Graph(edges)

                noun1 = []  # subject
                noun2 = []  # object

                noun = ['NOUN', 'NUM', 'PRONP', 'PRON']
                for count in range(len(sent)):
                    #subject
                    token = sent[count]
                    if token.pos_ in noun and pattern1.match(token.dep_):
                        noun1.append(token.idx) # add all subjects
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_answer.append(('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[token.head.idx])))
                    #object
                    if token.pos_ in noun and pattern2.match(token.dep_):
                        noun2.append(token.idx) # add all objects
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_answer.append(('{0}'.format(idx_b[token.head.idx]), '{0}'.format(idx_b[token.idx])))
                    if token.pos_ in noun and (token.head.pos_ == 'AUX'  and token.dep_ == 'attr'):
                        noun2.append(token.idx)
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_answer.append(('{0}'.format(idx_b[token.head.idx]), '{0}'.format(idx_b[token.idx])))
                    #adjunct
                    if token.head.pos_ in noun and (token.dep_ == 'compound' or pattern3.match(token.dep_)):
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_answer.append(('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[token.head.idx])))
                    #relationship
                    if token.pos_ in noun and pattern4.match(token.dep_):
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_answer.append(('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[token.head.idx])))
                    #negative word
                    if token.dep_ == 'neg':
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_answer.append(('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[token.head.idx])))
                    #conjunction
                    if token.dep_ == 'conj':
                        if token.text.lower() not in stop_words_alter and token.head.text.lower() not in stop_words_alter:
                            context_edges_answer.append(('{0}'.format(idx_b[token.idx]), '{0}'.format(idx_b[token.head.idx])))

                for i in noun1:
                    for j in noun2:
                        entity1 = idx_b[i]
                        entity2 = idx_b[j]
                        path = nx.shortest_path(graph, source=str(entity1), target=str(entity2))
                        stop_path = []
                        for num in path:
                            if compose_b[int(num)][0].lower() not in stop_words_alter:
                                stop_path.append(num)
                        for num in range(len(stop_path) - 1):
                            context_edges_answer.append(
                                ('{0}'.format(stop_path[num]), '{0}'.format(stop_path[num + 1])))

            for num in range(len(doc_b_root) - 1):
                context_edges_answer.append(('{0}'.format(doc_b_root[num]), "{0}".format(doc_b_root[num + 1])))

            #the struct of answer supergraph
            super_graph_answer = nx.Graph(context_edges_answer)
            for i in super_graph_answer.nodes():
                for j in super_graph_answer.nodes():
                    if compose_b[int(i)][1] == compose_b[int(j)][1] and i != j and compose_b[int(i)][1] not in STOP_WORDS:
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

            graph_nodes_a = []
            graph_nodes_b = []
            for i in super_graph_context.nodes():
                assert compose_a[int(i)][2][0] != compose_a[int(i)][2][1]
                graph_nodes_a.append((int(i), compose_a[int(i)][2][0], compose_a[int(i)][2][1]))
            for i in super_graph_answer.nodes():
                assert compose_b[int(i)][2][0] != compose_b[int(i)][2][1]
                graph_nodes_b.append((int(i), compose_b[int(i)][2][0], compose_b[int(i)][2][1]))

            # len_a_ave += len(graph_nodes_a)
            # len_a_max = max(len(graph_nodes_a), len_a_max)
            # len_a_min = min(len(graph_nodes_a), len_a_min)
            #
            # len_b_ave += len(graph_nodes_b)
            # len_b_max = max(len(graph_nodes_b), len_b_max)
            # len_b_min = min(len(graph_nodes_b), len_b_min)

            graph_nodes_a = graph_nodes_a + [(-1, -1, -1)] * (max_length - len(graph_nodes_a))
            graph_nodes_b = graph_nodes_b + [(-1, -1, -1)] * (max_length - len(graph_nodes_b))

            graph_edges_a = []
            graph_edges_b = []
            for i in super_graph_context.edges():
                graph_edges_a.append((int(i[0]), int(i[1])))
                graph_edges_a.append((int(i[1]), int(i[0])))
            for i in super_graph_answer.edges():
                graph_edges_b.append((int(i[0]), int(i[1])))
                graph_edges_b.append((int(i[1]), int(i[0])))

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

    # print("*" * 50)
    # print(len_a_ave)
    # print(len_a_max)
    # print(len_a_min)
    # print(len_b_ave)
    # print(len_b_max)
    # print(len_b_min)
    return features