
import json
import collections

import argparse
import logging

import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    BertModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    get_linear_schedule_with_warmup,
)

from eign import EIGN
from utils import convert_examples_to_features, ReClorProcessor, LogiQAProcessor, RaceProcessor
processors = {"reclor": ReClorProcessor, "logiqa": LogiQAProcessor, "race":RaceProcessor}
logger = logging.getLogger(__name__)
# torch.backends.cudnn.enabled=False
#=====================================================================init=============================================
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

#adversarial training
class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name = 'word_embeddings'):
        #emb_name should be replaced with the parameter name of embedding in the corresponding model
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
#======================================================================load data===========================================
def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    #the name of the saved file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_EIGN".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    save_file = True    #Is the data reprocessed
    if os.path.exists(cached_features_file) and not args.overwrite_cache and save_file:
        logger.info("Loading features from cached file %s", cached_features_file)
        print("load cache")
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    graph_nodes_a = torch.tensor(select_field(features, "graph_nodes_a"), dtype=torch.long)
    graph_nodes_b = torch.tensor(select_field(features, "graph_nodes_b"), dtype=torch.long)

    graph_edges_a = torch.tensor(select_field(features, "graph_edges_a"), dtype=torch.long)
    graph_edges_b = torch.tensor(select_field(features, "graph_edges_b"), dtype=torch.long)
    a_mask = torch.tensor(select_field(features, "a_mask"), dtype=torch.long)
    b_mask = torch.tensor(select_field(features, "b_mask"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, graph_nodes_a, graph_nodes_b, graph_edges_a, graph_edges_b, a_mask,
                            b_mask, all_label_ids)

    return dataset

#======================================================================Model configuration===========================================
def ArgParse():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type ",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. \
              Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_graph_size",
        default=64,
        type=int,
        help="The maximum size of graph of context and answer"
    )
    parser.add_argument("--do_fgm", action="store_true", help="Whether to run Adv-FGM training.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )


    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--adam_betas', default='(0.9, 0.999)', type=str, help='betas for Adam optimizer')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_clip_grad_norm", action="store_true", help="whether not to clip grad norm")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")


    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup over warmup ratios.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--use_gcn", action="store_true", default=False, help="Use GCN model for output")
    parser.add_argument("--use_pool", action="store_true", default=False, help="Use Pool for output")

    args = parser.parse_args()
    return args
#====================================================================train、dev、test======================================
def train(args, train_dataset, model, tokenizer):
    """Train the model"""

    # Adversarial training
    if args.do_fgm:
        fgm = FGM(model)

    #1.load training data, compute training rounds
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers = 0)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    #2.Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    exec('args.adam_betas = ' + args.adam_betas)
    #Update gradient
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=args.adam_betas, eps=args.adam_epsilon)

    assert not ((args.warmup_steps > 0) and (args.warmup_proportion > 0)), "--only can set one of --warmup_steps and --warm_ratio "
    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    #Update optimizer
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    #3.How to train and whether to use fp16 acceleration
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    #4.Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    def evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step):
        train_preds = np.argmax(train_preds, axis = 1)
        train_acc = simple_accuracy(train_preds, train_label_ids)
        train_preds = None
        train_label_ids = None
        results = evaluate(args, model, tokenizer)
        logger.info(
            "train acc: %s, dev acc: %s, loss: %s, global steps: %s",
            str(train_acc),
            str(results["eval_acc"]),
            str(results["eval_loss"]),
            str(global_step)
        )
        if results["eval_acc"] >= best_dev_acc:
            best_dev_acc = results["eval_acc"]
            best_steps = global_step
            logger.info("achieve BEST dev acc: %s at global step: %s", str(best_dev_acc), str(best_steps))

            #Save the model with the best validation set
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (model.module if hasattr(model, "module") else model)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_vocabulary(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            txt_dir = os.path.join(output_dir, 'best_dev_results.txt')
            with open(txt_dir, 'w') as f:
                rs = 'global_steps: {}; dev_acc: {}'.format(global_step, best_dev_acc)
                f.write(rs)

        return train_preds, train_label_ids, train_acc, best_steps, best_dev_acc

    def save_model(args, model, tokenizer):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(output_dir)  # save config.json , pytorch_model.bin files
        tokenizer.save_vocabulary(output_dir)  # save vocab.txt  file
        tokenizer.save_pretrained(output_dir)   # save added_tokens.json，special_token_map.json, tokenizer_config.json, vocab.txt files
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    train_preds = None
    train_label_ids = None

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "graph_nodes_a": batch[3],
                "graph_nodes_b": batch[4],
                "graph_edges_a": batch[5],
                "graph_edges_b": batch[6],
                "a_mask": batch[7],
                "b_mask": batch[8],
                "labels": batch[9],
            }
            outputs = model(**inputs)
            loss = outputs[0]
            logits = outputs[1]

            #work only gpu = 1
            if train_preds is None:
                train_preds = logits.detach().cpu().numpy()
                train_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                train_preds = np.append(train_preds, logits.detach().cpu().numpy(),axis=0)
                train_label_ids = np.append(train_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

            if args.n_gpu > 1:
                loss = loss.mean()  #mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                #Gradient clipping
                if not args.no_clip_grad_norm:
                    if not args.do_fgm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if args.do_fgm:
                fgm.attack()
                outputs_adv = model(**inputs)
                loss_adv =outputs_adv[0]
                loss_adv = loss_adv.mean() / args.gradient_accumulation_steps
                loss_adv.backward()
                fgm.restore()

            #gradient_accumulation_step
            if(step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                #logging
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training: #Only evaluate when single GPU ortherwise metrics may not average well
                        train_preds, train_labels_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step)
                    logger.info(
                        "Average loss: %s, average acc: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(train_acc),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                #save model
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, tokenizer)

            #Reach the maximum number of steps and jump out of the loop
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    #finish training and save model
    if args.local_rank in [-1, 0]:
        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step)
        save_model(args, model, tokenizer)
    return global_step, tr_loss / max(global_step, 1), best_steps

def evaluate(args, model, tokenizer, prefix="", test = False):
    eval_task_names = (args.task_name, )
    eval_outputs_dirs = (args.output_dir, )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate = not test, test = test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler = eval_sampler, batch_size = args.eval_batch_size, num_workers = 0)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size =  %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc = "Evaluating"):

            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "graph_nodes_a": batch[3],
                    "graph_nodes_b": batch[4],
                    "graph_edges_a": batch[5],
                    "graph_edges_b": batch[6],
                    "a_mask": batch[7],
                    "b_mask": batch[8],
                    "labels": batch[9],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]


                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis = 0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis = 0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis = 1)
        acc = simple_accuracy(preds, out_label_ids)

        result = {"eval_acc":acc, "eval_loss": eval_loss}
        results.update(result)

        #write results into files
        if not test:
            logger.info("***** Eval results {} *****".format(str(prefix)))
            txt_dir = os.path.join(args.output_dir, 'dev_result.txt')
            with open(txt_dir, 'w') as f:
                for key in sorted(result.keys()):
                    logger.info(" %s = %s", key, str(result[key]))
                    f.write("%s = %s\n" % (key, str(result[key])))
        elif test:
            logger.info("***** Test results {} *****".format(str(prefix)))
            if args.task_name == "reclor":
                txt_dir = os.path.join(args.output_dir, 'test_result.txt')
                with open(txt_dir, 'w') as f:
                    f.write("no mes")

            elif args.task_name == 'logiqa':
                txt_dir = os.path.join(args.output_dir, 'test_result.txt')
                with open(txt_dir, 'w') as f:
                    for key in sorted(result.keys()):
                        logger.info(" %s = %s", key, str(result[key]))
                        f.write("%s = %s\n" % (key, str(result[key])))

    if test:
        return results, preds
    else:
        return results

def main():

    args = ArgParse()

    #Judge whether output file exists or not
    if(
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.loacl_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup loggin
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels  = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     # Make sure only the first process in distributed training will download model & vocab

    config = RobertaConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = EIGN.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        max_graph_size=args.max_graph_size,
        use_gcn=args.use_gcn,
        use_pool=args.use_pool,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()     # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    #Training
    if args.do_train:
        #DataTensor data
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate = False)

        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    results = {}
    #Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.output_dir
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = EIGN.from_pretrained(
                checkpoint,
                from_tf=bool(".ckpt" in checkpoint),
                config=config,
                max_graph_size=args.max_graph_size,
                use_gcn=args.use_gcn,
                use_pool=args.use_pool,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix = prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    #Test
    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            checkpoint_dir = os.path.join(args.output_dir)
        if args.evaluate_during_training:
            checkpoint_dir = os.path.join(args.output_dir)
        if best_steps:
            logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
        model = EIGN.from_pretrained(
            checkpoint_dir,
            from_tf=bool(".ckpt" in checkpoint),
            config=config,
            max_graph_size=args.max_graph_size,
            use_gcn=args.use_gcn,
            use_pool=args.use_pool,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model.to(args.device)

        result, preds = evaluate(args, model, tokenizer, test = True)

        result = dict((k, v) for k, v in result.items())
        results.update(result)
        np.save(os.path.join(args.output_dir, "test_preds.npy" if args.output_dir is not None else "test_preds.npy"), preds)

if __name__ == "__main__":
    main()
