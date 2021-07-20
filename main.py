
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
    BertPreTrainedModel,
    BertTokenizer,
    BertModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
    RobertaForMultipleChoice,
    get_linear_schedule_with_warmup,
)

#from utils import ReClorProcessor
from dagn import DAGN, DAGN_GCN
from utils import convert_examples_to_features, ReClorProcessor
# #用来进行可视化的
# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

# torch.backends.cudnn.enabled=False
#=====================================================================初始化=============================================
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()
#======================================================================加载数据===========================================
def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]

def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = ReClorProcessor()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    #保存的文件的名字
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
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
            pad_on_left=False,  # pad on the left for xlnet
            pad_token_segment_id=0,
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
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

#======================================================================模型配置===========================================
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
#====================================================================训练、验证、测试======================================
def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    # if args.local_rank in [-1, 0]:
    #     str_list = str(args.output_dir).split('/')
    #     tb_log_dir = os.path.join('summaries', str_list[-1])
    #     tb_writer = SummaryWriter(tb_log_dir)
    #     print(args.output_dir)

    #一、加载用于训练的数据，统计训练的轮次
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers = 4)

    # for n, train in enumerate(train_dataloader):
    #     print(n)
    #     print(train)
    #     break
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # for n, p in model.named_parameters():
    #     print(n, p.shape)     #打印名字、以及参数的形状

    #二、设置优化器以及优化器参数的变化
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    #对于bias以及LayerNorm.weight不进行权重衰减，对于其余的权重进行权重衰减
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    exec('args.adam_betas = ' + args.adam_betas)
    #更新梯度
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=args.adam_betas, eps=args.adam_epsilon)

    assert not ((args.warmup_steps > 0) and (args.warmup_proportion > 0)), "--only can set one of --warmup_steps and --warm_ratio "
    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    #更新优化器
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    #三、用于训练的方式以及是否使用fp16加速
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level = args.fp16_opt_level)

    #使用多GPU进行训练
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #使用分布式进行训练
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    #四、开始训练
    # Train!
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
        # tb_write.add_scalar("training/acc", train_acc, global_step)
        # for key, value in results.items():
        #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        if results["eval_acc"] > best_dev_acc:
            best_dev_acc = results["eval_acc"]
            best_steps = global_step
            logger.info("achieve BEST dev acc: %s at global step: %s", str(best_dev_acc), str(best_steps))

            #保存验证集效果最好的模型
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
                # tb_writer.add_text('best_results', rs, global_step)

        return train_preds, train_label_ids, train_acc, best_steps, best_dev_acc

    def save_model(args, model, tokenizer):
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model #hasattr用于判断对象是否包含对应的属性
        )
        model_to_save.save_pretrained(output_dir)  # 保存了config.json , pytorch_model.bin 文件
        tokenizer.save_vocabulary(output_dir)  # 保存了vocab.txt文件
        tokenizer.save_pretrained(output_dir)   # 保存了added_tokens.json文件，special_token_map.json文件, tokenizer_config.json文件, vocab.txt文件
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    train_preds = None
    train_label_ids = None

    model.zero_grad()   #将参数的梯度设置为0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # print(batch[0].shape)     #(2, 4, 128)    （batch_size, 4, 128）
            # print(batch[1].shape)     #(2, 4, 128)    （batch_size, 4, 128）
            # print(batch[2].shape)     #(2, 4, 128)    （batch_size, 4, 128）
            # print(batch[3].shape)     #(2, 4, 128)    （batch_size, 4, 128）
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
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
                #梯度裁剪
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            #进行gradient_accumulation_step
            if(step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                #打印日志
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training: #Only evaluate when single GPU ortherwise metrics may not average well
                        train_preds, train_labels_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step)
                    # tb_writer.add_scalar("training/lr", scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar("training/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logger.info(
                        "Average loss: %s, average acc: %s at global step: %s",
                        str((tr_loss - logging_loss) / args.logging_steps),
                        str(train_acc),
                        str(global_step),
                    )
                    logging_loss = tr_loss

                #保存模型
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, tokenizer)

            #达到训练最大步数， 跳出循环
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    #全部训练结束后，保存模型
    if args.local_rank in [-1, 0]:
        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, args, model, tokenizer, best_steps, best_dev_acc, global_step)
        save_model(args, model, tokenizer)
        # tb_writer.close()
    return global_step, tr_loss / max(global_step, 1), best_steps


def evaluate(args, model, tokenizer, prefix="", test = False):
    eval_task_names = (args.task_name, )
    eval_outputs_dirs = (args.output_dir, )

    results = {}
    #这个eval_task_names, eval_outputs_dirs 只有一个
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate = not test, test = test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler = eval_sampler, batch_size = args.eval_batch_size)

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
                    "token_type_ids":batch[2],
                    "labels":batch[3]
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

        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + "_eval_results.txt")

        #将结果写到文件中
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            if not test:
                for key in sorted(result.keys()):
                    logger.info(" %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    if test:
        return results, preds
    else:
        return results

def main():

    args = ArgParse()

    #Output文件夹是否已经存在
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
    #是否使用分布式训练
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
    #设置打印的日志信息
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
    #设置种子
    set_seed(args)

    print("Parameters:")
    print(args)

    processor = ReClorProcessor()
    label_list = processor.get_labels()
    num_labels  = len(label_list)

    #加载预训练模型以及词表
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
    # vocab = tokenizer.vocab     #字典类型表示{token:id}
    # ids_to_tokens = tokenizer.ids_to_tokens #列表类型表示[id]
    model = DAGN.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        vocab_file=args.tokenizer_name,
        use_gcn=args.use_gcn,
        use_pool=args.use_pool,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print(config)
    print(tokenizer)
    # print(model)
    # print(model.named_parameters())
    # for x, y in model.named_parameters():
    #     print(x, y.shape)
    #     if 'classifier' in x:
    #         print(x, y)
    #     if 'pooler' in x:
    #         print(x, y)
        #训练一个epoch证明了模型未使用初始化 比 使用初始化的效果好
        #模型参数初始化
        #attention.self.query.weight (1024, 1024)   #query, key, value
        #attention.self.query.bias (1024)
        #attention.output.dense.weight  (1024, 1024)
        #attention.output.dense.bias (1024)
        #attention.output.LayerNorm.weigth (1024)
        #attention.output.LayerNorm.bias (1024)
        # ...
        #pooler.dense.weight (1024, 1024)
        #pooler.dense.bias (1024)
        # tensor([[-0.0391, 0.0444, -0.0249, ..., 0.0010, 0.0244, 0.0079],
        #         [-0.0225, 0.0373, -0.0350, ..., -0.0122, 0.0424, 0.0095],
        #         [-0.0328, -0.0122, -0.0519, ..., -0.0761, 0.0065, -0.0129],
        #         ...,
        #         [-0.0033, 0.0137, 0.0128, ..., 0.0554, -0.0628, -0.0112],
        #         [-0.0333, 0.0165, -0.0234, ..., -0.0384, -0.0343, -0.0629],
        #         [0.0137, 0.0049, -0.0149, ..., 0.0111, 0.0091, 0.0217]],
        #        requires_grad=True)
        # tensor([-0.0365, -0.0285, 0.0037, ..., 0.0227, 0.0211, -0.0185],
        #        requires_grad=True)
        #classifier.weight (1, 1024)
        #classifier.bias (1)

        # 未使用参数初始化
        # tensor([[-0.0110, -0.0260, -0.0111, ..., -0.0266, 0.0110, -0.0238]],
        #        requires_grad=True)
        # tensor([0.0120], requires_grad=True)

        # 使用了参数初始化
        # tensor([[ 0.0091,  0.0265,  0.0128,  ..., -0.0071, -0.0009, -0.0012]],
        #        requires_grad=True)
        # tensor([0.], requires_grad=True)
    '''
    # test
    text = 'I am a lovely boy who likes playing football'
    tokenizer_out = tokenizer.tokenize(text)
    print(tokenizer_out)
    tokenizer_encode = tokenizer.encode(text)
    print(tokenizer_encode)

    vocab = tokenizer.vocab     #字典类型表示{token:id}
    ids_to_tokens = tokenizer.ids_to_tokens #列表类型表示[id]

    for i in tokenizer_encode:
        print(ids_to_tokens[i])
    '''


    if args.local_rank == 0:
        torch.distributed.barrier()     # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # 模型配置完成、开始进行训练
    #Training
    # 训练
    if args.do_train:
        #形成了DataTensor数据
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate = False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    results = {}
    #Evaluation
    # 评估
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.output_dir
        checkpoints = [args.output_dir]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = DAGN.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix = prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    #Test
    # 测试
    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            checkpoint_dir = os.path.join(args.output_dir)
        if args.evaluate_during_training:
            checkpoint_dir = os.path.join(args.output_dir)
        if best_steps:
            logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)

        model = DAGN.from_pretrained(checkpoint_dir)
        model.to(args.device)

        result, preds = evaluate(args, model, tokenizer, test = True)

        result = dict((k, v) for k, v in result.items())
        results.update(result)
        np.save(os.path.join(args.output_dir, "test_preds.npy" if args.output_dir is not None else "test_preds.npy"),preds)


if __name__ == "__main__":
    main()