import argparse

import os
import yaml
import os.path as osp
from datetime import datetime

import torch
import torch.optim as optim
import torch.multiprocessing as mp

import utils
from config import device


def get_dataset(args):
    datasets = dict()

    if args.dataset == 'cifar10':
        from dataset.cifar10 import CIFAR10
        datasets['train'] = CIFAR10(train=True)
        datasets['eval'] = CIFAR10(train=False)
    elif args.dataset == 'imagenet2012':
        from dataset.imagenet import ImageNet2012
        datasets['train'] = ImageNet2012(train=True)
        datasets['eval'] = ImageNet2012(train=False)
    else:
        raise ValueError()

    return datasets


def get_agent(args):
    """
    import agent is slow, so seperately import each module
    """
    if args.agent == 'AgentClassification':
        from agent.agent_classifiaction import AgentClassification
        return AgentClassification(args)

    raise ValueError()


@utils.print_exception
def main(rank, world_size, args):
    utils.setup_seed(args.seed)
    print(f'RANK {rank}: Setup seed {args.seed}.')

    # set device for distributed training
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(rank)

    # prepare datasets
    datasets = get_dataset(args)

    # agents, used to adapt model to different datasets or training methods
    agent = get_agent(args)

    if args.load:
        print(f'RANK {rank}: Loading model from', args.load)
        agent.load(args.load, strict=not args.not_load_strict)
    agent.to(device)

    # create Trainer, use different trainer to support different task
    if args.trainer == 'TrainerCommon':
        from trainer.trainer_common import TrainerCommon
        trainer = TrainerCommon(datasets, agent, args.main_metric)

    if world_size > 1:
        trainer.set_distributed(rank, world_size)

    trainer.set_optimizer(optim.AdamW, lr=args.lr, weight_decay=1e-4)
    trainer.set_lrscheduler(
        max_step=args.iteration_num // args.eval_every,
        warmup_ratio=args.warmup_ratio,
        fn=args.lr_scheduler)

    if args.log:
        time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_dir = osp.join('log', args.method, f'{time}_{args.expr_name}')
        trainer.set_log_dir(log_dir)  # create log dir
        if rank == 0:
            with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
                yaml.dump(args.__dict__, f, allow_unicode=True)
            print('Configurations saved')

    # start training
    trainer.train(
        args.train_batch_size,
        args.eval_batch_size,
        args.iteration_num,
        args.train_log_every,
        args.eval_every,
        args.evaluate_first)
    # trainer.test()  # implement test() to save evalution result and submit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='distributed training: --gpu 0,1,2')
    parser.add_argument('--log', action='store_true', help='save training data or not')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--train_log_every', type=int, default=100)

    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--expr_name', type=str, required=True)
    parser.add_argument('--load', type=str, default='', help='path to the dir of checkpoint model')
    parser.add_argument('--not_load_strict', default=False, action='store_true')
    parser.add_argument('--evaluate_first', default=False, action='store_true')

    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--trainer', type=str, required=True)

    # training
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--eval_batch_size', type=int, required=True)
    parser.add_argument('--iteration_num', type=int, required=True)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--lr_scheduler', type=str, default='none', help='only support none,linear,cosine. You need to change the code.')
    parser.add_argument('--accumulate_grad', default=False, action='store_true', help='accumulate gradient during navigation')

    parser.add_argument('--loss_weight', type=int, default=0)
    parser.add_argument('--main_metric', type=str, required=True)

    args = parser.parse_args()
    assert args.main_metric in ('none', 'accuracy')
    assert (args.iteration_num % args.eval_every == 0)

    gpus = args.gpu.split(',')
    world_size = len(gpus)

    # START RUNNIG
    if not torch.cuda.is_available():
        print('Training with CPU')
        main(0, world_size, args)
    elif world_size == 1:
        print(f'Training with a single GPU cuda:{args.gpu}')
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        main(0, world_size, args)
    else:
        print('Training with GPUs', args.gpu)
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = str(utils.get_available_port())
            mp.spawn(main, (world_size, args), nprocs=world_size)
        except KeyboardInterrupt:  # kill all subprocess by ctrl+c
            for p in mp.active_children():
                p.kill()
            print('KeyboardInterrupt')
