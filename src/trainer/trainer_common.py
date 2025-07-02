import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.distributed as dist
from datetime import datetime

import utils

from.trainer_base import TrainerBase


class TrainerCommon(TrainerBase):
    """
    This is a simple trainer for common tasks like image classification.
    """

    def __init__(self, datasets, agent, main_metric):
        super().__init__(datasets, agent)

        self.main_metric = main_metric

    def _reduce(self, result):
        if isinstance(result, pd.Series):  # for forward_loss
            object_list = [None] * self.world_size
            dist.all_gather_object(object_list, result)
            if self.rank == 0:
                result = sum(object_list) / self.world_size
            return result
        elif isinstance(result, pd.DataFrame):  # for forward_eval
            object_list = [None] * self.world_size
            dist.all_gather_object(object_list, result)
            if self.rank == 0:
                result = pd.concat(object_list, axis=0)
            return result
        else:
            raise NotImplementedError()

    def _train_n_iteration(self, train_loader, base_iteration, n, log_every):
        """
        Args:
            train_loader
            iteration: int
            n: int
            log_every: interval of recording training metrics(eg, loss) in tensorboard

        Return:
            log_dict, dict[str, numpy 1d array]
        """
        self.agent.train()

        loss_dict = 0
        grad_norm_list = []
        print_progress = utils.PrintProgress(desc='TRAIN: ', total=n)

        if self.log_dir is not None:
            self.writer.add_scalar('learning_rate', self.lr_scheduler.get_last_lr()[0], base_iteration)

        for i in range(n):
            # tensorbaord setting
            global_step = base_iteration + i
            record_tensorboard = (self.log_dir is not None) and (global_step % log_every) == 0

            agent = self.agent.module if self.distributed else self.agent
            agent.set_global_step(global_step)
            if self.rank == 0 and record_tensorboard:
                agent.set_tensorboard_writer(self.writer)
                agent.add_hook()

            # train
            batch = next(train_loader)
            loss, loss_log_temp = self.agent(batch, mode='loss')  # for distributed training, we should call DDP_Agent
            grad_norm = self._optimize(loss, max_norm=10)
            grad_norm_list.append(grad_norm)

            # logging
            loss_dict = loss_dict + loss_log_temp
            if self.rank == 0:
                print_progress(postfix=(loss_dict/(i+1)).to_dict())

                if record_tensorboard:
                    for key, value in loss_log_temp.items():
                        self.writer.add_scalar(f'loss/{key}', value, global_step)
                    agent.remove_all_hook()

        self.lr_scheduler.step()

        if self.log_dir is not None:
            grad_norm_list = np.array(grad_norm_list, dtype=np.float32)
            self.writer.add_histogram('grad_norm', grad_norm_list, base_iteration)

        loss_dict = loss_dict / (i+1)  # pd.Series to dict
        return loss_dict

    @torch.no_grad()
    def _evaluate(self, dataloader):
        agent = self.agent.module if self.distributed else self.agent
        agent.eval()

        results = []
        iteration_num = len(dataloader)
        if self.rank == 0:
            print_progress = utils.PrintProgress(desc='EVALUATE: ', total=iteration_num)

        loss_dict = 0
        for i, batch in enumerate(dataloader):
            logs = agent.forward_eval(batch)
            results.append(logs)

            if self.rank == 0:
                loss_dict = loss_dict + logs.mean()
                postfix = (loss_dict/(i+1)).to_dict()
                print_progress(postfix=postfix)

        results = pd.concat(results, axis=0)
        return results

    def train(
        self,
        train_batch_size,
        eval_batch_size,
        iteration_num,
        train_log_every,
        eval_every,
        evaluate_first
    ):
        if self.distributed:
            dist.barrier()

        train_loader = self._get_dataloader(self.datasets['train'], train_batch_size)

        epoch_num = iteration_num / len(train_loader)
        self.log(f'Start training {iteration_num} iteration({epoch_num:.2f} epoch).')

        train_loader = iter(utils.EndlessDataLoader(train_loader))
        eval_loader = self._get_dataloader(self.datasets['eval'], eval_batch_size)

        # START TRAINING
        start_time = datetime.now()
        best_iteration, best_result = -1, None
        for iteration in range(0, iteration_num + eval_every, eval_every):
            self.log('', no_prefix=True)  # empty line

            # train
            if iteration > 0:  # iteration 0 is used for evaluate_first
                self.log(f'{iteration - eval_every}-{iteration} / {iteration_num}')
                train_start_time = datetime.now()
                with torch.autocast('cuda', enabled=self.use_amp):
                    loss_dict = self._train_n_iteration(
                        train_loader,
                        base_iteration=1 + iteration - eval_every,
                        n=eval_every,
                        log_every=train_log_every)
                loss_dict = self._reduce(loss_dict) if self.distributed else loss_dict
                self.log(f'TRAIN time: {str(datetime.now() - train_start_time)}')
                self.log(f'losses\n{loss_dict}')

            # evaluate first
            if iteration == 0 and not evaluate_first:
                continue
            if iteration == 0:
                self.log(f'Evaluate First')

            # evaluate
            val_start_time = datetime.now()
            all_result = self._evaluate(eval_loader)
            all_result = self._reduce(all_result) if self.distributed else all_result
            metrics = all_result.mean()
            assert self.main_metric == 'none' or self.main_metric in metrics

            message = ', '.join(f'{key} {round(value, 4)}' for key, value in metrics.items())
            self.log(message)
            if self.rank == 0:
                if self.log_dir is not None:
                    for key, value in metrics.items():
                        self.writer.add_scalar(key, value, iteration)

                better = best_result is None \
                      or (metrics[self.main_metric] > best_result[self.main_metric])
                if self.main_metric != 'none' and better:
                    best_iteration = iteration
                    best_result = metrics
                    self.log('*** best up to now ***')
            self.log(f'EVALUATING time: {str(datetime.now() - val_start_time)}')

            # save model
            if self.rank == 0 and self.log_dir is not None:
                agent = self.agent.module if self.distributed else self.agent
                if self.main_metric == 'none':
                    agent.save(self.log_dir, iteration)
                elif iteration == best_iteration:
                    agent.save(self.log_dir)
            # end loop

        self.log(
            f'---finished training---\n'
            f'best iteration: {best_iteration}\n'
            f'best result:\n{best_result}')
        self.log(f'Time consuming: {str(datetime.now() - start_time)}')

