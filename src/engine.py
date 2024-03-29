import os
import sys
import logging

import torch
from torch import distributed as dist
from torch.cuda import amp as amp
from torch import nn
from torch.utils import data as torch_data

from torchdrug import data, core, utils
from torchdrug.utils import comm, pretty
import torch.nn.functional as F

module = sys.modules[__name__]
logger = logging.getLogger(__name__)



class Engine(core.Configurable):
    """
    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multiprocess case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        num_worker (int, optional): number of CPU workers per GPU
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
        log_interval (int, optional): log every n gradient updates
        half_precision(bool, optional): use the half precision mode

    """

    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, batch_size=1,
                 num_worker=0, logger="logging", log_interval=100, half_precision=False, criterion=None):
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.half_precision = half_precision

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            if self.rank == 0:
                module.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            if self.rank == 0:
                module.logger.warning("Preprocess training set")
            # TODO: more elegant implementation
            # handle dynamic parameters in optimizer
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params):]})

        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        if criterion == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
            self.criterion_name = "Binary Cross Entropy Loss"
        elif criterion == "ce":
            self.criterion = nn.CrossEntropyLoss()
            self.criterion_name = "Cross Entropy Loss"
        else:
            raise ValueError("Not Support Loss")

        if self.half_precision:
            self.scaler = amp.GradScaler(enabled=True)

        if isinstance(logger, str):
            if logger == "logging":
                logger = core.LoggingLogger()
            elif logger == "wandb":
                logger = core.WandbLogger(project=task.__class__.__name__)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(log_interval=log_interval, silent=self.rank > 0, logger=logger)
        self.meter.log_config(self.config_dict())
        module.logger.warning("Mix Precison Training:{}".format(self.half_precision))


    def train(self, num_epoch=1):
        """
        Parameters:
            num_epoch (int, optional): number of epochs
        """
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        model = self.model

        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                                 find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        
        model.train()

        for epoch in self.meter(num_epoch):
            sampler.set_epoch(epoch)
            losses = []

            for i, batch in enumerate(dataloader):
                if self.device.type == "cuda":
                    batch = utils.cuda(batch, device=self.device)
                
                # DDP warp the original model so use the `self.model` to get label 
                target = self.model.target(batch)
                
                with torch.cuda.amp.autocast(enabled=self.half_precision):
                    # TODO: Be Sure to Use `model` (Not `Self.Model`) 
                    pred = model(batch)
                    loss = self.criterion(pred, target)

                if self.half_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    loss.backward()
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                # Here We Can Add Self-Designed Loss to Update the Model
                # Key: Loss Name (Support Multiple Loss Record)
                cur_loss = {self.criterion_name: loss}
                losses.append(cur_loss)

                cur_loss = utils.stack(losses, dim=0)
                cur_loss = utils.mean(cur_loss, dim=0)

                if self.world_size > 1:
                    cur_loss = comm.reduce(cur_loss, op="mean")
                self.meter.update(cur_loss)

                losses = []

            if self.scheduler:
                self.scheduler.step()

    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not
        Returns:
            dict: metrics
        """
        if comm.get_rank() == 0:
            logger.warning(pretty.separator)
            logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)

        model = self.model
        model.eval()

        preds = []
        targets = []
        for i, batch in enumerate(dataloader):
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)
                
            target = model.target(batch)    
            pred = model(batch)
            
            preds.append(pred)
            targets.append(target)

        pred = utils.cat(preds)
        target = utils.cat(targets)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)

        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch" % split)
        return metric

    def load(self, checkpoint, load_optimizer=True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
        """
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        self.model.load_state_dict(state["model"])

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()

    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        if comm.get_rank() == 0:
            logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()

    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id
