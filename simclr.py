import logging
import os
import sys
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils import save_config_file, accuracy, save_checkpoint
from evaluate import Evaluator

root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logging.getLogger('matplotlib.font_manager').disabled = True
NORMALIZE_B4_COMPACTNESS = False


class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.neptune_run = kwargs['neptune_run']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.positive_indices = torch.as_tensor(kwargs['positive_indices'])
        self.params = kwargs['params']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.compactness_criterion = torch.nn.MSELoss().to(self.args.device)
        self.evaluator = Evaluator(self.neptune_run, self.args, self.model, logging, params=self.params)
        self.train_labeled_mean = None

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.params['batch_size']) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)
        logits = logits / self.params['temperature']
        return logits, labels

    def compactness_loss_info(self, features, indices):
        if NORMALIZE_B4_COMPACTNESS:
            features = F.normalize(features, dim=1)
        rel_indices = (indices[:, None] == self.positive_indices[None,:]).any(-1)
        rel_features: torch.Tensor = features[torch.cat([rel_indices, rel_indices])].to(self.args.device)
        meaned_rel_features = rel_features - self.train_labeled_mean.to(self.args.device)

        return meaned_rel_features

    def train(self, train_loader, test_loader, train_labeled_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        n_iter = 0
        self.train_labeled_mean = torch.tensor(self.evaluator.evaluate(test_loader, train_labeled_loader))
        logging.info(f"Start SimCLR training for {self.params['epochs']} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        for epoch_counter in range(self.params['epochs']):
            print(f"epoch {epoch_counter}")
            self.neptune_run['cur_epoch'].log(epoch_counter)
            for images, classes, indices in train_loader:
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    constructive_loss = self.criterion(logits, labels)
                    zero_mean_labeled = self.compactness_loss_info(features, indices)
                    compactness_loss = self.params['lambda'] * self.compactness_criterion(zero_mean_labeled,
                                                                                   torch.zeros_like(zero_mean_labeled))
                    # convert nan loss to 0, when there are no samples to make compact
                    compactness_loss[compactness_loss != compactness_loss] = 0

                self.optimizer.zero_grad()
                scaler.scale(constructive_loss + compactness_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.neptune_run['losses/constructive_loss'].log(constructive_loss)
                    self.neptune_run['losses/compactness_loss'].log(compactness_loss)
                    self.neptune_run['losses/total_loss'].log(constructive_loss + compactness_loss)
                    self.neptune_run['acc/top1'].log(top1[0])
                    self.neptune_run['acc/top5'].log(top5[0])
                    self.neptune_run['losses/learning_rate'].log(self.scheduler.get_last_lr()[0])
                n_iter += 1
            logging.info(f"evaluating on epoch {epoch_counter + 1}")
            self.train_labeled_mean = torch.tensor(self.evaluator.evaluate(test_loader, train_labeled_loader))
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            if (epoch_counter + 1) % 20 == 0:
                # todo: save checkpoint
                logging.info("saving checkpoint - to be implemented")
                # save_checkpoint({
                #     'epoch': self.params['epochs'],
                #     'arch': self.args.arch,
                #     'state_dict': self.model.state_dict(),
                #     'optimizer': self.optimizer.state_dict(),
                # }, is_best=False, filename=os.path.join(self.neptune_run.log_dir, checkpoint_name))
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {constructive_loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints

