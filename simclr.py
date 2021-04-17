import logging
import os
import sys
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from evaluate import Evaluator

torch.manual_seed(0)
root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)
logging.getLogger('matplotlib.font_manager').disabled = True


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.neptune_run = kwargs['neptune_run']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.last_valid_index = kwargs['last_valid_index']

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        # log_dir = "runs/{}_lamb{}{:.2f}_exmp{}_class{}".format(current_time, self.params["lambda'], self.params['num_examples'], self.params['relevant_class'])
        # self.writer = SummaryWriter(log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.compactness_criterion = torch.nn.MSELoss().to(self.args.device)
        self.evaluator = Evaluator(self.writer, self.args, self.model, logging)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def compactness_loss_info(self, features, classes, indices):
        features = F.normalize(features, dim=1)
        rel_indices = (classes == self.args.rel_class) & (indices < self.last_valid_index)
        rel_features: torch.Tensor = features[torch.cat([rel_indices, rel_indices])].to(self.args.device)
        meaned_rel_features = rel_features - rel_features.mean(dim=0, keepdim=True)
        return meaned_rel_features

    def train(self, train_loader, test_loader, train_labeled_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        # save config file
        save_config_file(self.neptune_run.log_dir, self.args)
        n_iter = 0
        self.evaluator.evaluate(test_loader, n_iter, train_loader.dataset.dataset, train_labeled_loader)
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
        for epoch_counter in range(self.args.epochs):
            for images, classes, indices in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    constructive_loss = self.criterion(logits, labels)
                    zero_mean_labeled = self.compactness_loss_info(features, classes, indices)
                    compactness_loss = self.args.lamb * self.compactness_criterion(zero_mean_labeled,
                                                                                   torch.zeros_like(zero_mean_labeled))
                    # convert nan loss to 0, when there are no samples to make compact
                    compactness_loss[compactness_loss != compactness_loss] = 0

                self.optimizer.zero_grad()

                scaler.scale(constructive_loss + self.args.lamb * compactness_loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.neptune_run.add_scalar('losses/constructive_loss', constructive_loss, global_step=n_iter)
                    self.neptune_run.add_scalar('losses/compactness_loss', compactness_loss, global_step=n_iter)
                    self.neptune_run.add_scalar('losses/total_loss', constructive_loss + compactness_loss,
                                           global_step=n_iter)
                    self.neptune_run.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.neptune_run.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.neptune_run.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
            # run tests every 5 epochs

            logging.info(f"evaluating on epoch {epoch_counter + 1}")
            self.evaluator.evaluate(test_loader, n_iter, train_loader.dataset.dataset, train_labeled_loader)
            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            if epoch_counter + 1 % 20 == 0:
                logging.info("saving checkpoint")
                checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.neptune_run.log_dir, checkpoint_name))
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {constructive_loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints

        logging.info(f"Model checkpoint and metadata has been saved at {self.neptune_run.log_dir}.")

