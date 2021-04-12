import logging
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.metrics import silhouette_score

torch.manual_seed(0)
root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.last_valid_index = kwargs['last_valid_index']

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = f'runs/{current_time}_lamb{self.args.lamb:.2f}_exmp{self.args.num_examples}'
        self.writer = SummaryWriter(log_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.compactness_criterion = torch.nn.MSELoss().to(self.args.device)

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

    def train(self, train_loader, test_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        # save config file
        save_config_file(self.writer.log_dir, self.args)
        n_iter = 0
        self.run_on_test(test_loader, n_iter, train_loader.dataset.dataset)
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")
        for epoch_counter in range(self.args.epochs):
            for images, classes, indices in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    constructive_loss = self.criterion(logits, labels)
                    meaned_rel_features = self.compactness_loss_info(features, classes, indices)
                    compactness_loss = self.args.lamb * self.compactness_criterion(meaned_rel_features,
                                                                                   torch.zeros_like(
                                                                                       meaned_rel_features))
                    # convert nan loss to 0, when there are no samples to make compact
                    compactness_loss[compactness_loss != compactness_loss] = 0

                self.optimizer.zero_grad()

                scaler.scale(constructive_loss + self.args.lamb * compactness_loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))

                    self.writer.add_scalar('losses/constructive_loss', constructive_loss, global_step=n_iter)
                    self.writer.add_scalar('losses/compactness_loss', compactness_loss, global_step=n_iter)
                    self.writer.add_scalar('losses/total_loss', constructive_loss + compactness_loss,
                                           global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
            # run tests every 5 epochs

            logging.info(f"evaluating on epoch {epoch_counter + 1}")
            self.run_on_test(test_loader, n_iter, train_loader.dataset.dataset)
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
                }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {constructive_loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints

        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")

    def run_on_test(self, test_loader, n_iter, train_dataset):
        all_features = []
        all_labels = []

        for batch_id, (images, labels) in enumerate(test_loader):
            features = self.model(images.to(self.args.device))
            all_features.append(features.cpu().detach().numpy())
            all_labels.append(labels.detach().numpy())

        all_labels = np.concatenate(all_labels, axis=0)
        # maybe unite all other colors to -1?
        all_features = np.concatenate(all_features, axis=0)
        # solve linear svm to check separability of the class
        self.svm(all_features, all_labels, n_iter)
        self.separation_metrics(all_features, all_labels, n_iter)
        self.plot_tsne(all_labels, n_iter, all_features, test_loader.dataset.classes)
        # self.one_class_svm(all_features, all_labels, n_iter, train_dataset)

    def svm(self, all_features, all_labels, n_iter):
        """
        learning an SVM and checking it to see how separable the sets are
        """
        binary_labels = (all_labels == self.args.rel_class).astype(int)
        svc = LinearSVC()
        svc.fit(all_features, binary_labels)
        binary_preds = svc.predict(all_features)
        TP = (binary_labels & binary_preds).sum()
        precision = TP / binary_preds.sum()
        recall = TP / binary_labels.sum()
        accur = (binary_labels == binary_preds).mean()
        self.writer.add_scalar('SVC/precision', precision, n_iter)
        self.writer.add_scalar('SVC/recall', recall, n_iter)
        self.writer.add_scalar('SVC/accuracy', accur, n_iter)
        logging.info(f"SVM accuracy: {accur}")

    def one_class_svm(self, all_features, all_labels, n_iter, train_dataset):
        positive_indices = (train_dataset.labels == 0).nonzero()[0][:self.args.num_examples]
        positive_samples = torch.FloatTensor(train_dataset.data[positive_indices] / 255.)
        positive_features = self.model(positive_samples.to(self.args.device))
        positive_features = positive_features.cpu().detach().numpy()
        svm = OneClassSVM(nu=0.9)
        svm.fit(positive_features)

        is_novelty_pred = svm.predict(all_features) == -1
        is_novelty_true = all_labels == self.args.rel_class
        accur = (is_novelty_true == is_novelty_pred).mean()
        TP = (is_novelty_true & is_novelty_pred).sum()
        recall = np.nan_to_num(TP / is_novelty_true.sum())
        precision = np.nan_to_num(TP / is_novelty_pred.sum())
        self.writer.add_scalar('OneClassSVM/recall', recall, n_iter)
        self.writer.add_scalar('OneClassSVM/precision', precision, n_iter)
        self.writer.add_scalar('OneClassSVM/accuracy', accur, n_iter)

    def plot_tsne(self, all_labels, n_iter, all_features, classes):
        tsne = TSNE()
        reduced_features = tsne.fit_transform(all_features)
        rel_centroid = reduced_features[all_labels == self.args.rel_class].mean(axis=0)
        rel_centroid = rel_centroid / np.linalg.norm(rel_centroid)
        R = np.stack([rel_centroid, rel_centroid[::-1]])
        R[0, 1] = -R[0, 1]

        reduced_features = (R @ reduced_features.T).T
        fig, axis = plt.subplots()
        cmap = plt.cm.get_cmap('tab10')(np.arange(10))
        cmap[:, -1] = 0.5
        # seeing our class last
        for lab in np.unique(all_labels)[::-1]:
            mask = all_labels == lab
            x, y = reduced_features[mask].T
            axis.scatter(x, y, c=cmap[[lab]], label=classes[lab])

        axis.legend(loc='upper left')
        fig.canvas.draw()
        tsne_plot_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        tsne_plot_array = tsne_plot_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.writer.add_image('tSNE', tsne_plot_array, global_step=n_iter, dataformats='HWC')
        plt.close('all')

    def separation_metrics(self, all_features, all_labels, n_iter):
        binary_labels = (all_labels == self.args.rel_class).astype(int)
        rel_features = all_features[binary_labels]
        rel_centroid = rel_features.mean(axis=0)
        rel_variance = ((rel_features - rel_centroid[None, :]) ** 2).mean()
        orig_centroid = all_features.mean(axis=0)
        orig_variance = ((all_features - orig_centroid[None, :]) ** 2).mean()
        variance_ratio = rel_variance / orig_variance
        self.writer.add_scalar('separation/variance_ratio', variance_ratio, n_iter)
        centroids_distance = ((rel_centroid - orig_centroid) ** 2).sum()
        self.writer.add_scalar('separation/mean_centroid_distance', centroids_distance, n_iter)
        # less sensitive to scales
        var_normalized_distance = centroids_distance / (rel_variance + orig_variance)
        self.writer.add_scalar('separation/var_normalized_distance', var_normalized_distance, n_iter)
        # between -1 to 1. best result is 1.
        silhouette = silhouette_score(all_features, binary_labels)
        self.writer.add_scalar('separation/silhouette', silhouette, n_iter)
        logging.info(f"normalized distance: {var_normalized_distance}. silhouette: {silhouette}")
