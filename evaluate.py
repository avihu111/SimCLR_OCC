from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.metrics import silhouette_score, roc_curve, roc_auc_score, plot_roc_curve
from sklearn.neighbors import KDTree, NearestNeighbors, KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE

class Evaluator():
    def __init__(self, writer, args, model, logging):
        self.writer = writer
        self.args = args
        self.model = model
        self.logging = logging

    def evaluate(self, test_loader, n_iter, train_dataset, train_labeled_loader):
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
        positive_features = self.get_labeled_features(train_labeled_loader)
        reduced_features = self.calculate_tsne(all_features, all_labels)
        self.plot_tsne(all_labels, n_iter, reduced_features, test_loader.dataset.classes, 'tSNE')
        # self.one_class_svm(all_features, all_labels, n_iter, positive_features, reduced_features)
        self.k_nearest(all_features, all_labels, n_iter, positive_features, reduced_features)
        self.writer.flush()

    def get_labeled_features(self, train_labeled_loader):
        all_features = []
        for batch_id, (images, labels) in enumerate(train_labeled_loader):
            features = self.model(images.to(self.args.device))
            all_features.append(features.cpu().detach().numpy())

        # maybe unite all other colors to -1?
        all_features = np.concatenate(all_features, axis=0)
        return all_features

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
        IoU = TP / (binary_labels | binary_preds).sum()
        self.writer.add_scalar('SVC/precision', precision, n_iter)
        self.writer.add_scalar('SVC/recall', recall, n_iter)
        self.writer.add_scalar('SVC/accuracy', accur, n_iter)
        self.writer.add_scalar('SVC/IoU', IoU, n_iter)
        self.logging.info(f"SVM accuracy: {accur}")

    def one_class_svm(self, all_features, all_labels, n_iter, positive_features, reduced_features):
        # ratio of outliers to expect in the features we try to contour
        for nu in [0.05, 0.1, 0.2]:
            for gamma in [1e-7, 1e-6, 1e-5, 1e-4]:
                svm = OneClassSVM(nu=nu, gamma=gamma)
                svm.fit(positive_features)
                # inliers are 1, outliers are -1
                is_inlier_pred = svm.predict(all_features) == 1
                is_inlier_true = all_labels == self.args.rel_class
                accur = (is_inlier_true == is_inlier_pred).mean()
                TP = (is_inlier_true & is_inlier_pred).sum()
                recall = np.nan_to_num(TP / is_inlier_true.sum())
                precision = np.nan_to_num(TP / is_inlier_pred.sum())
                IoU = np.nan_to_num(TP / (is_inlier_true | is_inlier_pred).sum())
                f_score = 2 * precision * recall / (precision + recall)
                self.writer.add_scalar(f'OneClassSVM_recall/nu={nu}_gamma={gamma}', recall, n_iter)
                self.writer.add_scalar(f'OneClassSVM_precision/nu={nu}_gamma={gamma}', precision, n_iter)
                self.writer.add_scalar(f'OneClassSVM_accuracy/nu={nu}_gamma={gamma}', accur, n_iter)
                self.writer.add_scalar(f'OneClassSVM_IoU/nu={nu}_gamma={gamma}', IoU, n_iter)
                self.writer.add_scalar(f'OneClassSVM_Fscore/nu={nu}_gamma={gamma}', f_score, n_iter)
                self.plot_tsne(is_inlier_pred.astype(int), n_iter, reduced_features,
                               classes=['outlier', 'inlier'],
                               plot_name=f'OneClassSVM/nu={nu}_gamma={gamma}')

    def k_nearest(self, all_features, all_labels, n_iter, positive_features, reduced_features):
        # ratio of outliers to expect in the features we try to contour
        roc_fig, roc_axis = plt.subplots()
        for k in [1, 2, 10, 25, 50]:
            nearest = NearestNeighbors(n_neighbors=k)
            nearest.fit(positive_features)
            distances, indices = nearest.kneighbors(all_features)
            mean_distance = distances.mean(axis=1)
            is_outlier_true = (all_labels != self.args.rel_class).astype(int)
            fpr, tpr, thresholds = roc_curve(is_outlier_true, mean_distance)
            auc = roc_auc_score(is_outlier_true, mean_distance)
            self.writer.add_scalar(f'k_nearest_auc/k={k}', auc, n_iter)
            roc_axis.plot(fpr, tpr, label=f'k={k}_auc={auc:.3f}')
            # optimal is closest point to [0,1]
            diff_from_best = (1 - tpr) ** 2 + fpr ** 2
            optimal_cutoff = thresholds[np.argmin(diff_from_best)]
            self.plot_tsne((mean_distance > optimal_cutoff).astype(int), n_iter, reduced_features,
                           classes=['inlier', 'outlier'], plot_name=f'k_nearest_tSNE/k={k}')

        roc_axis.set_xlabel('FPR')
        roc_axis.set_ylabel('TPR')
        roc_axis.legend()
        roc_fig.canvas.draw()
        roc_im = np.fromstring(roc_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        roc_im = roc_im.reshape(roc_fig.canvas.get_width_height()[::-1] + (3,))
        self.writer.add_image(f'k_nearest_roc/k={k}', roc_im, global_step=n_iter, dataformats='HWC')
        plt.close('all')

    def plot_tsne(self, all_labels, n_iter, reduced_features, classes, plot_name):
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
        self.writer.add_image(plot_name, tsne_plot_array, global_step=n_iter, dataformats='HWC')
        plt.close('all')

    def calculate_tsne(self, all_features, all_labels):
        tsne = TSNE(n_jobs=self.args.workers)
        reduced_features = tsne.fit_transform(all_features)
        rel_centroid = reduced_features[all_labels == self.args.rel_class].mean(axis=0)
        rel_centroid = rel_centroid / np.linalg.norm(rel_centroid)
        # rotate so the centroid will be at the positive x axis
        R = np.stack([rel_centroid, rel_centroid[::-1]])
        R[0, 1] = -R[0, 1]
        reduced_features = (reduced_features @ R)
        return reduced_features

    def separation_metrics(self, all_features, all_labels, n_iter):
        def get_mean_cov(features):
            mean = features.mean(axis=0)
            zero_mean_features = features - mean[None, :]
            cov = zero_mean_features.T @ zero_mean_features
            return mean, cov

        binary_labels = (all_labels == self.args.rel_class)
        myu_1, sigma_1 = get_mean_cov(all_features[binary_labels])
        myu_2, sigma_2 = get_mean_cov(all_features[~binary_labels])
        w = np.linalg.inv(sigma_1 + sigma_2) @ (myu_1 - myu_2)
        w = w / np.linalg.norm(w)
        fisher_score = ((w @ (myu_1 - myu_2)) ** 2) / (w.T @ ((sigma_1 + sigma_2) @ w))
        # between -1 to 1. best result is 1.
        silhouette = silhouette_score(all_features, binary_labels.astype(int))
        silhouette_all = silhouette_score(all_features, all_labels)
        self.writer.add_scalar('separation/silhouette', silhouette, n_iter)
        self.writer.add_scalar('separation/silhouette_all', silhouette_all, n_iter)
        self.writer.add_scalar('separation/FLD', fisher_score, n_iter)
        self.logging.info(f"silhouette: {silhouette}")
