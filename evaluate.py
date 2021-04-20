from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.metrics import silhouette_score, roc_curve, roc_auc_score, plot_roc_curve
from sklearn.neighbors import KDTree, NearestNeighbors, KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
from neptune.new.types import File

IMAGES_TO_LOG = 2

CLASSES = ['outlier', 'inlier', 'train_inlier']
ENUM = {k:i for i,k in enumerate(CLASSES)}
CLASSIFIER_RESULTS = ['TP', 'FP', 'FN', 'TN', 'trainP']
CLASSIFIER_RESULTS_ENUM = {k:i for i,k in enumerate(CLASSIFIER_RESULTS)}

class Evaluator():
    def __init__(self, neptune_run, args, model, logging, params):
        self.neptune_run = neptune_run
        self.args = args
        self.model = model
        self.logging = logging
        self.params = params
        self.neptune_run['enum'] = ENUM
        self.neptune_run['classifier_enum'] = CLASSIFIER_RESULTS_ENUM

    def evaluate(self, test_loader, train_labeled_loader):
        test_features = []
        test_labels = []

        for batch_id, (images, labels) in enumerate(test_loader):
            features = self.model(images.to(self.args.device))
            test_features.append(features.cpu().detach().numpy())
            test_labels.append(labels.detach().numpy())

        positive_train_features = self.get_labeled_features(train_labeled_loader)
        all_features = np.concatenate([positive_train_features] + test_features, axis=0)

        positive_train_labels = np.full(shape=self.args.num_labeled_examples, dtype=int,
                                        fill_value=ENUM['train_inlier'])
        reduced_test_labels = (np.concatenate(test_labels, axis=0) == self.args.relevant_class).astype(int)
        all_labels = np.concatenate([positive_train_labels, reduced_test_labels])
        self.svm(all_features, all_labels)
        self.separation_metrics(all_features, all_labels)
        reduced_features = self.calculate_tsne(all_features, all_labels)
        self.plot_tsne(all_labels, reduced_features, CLASSES, 'tSNE')
        self.k_nearest(all_features, all_labels, reduced_features)

    def get_labeled_features(self, train_labeled_loader):
        all_features = []
        for batch_id, (images, labels) in enumerate(train_labeled_loader):
            features = self.model(images.to(self.args.device))
            all_features.append(features.cpu().detach().numpy())

        np_images = images.cpu().detach().numpy().transpose(0,2,3,1)
        assert np.all(np_images <= 1) and np.all(np_images >= 0)
        chosen_im = np_images[np.random.choice(len(np_images))]
        self.neptune_run['plots/train_images'].log(File.as_image(chosen_im))
        # maybe unite all other colors to -1?
        all_features = np.concatenate(all_features, axis=0)
        return all_features

    def svm(self, all_features, all_labels):
        """
        learning an SVM and checking it to see how separable the sets are
        """
        is_relevant_label = np.isin(all_labels, [ENUM['inlier'], ENUM['train_inlier']]).astype(int)
        svc = LinearSVC()
        svc.fit(all_features, is_relevant_label)
        is_relevant_pred = svc.predict(all_features)
        TP = (is_relevant_label & is_relevant_pred).sum()
        precision = TP / is_relevant_pred.sum()
        recall = TP / is_relevant_label.sum()
        accur = (is_relevant_label == is_relevant_pred).mean()
        IoU = TP / (is_relevant_label | is_relevant_pred).sum()
        f_score = 2 * precision * recall / (precision + recall)
        self.neptune_run['metrics/SVC/precision'].log(precision)
        self.neptune_run['metrics/SVC/recall'].log(recall)
        self.neptune_run['metrics/SVC/accuracy'].log(accur)
        self.neptune_run['metrics/SVC/IoU'].log(IoU)
        self.neptune_run['metrics/SVC/F-score'].log(f_score)

    def k_nearest(self, all_features, all_labels, reduced_features):
        # ratio of outliers to expect in the features we try to contour
        roc_fig, roc_axis = plt.subplots()
        is_train_feature = all_labels == ENUM['train_inlier']
        for features, mode in [(all_features, 'regular'), (reduced_features, 'reduced')]:
            for k in [1, 2, 10, 25, 50]:
                nearest = NearestNeighbors(n_neighbors=k)
                nearest.fit(features[is_train_feature])
                distances, indices = nearest.kneighbors(features[~is_train_feature])
                mean_distance = distances.mean(axis=1)
                is_outlier_true = (all_labels[~is_train_feature] == ENUM['outlier']).astype(int)
                fpr, tpr, thresholds = roc_curve(is_outlier_true, mean_distance)
                auc = roc_auc_score(is_outlier_true, mean_distance)
                self.neptune_run[f'metrics/k_nearest_auc/{mode}_k={k}'].log(auc)
                roc_axis.plot(fpr, tpr, label=f'k={k}_auc={auc:.2f}')
                # optimal is closest point to [0,1]
                diff_from_best = (1 - tpr) ** 2 + fpr ** 2
                cutoff_idx = np.argmin(diff_from_best)
                optimal_cutoff = thresholds[cutoff_idx]
                roc_axis.plot(fpr[cutoff_idx], tpr[cutoff_idx], 'xk')
                is_outlier_pred = mean_distance > optimal_cutoff
                classifier_labels = np.full_like(all_labels, fill_value=CLASSIFIER_RESULTS_ENUM['trainP'])
                classifier_labels[~is_train_feature] = 2 * is_outlier_pred + is_outlier_true
                self.plot_tsne(classifier_labels, reduced_features,
                               classes=CLASSIFIER_RESULTS, plot_name=f'{mode}_{k}_nearest')

        roc_axis.set_xlabel('FPR')
        roc_axis.set_ylabel('TPR')
        roc_axis.legend('lower left')
        self.neptune_run[f'plots/k_nearest_roc'].log(roc_fig)
        plt.close('all')

    def plot_tsne(self, all_labels, reduced_features, classes, plot_name):
        fig, axis = plt.subplots()
        cmap = plt.cm.get_cmap('tab10')(np.arange(11))
        cmap[:, -1] = 0.5
        for lab in np.unique(all_labels):
            mask = all_labels == lab
            x, y = reduced_features[mask].T
            axis.scatter(x, y, c=cmap[[lab]], label=classes[lab], s=10)

        axis.legend(loc='upper left')
        self.neptune_run['plots/' + plot_name].log(fig)
        plt.close('all')

    def calculate_tsne(self, all_features, all_labels):
        tsne = TSNE(n_jobs=self.args.workers)
        reduced_features = tsne.fit_transform(all_features)
        rel_centroid = reduced_features[np.isin(all_labels, [ENUM['inlier'], ENUM['train_inlier']])].mean(axis=0)
        rel_centroid = rel_centroid / np.linalg.norm(rel_centroid)
        # rotate so the centroid will be at the positive x axis
        R = np.stack([rel_centroid, rel_centroid[::-1]])
        R[0, 1] = -R[0, 1]
        assert np.allclose(R @ R.T, np.eye(2))
        reduced_features = (reduced_features @ R)
        return reduced_features

    def separation_metrics(self, all_features, all_labels):
        def get_mean_cov(features):
            mean = features.mean(axis=0)
            zero_mean_features = features - mean[None, :]
            cov = zero_mean_features.T @ zero_mean_features
            return mean, cov

        binary_labels = np.isin(all_labels, [ENUM['inlier'], ENUM['train_inlier']])
        myu_1, sigma_1 = get_mean_cov(all_features[binary_labels])
        myu_2, sigma_2 = get_mean_cov(all_features[~binary_labels])
        w = np.linalg.inv(sigma_1 + sigma_2) @ (myu_1 - myu_2)
        w = w / np.linalg.norm(w)
        fisher_score = ((w @ (myu_1 - myu_2)) ** 2) / (w.T @ ((sigma_1 + sigma_2) @ w))
        # between -1 to 1. best result is 1.
        silhouette = silhouette_score(all_features, binary_labels.astype(int))
        silhouette_all = silhouette_score(all_features, all_labels)
        self.neptune_run['metrics/separation/silhouette'].log(silhouette)
        self.neptune_run['metrics/separation/silhouette_all'].log(silhouette_all)
        self.neptune_run['metrics/separation/FLD'].log(fisher_score)
