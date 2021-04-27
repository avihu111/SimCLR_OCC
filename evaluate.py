from sklearn.svm import OneClassSVM, LinearSVC
from sklearn.metrics import silhouette_score, roc_curve, roc_auc_score, plot_roc_curve
from sklearn.neighbors import KDTree, NearestNeighbors, KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
from neptune.new.types import File
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
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
        all_features, all_labels = self.get_all_features(test_loader, train_labeled_loader)
        selected_indices = self.select_test_indices(all_labels)
        all_features = all_features[selected_indices]
        all_labels = all_labels[selected_indices]
        # self.svm(all_features, all_labels)
        # self.separation_metrics(all_features, all_labels)
        reduced_features = self.calculate_tsne(all_features, all_labels)
        self.plot_tsne(all_labels, reduced_features, CLASSES, 'tSNE')
        self.k_nearest(all_features, all_labels, reduced_features)
        return np.mean(all_features[all_labels == ENUM['train_inlier']], axis=0)

    @staticmethod
    def select_test_indices(all_labels):
        train_indices = np.nonzero(all_labels == ENUM['train_inlier'])[0]
        inlier_indices = np.nonzero(all_labels == ENUM['inlier'])[0]
        outlier_indices = np.nonzero(all_labels == ENUM['outlier'])[0]
        assert len(inlier_indices) <= len(outlier_indices)
        outlier_indices = np.random.choice(outlier_indices, len(inlier_indices))
        selected_indices = np.concatenate([train_indices, inlier_indices, outlier_indices])
        return selected_indices

    def get_all_features(self, test_loader, train_labeled_loader):
        test_features = []
        test_labels = []
        for images, labels, indices in test_loader:
            features = self.model(images.to(self.args.device))
            features = self.model.last_activations['output'] if self.params['large_features'] else features.detach()
            test_features.append(features.cpu().numpy())
            test_labels.append(labels.detach().numpy())
        positive_train_features = self.get_labeled_features(train_labeled_loader)
        all_features = np.concatenate([positive_train_features] + test_features, axis=0)
        positive_train_labels = np.full(shape=self.args.num_labeled_examples, dtype=int,
                                        fill_value=ENUM['train_inlier'])
        assert ENUM['outlier'] == 0 and ENUM['inlier'] == 1, 'next line is using this assumption'
        reduced_test_labels = np.isin(np.concatenate(test_labels, axis=0), self.args.relevant_classes).astype(int)
        all_labels = np.concatenate([positive_train_labels, reduced_test_labels])
        return all_features, all_labels

    def get_labeled_features(self, train_labeled_loader):
        all_features = []
        for images, labels, indices in train_labeled_loader:
            features = self.model(images.to(self.args.device))
            features = self.model.last_activations['output'] if self.params['large_features'] else features.detach()
            all_features.append(features.cpu().numpy())

        np_images = images.cpu().detach().numpy().transpose(0,2,3,1)
        assert np.all(np_images <= 1) and np.all(np_images >= 0)
        chosen_im = np_images[np.random.choice(len(np_images))]
        self.neptune_run['plots/train_images'].log(File.as_image(chosen_im))
        # maybe unite all other colors to -1?
        all_features = np.concatenate(all_features, axis=0)
        assert len(all_features) == self.params['num_labeled_examples']
        return all_features

    def svm(self, all_features, all_labels):
        # todo: remove train data from metric?
        is_inlier_label = np.isin(all_labels, [ENUM['inlier'], ENUM['train_inlier']]).astype(int)
        svc = LinearSVC()
        svc.fit(all_features, is_inlier_label)
        is_relevant_pred = svc.predict(all_features)
        TP = (is_inlier_label & is_relevant_pred).sum()
        precision = np.nan_to_num(TP / is_relevant_pred.sum())
        recall = np.nan_to_num(TP / is_inlier_label.sum())
        accur = (is_inlier_label == is_relevant_pred).mean()
        IoU = TP / (is_inlier_label | is_relevant_pred).sum()
        f_score = 2 * precision * recall / (precision + recall)
        self.neptune_run['metrics/SVC/precision'].log(precision)
        self.neptune_run['metrics/SVC/recall'].log(recall)
        self.neptune_run['metrics/SVC/accuracy'].log(accur)
        self.neptune_run['metrics/SVC/IoU'].log(IoU)
        self.neptune_run['metrics/SVC/F-score'].log(f_score)

    def k_nearest(self, all_features, all_labels, reduced_features):
        fig = go.Figure()
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        # ratio of outliers to expect in the features we try to contour
        is_train_feature = all_labels == ENUM['train_inlier']
        for features, mode in [(all_features, 'regular')]:#, (reduced_features, 'reduced')]:
            for k in [1, 2, 10, 25, 50]:
                if is_train_feature.sum() < k:
                    continue
                nearest = NearestNeighbors(n_neighbors=k)
                nearest.fit(features[is_train_feature])
                distances, indices = nearest.kneighbors(features[~is_train_feature])
                mean_distance = distances.mean(axis=1)
                is_outlier_true = (all_labels[~is_train_feature] == ENUM['outlier']).astype(int)
                fpr, tpr, thresholds = roc_curve(is_outlier_true, mean_distance)
                auc = roc_auc_score(is_outlier_true, mean_distance)
                self.neptune_run[f'metrics/k_nearest_auc/{mode}_k={k}'].log(auc)
                # roc_axis.plot(fpr, tpr, label=f'k={k}_auc={auc:.2f}')
                # optimal is closest point to [0,1]
                diff_from_best = (1 - tpr) ** 2 + fpr ** 2
                cutoff_idx = np.argmin(diff_from_best)
                optimal_cutoff = thresholds[cutoff_idx]
                name = f"k={k} (AUC={auc:.2f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))
                # fig.add_trace(go.Scatter(x=fpr[cutoff_idx], y=tpr[cutoff_idx]))
                # roc_axis.plot(fpr[cutoff_idx], tpr[cutoff_idx], 'xk')
                is_outlier_pred = mean_distance > optimal_cutoff
                classifier_labels = np.full_like(all_labels, fill_value=CLASSIFIER_RESULTS_ENUM['trainP'])
                classifier_labels[~is_train_feature] = 2 * is_outlier_pred + is_outlier_true
                self.plot_tsne(classifier_labels, reduced_features,
                               classes=CLASSIFIER_RESULTS, plot_name=f'{mode}_{k}_nearest')
        fig.update_layout(
            xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1), xaxis=dict(constrain='domain'),
            width=700, height=500)
        self.neptune_run[f'plots/ruc_curve'] = File.as_html(fig)

    def plot_tsne(self, all_labels, reduced_features, classes, plot_name):
        source = pd.DataFrame({'x': reduced_features[:, 0], 'y': reduced_features[:, 1], 'label': np.array(classes)[all_labels]})
        brush = alt.selection(type='interval')
        points = alt.Chart(source).mark_point().encode(x='x:Q', y='y:Q', color=alt.condition(brush, 'label:N', alt.value('lightgray'))).add_selection(brush)
        bars = alt.Chart(source).mark_bar().encode(y='label:N', color='label:N', x='count(label):Q').transform_filter(brush)
        chart = points & bars
        self.neptune_run[f'plots/{plot_name}'] = File.as_html(chart)

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
        self.neptune_run['metrics/separation/silhouette'].log(silhouette)
        self.neptune_run['metrics/separation/FLD'].log(fisher_score)
