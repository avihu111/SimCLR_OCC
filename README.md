# SimCLR OCC : Semi-Supervised One-Class Classification using Self-Supervised Learning

### Highly relays on the code of SimCLR (cloned and modified): [Exploring SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://github.com/sthalles/SimCLR)

### Project description is here: [Semi Supervised One Class Classification](https://docs.google.com/document/d/1LiRSr5Vj4lcW-7OIBbXlYpNTfiL1aYca-yd8GWFrErY/edit?usp=sharing)

![Image of SimCLR Arch](https://sthalles.github.io/assets/contrastive-self-supervised/cover.png)


One-Class Classification (OCC) is the problem of learning a classifier for a class, while exposed only to samples from that particular class. For example, at training time, the classifier is given a set of images of dogs, and at test time is asked to correctly classify new dog/non-dog images.
This problem is more difficult than binary classification, since the classifier cannot try to find distinguishing features between two classes. To illustrate the problem, consider the Support Vector Machine (SVM) algorithm, and recall that SVM chooses the optimal separating hyperplane by maximizing the margin between the two classes. One-Class methods cannot utilize such information, since outlier examples are absent.

OCC is closely related to Anomaly Detection and Outlier Detection (during training, it receives normal/inlier samples, and is asked to detect abnormal/outlier samples). Therefore, in many applications, such a One-Class Classifier has high importance. Intrusion detection and fraud detection are classical applications. In other applications, such as autonomous driving and medical diagnosis, it is important to know that a given input is out of distribution, and therefore  continue processing with a more conservative fallback protocol.

In the Semi-Supervised OCC setting, the classifier is given a positively labeled set P, and a mixed unlabeled set U, and attempts to utilize U to improve it’s OCC performance. Recent Anomaly Detection methods assume that the pollution ratio (outliers ratio) of the unlabeled set is low. 
In this work we show that current Semi-Supervised OCC methods perform poorly when the unlabeled dataset is highly polluted (contains many out-of-class examples), and suggest a novel method that outperforms SOTA methods in this setup, called Sim-OCC.
