from torch.autograd import Function
import torch.nn as nn
import torch


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.mm(input2_l2.t())).pow(2))

        return diff_loss

def cluster_contrast(fushed,centroid,labels):
    S = torch.matmul(fushed, centroid.t())
    bs = fushed.shape[0]

    target = torch.zeros(bs,centroid.shape[0]).to(S.device)

    target[range(target.shape[0]), labels] = 1

    S = S - target * (0.001)

    S = S.view(S.shape[0], S.shape[1], -1)
    nominator = S * target[:, :, None]
    nominator = nominator.sum(dim=1)
    nominator = torch.logsumexp(nominator, dim=1)
    denominator = S.view(S.shape[0], -1)
    denominator = torch.logsumexp(denominator, dim=1)
    I2C_loss = torch.mean(denominator - nominator)

    return I2C_loss


from sklearn import metrics
from sklearn.metrics.cluster._supervised import contingency_matrix
from munkres import Munkres
import numpy as np
def best_map(L1, L2):
    # L1 should be the ground-truth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = (L1 == Label1[i]).astype(float)
        for j in range(nClass2):
            ind_cla2 = (L2 == Label2[j]).astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def get_ar(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)


def get_nmi(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')


def get_fpr(y_true, y_pred):
    n_samples = np.shape(y_true)[0]
    c = contingency_matrix(y_true, y_pred, sparse=True)
    tk = np.dot(c.data, np.transpose(c.data)) - n_samples  # TP
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples  # TP+FP
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples  # TP+FN
    precision = 1. * tk / pk if tk != 0. else 0.
    recall = 1. * tk / qk if tk != 0. else 0.
    f = 2 * precision * recall / (precision + recall) if (precision +
                                                          recall) != 0. else 0.
    return f, precision, recall


def get_purity(y_true, y_pred):
    c = metrics.confusion_matrix(y_true, y_pred)
    return 1. * c.max(axis=0).sum() / np.shape(y_true)[0]


def calculate_metrics(y, y_pred):
    y_new = best_map(y, y_pred)
    acc = metrics.accuracy_score(y, y_new)
    ar = get_ar(y, y_pred)
    nmi = get_nmi(y, y_pred)
    f, p, r = get_fpr(y, y_pred)
    purity = get_purity(y, y_pred)

    # f_score = metrics.f1_score(y, y_pred, average='macro')
    # f_score = np.round(f_score, 4)
    #wandb.log({'Accuracy': acc, 'MNI': nmi, 'F': f, 'purity': purity})
    #return acc, ar, nmi, f, p, r, purity

    return acc, nmi, f, purity
