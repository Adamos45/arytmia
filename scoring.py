import sklearn.metrics as metrics
import numpy as np


def scorer(estimator, X, Y):
    n_classes = 4
    pf_ms = performance_measures(n_classes)

    #In case of knn or mlp:
    predictions = estimator.predict(X)
    #In case of SVM:
    # decision_ovo = estimator.decision_function(X)
    # predictions, counter = ovo_voting(decision_ovo, n_classes)

    # Confussion matrix
    conf_mat = metrics.confusion_matrix(Y, predictions, labels=[0, 1, 2, 3])
    conf_mat = conf_mat.astype(float)
    pf_ms.confusion_matrix = conf_mat

    # Overall Acc
    pf_ms.Overall_Acc = metrics.accuracy_score(Y, predictions)

    # AAMI: Sens, Spec, Acc
    # N: 0, S: 1, V: 2, F: 3
    for i in range(0, n_classes):
        TP = conf_mat[i, i]
        FP = sum(conf_mat[:, i]) - conf_mat[i, i]
        TN = sum(sum(conf_mat)) - sum(conf_mat[i, :]) - sum(conf_mat[:, i]) + conf_mat[i, i]
        FN = sum(conf_mat[i, :]) - conf_mat[i, i]

        pf_ms.Recall[i] = TP / (TP + FN)
        pf_ms.Precision[i] = TP / (TP + FP)
        pf_ms.Specificity[i] = TN / (TN + FP)  # 1-FPR
        pf_ms.Acc[i] = (TP + TN) / (TP + TN + FP + FN)

        if TP == 0:
            pf_ms.F_measure[i] = 0.0
        else:
            pf_ms.F_measure[i] = 2 * (pf_ms.Precision[i] * pf_ms.Recall[i]) / (pf_ms.Precision[i] + pf_ms.Recall[i])

    # Compute Cohen's Kappa
    pf_ms.kappa, prob_obsv, prob_expect = compute_cohen_kappa(conf_mat)

    # Compute Index-j   recall_S + recall_V + precision_S + precision_V
    pf_ms.Ij = pf_ms.Recall[1] + pf_ms.Recall[2] + pf_ms.Precision[1] + pf_ms.Precision[2]

    # Compute Index-jk
    w1 = 0.5
    w2 = 0.125
    pf_ms.Ijk = w1 * pf_ms.kappa + w2 * pf_ms.Ij

    print("Ijk: " + str(format(pf_ms.Ijk, '.4f')) + "\n")
    print("Ij: " + str(format(pf_ms.Ij, '.4f')) + "\n")
    print("Cohen's Kappa: " + str(format(pf_ms.kappa, '.4f')) + "\n\n")
    # Conf matrix
    print("Confusion Matrix:" + "\n\n")
    print("\n".join(str(elem) for elem in pf_ms.confusion_matrix.astype(int)) + "\n\n")

    print("Overall ACC: " + str(format(pf_ms.Overall_Acc, '.4f')) + "\n\n")

    print("mean Acc: " + str(format(np.average(pf_ms.Acc[:]), '.4f')) + "\n")
    print("mean Recall: " + str(format(np.average(pf_ms.Recall[:]), '.4f')) + "\n")
    print("mean Precision: " + str(format(np.average(pf_ms.Precision[:]), '.4f')) + "\n")

    print("N:" + "\n\n")
    print("Sens: " + str(format(pf_ms.Recall[0], '.4f')) + "\n")
    print("Prec: " + str(format(pf_ms.Precision[0], '.4f')) + "\n")
    print("Acc: " + str(format(pf_ms.Acc[0], '.4f')) + "\n")

    print("SVEB:" + "\n\n")
    print("Sens: " + str(format(pf_ms.Recall[1], '.4f')) + "\n")
    print("Prec: " + str(format(pf_ms.Precision[1], '.4f')) + "\n")
    print("Acc: " + str(format(pf_ms.Acc[1], '.4f')) + "\n")

    print("VEB:" + "\n\n")
    print("Sens: " + str(format(pf_ms.Recall[2], '.4f')) + "\n")
    print("Prec: " + str(format(pf_ms.Precision[2], '.4f')) + "\n")
    print("Acc: " + str(format(pf_ms.Acc[2], '.4f')) + "\n")

    print("F:" + "\n\n")
    print("Sens: " + str(format(pf_ms.Recall[3], '.4f')) + "\n")
    print("Prec: " + str(format(pf_ms.Precision[3], '.4f')) + "\n")
    print("Acc: " + str(format(pf_ms.Acc[3], '.4f')) + "\n")

    return pf_ms.Ijk


class performance_measures:
    def __init__(self, n):
        self.n_classes = n
        self.confusion_matrix = np.empty([])
        self.Recall = np.empty(n)
        self.Precision = np.empty(n)
        self.Specificity = np.empty(n)
        self.Acc = np.empty(n)
        self.F_measure = np.empty(n)

        self.gmean_se = 0.0
        self.gmean_p = 0.0

        self.Overall_Acc = 0.0
        self.kappa = 0.0
        self.Ij = 0.0
        self.Ijk = 0.0


# def ovo_class_combinations(n_classes):
#     class_pos = []
#     class_neg = []
#     for c1 in range(n_classes - 1):
#         for c2 in range(c1 + 1, n_classes):
#             class_pos.append(c1)
#             class_neg.append(c2)
#
#     return class_pos, class_neg


# def ovo_voting(decision_ovo, n_classes):
#     predictions = np.zeros(len(decision_ovo))
#     class_pos, class_neg = ovo_class_combinations(n_classes)
#
#     counter = np.zeros([len(decision_ovo), n_classes])
#
#     for p in range(len(decision_ovo)):
#         for i in range(len(decision_ovo[p])):
#             if decision_ovo[p, i] > 0:
#                 counter[p, class_pos[i]] += 1
#             else:
#                 counter[p, class_neg[i]] += 1
#
#         predictions[p] = np.argmax(counter[p])
#
#     return predictions, counter


def compute_cohen_kappa(confusion_matrix):
    prob_expectedA = np.empty(len(confusion_matrix))
    prob_expectedB = np.empty(len(confusion_matrix))
    prob_observed = 0

    for n in range(0, len(confusion_matrix)):
        prob_expectedA[n] = sum(confusion_matrix[n, :]) / sum(sum(confusion_matrix))
        prob_expectedB[n] = sum(confusion_matrix[:, n]) / sum(sum(confusion_matrix))

        prob_observed = prob_observed + confusion_matrix[n][n]

    prob_expected = np.dot(prob_expectedA, prob_expectedB)
    prob_observed = prob_observed / sum(sum(confusion_matrix))

    kappa = (prob_observed - prob_expected) / (1 - prob_expected)

    return kappa, prob_observed, prob_expected
