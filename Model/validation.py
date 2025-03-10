import numpy as np

def dice_coef(x, y, dataset='OASIS'):
    labels = []

    if dataset == 'OASIS':
        labels = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53,
                    54, 60]
    elif dataset == 'IXI':
        labels = [2, 41, 3, 42, 4, 43, 7, 46, 8, 47, 11, 50, 12, 51, 13, 52, 14, 15, 16, 17, 53, 18, 54, 24, 28, 60]
    elif dataset == 'Mind101':
        labels = [1, 2, 3, 4, 5]

    dices = np.zeros(shape=len(labels))
    for id, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(x == label, y == label))
        bottom = np.sum(x == label) + np.sum(y == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)
        dice = top/bottom
        dices[id] = dice
    return np.mean(dices)


def NJD(displacement):
    datasize = np.prod(displacement.shape)
    D_y = (displacement[1:, :-1, :-1, :] - displacement[:-1, :-1, :-1, :])
    D_x = (displacement[:-1, 1:, :-1, :] - displacement[:-1, :-1, :-1, :])
    D_z = (displacement[:-1, :-1, 1:, :] - displacement[:-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])

    Ja_value = D1 - D2 + D3

    return np.sum(Ja_value < 0)/datasize


from medpy import metric

def hd95_val(y_pred, y_true, dataset=None):
    labels = []
    if dataset == 'OASIS':
        labels = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53,
                    54, 60]
    elif dataset == 'IXI':
        labels = [2, 41, 3, 42, 4, 43, 7, 46, 8, 47, 11, 50, 12, 51, 13, 52, 14, 15, 16, 17, 53, 18, 54, 24, 28, 60]
    elif dataset == 'Mind101':
        labels = [1, 2, 3, 4, 5]

    else:
        return

    total = []
    for i in labels:
        total.append(metric.hd95(y_pred == i, y_true == i))
    result = np.mean(total)
    # total.append(np.mean(total))
    return result

def assd_val(y_pred, y_true, dataset=None):
    labels = []
    if dataset == 'OASIS':
        labels = [2, 3, 4, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 28, 41, 42, 43, 46, 47, 49, 50, 51, 52, 53,
                    54, 60]
    elif dataset == 'IXI':
        labels = [2, 41, 3, 42, 4, 43, 7, 46, 8, 47, 11, 50, 12, 51, 13, 52, 14, 15, 16, 17, 53, 18, 54, 24, 28, 60]
    elif dataset == 'Mind101':
        labels = [1, 2, 3, 4, 5]
    else:
        return

    total = []
    for i in labels:
        total.append(metric.assd(y_pred == i, y_true == i))
    result = np.mean(total)
    # total.append(np.mean(total))
    return result

