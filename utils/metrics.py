def accuracy(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)


def precision(y_true, y_pred):
    tp = sum((y_true == y_pred) & (y_pred == 1))
    fp = sum((y_pred == 1) & (y_true != 1))
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def recall(y_true, y_pred):
    tp = sum((y_true == y_pred) & (y_pred == 1))
    fn = sum((y_pred != 1) & (y_true == 1))
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    if rec + prec:
        return 0
    return 2 * (prec * rec) / (prec + rec)
