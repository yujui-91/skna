import pandas as pd
import numpy as np


def metrics(total_cm):
    tp = total_cm[1, 1].astype(np.float32)
    fp = total_cm[0, 1].astype(np.float32)
    fn = total_cm[1, 0].astype(np.float32)
    tn = total_cm[0, 0].astype(np.float32)

    cm = {
        "predict/actual": ["Positive", "Negative"],
        "Positive": [tp, fn],
        "Negative": [fp, tn]
    }

    cm_df = pd.DataFrame(cm)
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 4) if (tp + tn + fp + fn) != 0 else 0
    sensitivity = round(tp / (tp + fn), 4) if (tp + fn) != 0 else 0
    specificity = round(tn / (tn + fp), 4) if (tn + fp) != 0 else 0
    ppv = round(tp / (tp + fp), 4) if (tp + fp) != 0 else 0
    npv = round(tn / (tn + fn), 4) if (tn + fn) != 0 else 0
    mcc = round((tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5), 4) if ((tp + fp) * (
                tp + fn) * (tn + fp) * (tn + fn)) != 0 else 0

    metric = {
        "Accuracy": ["Sensitivity", "Specificity", "PPV", "NPV", "MCC"],
        accuracy: [sensitivity, specificity, ppv, npv, mcc]
    }
    metric_df = pd.DataFrame(metric)
    return cm_df, metric_df