from __future__ import division
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def accuracy_K(pred,label):
    Total=len(pred)
    acc1=0
    acc5=0
    for index in range(Total):
        top1 = np.argpartition(a=-pred[index], kth=1)[:1]
        top5 = np.argpartition(a=-pred[index], kth=5)[:5]
        if label[index] in top1:
            acc1+= 1
        if label[index] in top5:
            acc5+= 1
    acc1_rate=acc1/Total
    acc5_rate=acc5/Total
    return acc1_rate,acc5_rate

def accuracy_at_k(predicted_labels, true_labels, k):
    if len(predicted_labels) != len(true_labels):
        raise ValueError("Predicted labels and true labels must have the same length.")

    total_samples = len(predicted_labels)
    correct_at_k = 0

    for i in range(total_samples):
        if isinstance(predicted_labels[i], (list, np.ndarray)):
            top_k_predictions = predicted_labels[i][:k]
        else:
            # Handle single integer predictions
            top_k_predictions = [predicted_labels[i]]

        if true_labels[i] in top_k_predictions:
            correct_at_k += 1

    accuracy = correct_at_k / total_samples
    return accuracy

def macro_F(pred,label):
    pred_label=[]
    Total = len(pred)
    for i in range(Total):
        pred_label.append(np.argmax(pred[i]))
    macro_r=recall_score(label[:Total],pred_label,average='macro')
    macro_p=precision_score(label[:Total],pred_label,average='macro')
    macro_f=2*macro_p*macro_r/(macro_p+macro_r)
    return macro_f,macro_r,macro_p

def calculate_macro_metrics(predicted_labels, true_labels):
    # Calculate precision, recall and F1 score
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)
    return precision, recall, f1



