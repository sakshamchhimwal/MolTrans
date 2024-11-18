import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, recall_score, precision_score, \
    roc_auc_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(data_loader, model):
    predictions = []
    true_labels = []
    model.eval()
    total_loss = 0.0
    batch_count = 0.0
    loss_function = torch.nn.BCELoss()

    for i, (drug_seq, protein_seq, drug_mask, protein_mask, labels) in enumerate(data_loader):
        # Move all inputs to the same device
        drug_seq = drug_seq.long().to(device)
        protein_seq = protein_seq.long().to(device)
        drug_mask = drug_mask.long().to(device)
        protein_mask = protein_mask.long().to(device)
        labels = torch.tensor(labels, dtype=torch.float).to(device)

        scores = model(
            drug_seq,
            protein_seq,
            drug_mask,
            protein_mask,
        )

        sigmoid = torch.nn.Sigmoid()
        probabilities = torch.squeeze(sigmoid(scores))

        batch_loss = loss_function(probabilities, labels)
        total_loss += batch_loss
        batch_count += 1

        # Move predictions and labels to CPU for numpy operations
        probabilities = probabilities.detach().cpu().numpy()
        label_ids = labels.cpu().numpy()

        true_labels.extend(label_ids.flatten().tolist())
        predictions.extend(probabilities.flatten().tolist())

    avg_loss = total_loss / batch_count

    false_positive_rate, true_positive_rate, thresholds = roc_curve(true_labels, predictions)
    precision_scores = true_positive_rate / (true_positive_rate + false_positive_rate)
    f1_scores = 2 * precision_scores * true_positive_rate / (true_positive_rate + precision_scores + 1e-5)
    optimal_threshold = thresholds[5:][np.argmax(f1_scores[5:])]


    binary_predictions = [1 if pred >= optimal_threshold else 0 for pred in predictions]

    confusion_mat = confusion_matrix(true_labels, binary_predictions)

    # auc_score = auc(false_positive_rate, true_positive_rate)
    # print("Confusion Matrix:\n", confusion_mat)
    # print("Optimal Threshold: " + str(optimal_threshold))
    # total_samples = sum(sum(confusion_mat))
    # accuracy = (confusion_mat[0, 0] + confusion_mat[1, 1]) / total_samples
    # print("Accuracy: ", accuracy)
    # sensitivity = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
    # print("Sensitivity: ", sensitivity)
    # specificity = confusion_mat[1, 1] / (confusion_mat[1, 0] + confusion_mat[1, 1])
    # print("Specificity: ", specificity)

    outputs = np.asarray([1 if pred >= 0.5 else 0 for pred in predictions])
    return (
        roc_auc_score(true_labels, predictions),
        average_precision_score(true_labels, predictions),
        f1_score(true_labels, outputs),
        predictions,
        avg_loss.item(),
        confusion_mat,
    )