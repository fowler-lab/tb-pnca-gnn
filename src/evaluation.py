import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix
import numpy as np

def calculate_metrics(model, data_loader):
    eval_model = model.to('cpu')
    eval_model.eval()

    y_preds = torch.tensor([])
    y_trues = torch.tensor([])
    for data in data_loader:  # Iterate in batches over the test dataset.
        eval_data = data.to('cpu')
        out = eval_model(eval_data.x, eval_data.edge_index, eval_data.edge_attr, eval_data.batch)
        
        if eval_model.lin.out_features == 1:
            pred = (out > 0.5).float()  # Convert output to binary predictions using a threshold of 0.5
        else:
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            
        # pred = out[:, 1]
        # print(pred)
        
        y_preds = torch.cat((y_preds, pred), 0)
        y_trues = torch.cat((y_trues, eval_data.y), 0)

    y_preds = y_preds.detach().numpy()
    # print(y_preds)
    
    num_pred_pos = np.count_nonzero(y_preds==1)
    num_pred_neg = np.count_nonzero(y_preds==0)
    num_true_pos = np.count_nonzero(y_trues==1)
    num_true_neg = np.count_nonzero(y_trues==0)
    
    print(f'Number predicted resistant = {num_pred_pos}. Number labeled R = {num_true_pos}')
    print(f'Number predicted susceptible = {num_pred_neg}. Number labeled S = {num_true_neg}')
    
    sensitivity = recall_score(y_trues, y_preds)
    specificity = recall_score(y_trues, y_preds, pos_label=0)
    f1 = f1_score(y_trues, y_preds)
    cm = confusion_matrix(y_trues, y_preds)

    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    return sensitivity, specificity, f1, cm


def calculate_roc(model, data_loader):
    eval_model = model.to('cpu')
    eval_model.eval()

    y_preds = torch.tensor([])
    y_trues = torch.tensor([])
    for data in data_loader:  # Iterate in batches over the test dataset.
        eval_data = data.to('cpu')
        out = eval_model(eval_data.x, eval_data.edge_index, eval_data.edge_attr, eval_data.batch)
        
        if eval_model.lin.out_features == 1:
            pred = out.squeeze() 
        else:
            pred = out[:, 1]
        # print(pred)
        
        y_preds = torch.cat((y_preds, pred), 0)
        y_trues = torch.cat((y_trues, eval_data.y), 0)

    y_preds = y_preds.detach().numpy()

    # ROC: TPR vs FPR
    # recall (sensitivity) vs 1-specificity
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    roc_auc = roc_auc_score(y_trues, y_preds)
    
    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc = None):
    
    label = f'ROC curve (AUC = {roc_auc:.2f})' if roc_auc else 'ROC curve'
        
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=label)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Senstivity)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()