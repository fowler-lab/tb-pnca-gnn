import torch
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_curve, roc_auc_score
import numpy as np

def calculate_sens_spec(model, data_loader):
    model.eval()

    y_preds = torch.tensor([])
    y_trues = torch.tensor([])
    for data in data_loader:  # Iterate in batches over the test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # pred = out[:, 1]
        # print(pred)
        
        y_preds = torch.cat((y_preds, pred), 0)
        y_trues = torch.cat((y_trues, data.y), 0)

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
    
    print(f"Sensitivity: {sensitivity:.2f}")
    print(f"Specificity: {specificity:.2f}")
    
    return sensitivity, specificity
    

def calculate_roc(model, data_loader):
    
    model.eval()

    y_preds = torch.tensor([])
    y_trues = torch.tensor([])
    for data in data_loader:  # Iterate in batches over the test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # pred = out.argmax(dim=1)  # Use the class with highest probability.
        pred = out[:, 1]
        # print(pred)
        
        y_preds = torch.cat((y_preds, pred), 0)
        y_trues = torch.cat((y_trues, data.y), 0)

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