import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, f1_score

import wandb
import random

from src.model_helpers import EarlyStopping


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels = 2, p=0.5):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, output_channels)
        self.batchnorm1 = BatchNorm(hidden_channels)
        self.batchnorm2 = BatchNorm(hidden_channels)
        self.p = p

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_weight)
        x = self.batchnorm1(x)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_weight)
        x = self.batchnorm2(x)
        x = x.relu()
        
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.lin(x)
        
        return x

class GCNTrainer:
    def __init__(self, model, loss_func, optimizer, train_loader, test_loader, scheduler=None, output_dim=2):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        # self.output_dim = output_dim
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self):
        self.model.train()
        for data in self.train_loader:

            data = data.to(self.device)
            
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            # loss = self.loss_func(out.squeeze(), data.y.float()) if self.output_dim == 1 else self.loss_func(out, data.y) 
            loss = self.loss_func(out, data.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def test(self, loader):
        self.model.eval()
        correct = 0
        total_loss = 0
        y_true = []
        y_pred = []
        
        with torch.no_grad(): # improves efficiency ? during evaluation gradients do not need to be computed
            for data in loader:
                data = data.to(self.device)

                out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
               
                # pred = (out.squeeze() > 0.5).int() if self.output_dim == 1 else out.argmax(dim=1)
                pred = out.argmax(dim=1)
                
                correct += int((pred == data.y).sum())
                
                # if self.output_dim == 1:
                #     total_loss += float(self.loss_func(out.squeeze(), data.y.float()))
                # else:
                #     total_loss += float(self.loss_func(out, data.y))
                total_loss += float(self.loss_func(out, data.y))
                
                y_true += data.y.tolist()
                y_pred += pred.tolist()
                
                
        accuracy = correct / len(loader.dataset) 
        average_loss = total_loss / len(loader) # give average for whole test set rather than just the batch
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Sensitivity - true positive rate
        sensitivity = tp / (tp + fn)
        # Specificity - true negative rate
        specificity = tn / (tn + fp)
        # f1 score
        f1 = f1_score(y_true, y_pred)
        
        return accuracy, average_loss, sensitivity, specificity, f1

    def run(
        self, 
        epochs, 
        use_wandb=False,
        path:str = None,
        early_stop={
            'patience': 20, 
            'min_delta': 0
            }, 
        abort_on_thresh=0
        ):
        
        train_accuracy = []
        test_accuracy = []
        train_loss = []
        test_loss = []
        train_sensitivity = []
        test_sensitivity = []
        train_specificity = []
        test_specificity = []
        train_f1 = []
        test_f1 = []
        
        best_test_f1 = 0.0
        
        if early_stop:
            patience = early_stop['patience']
            min_delta = early_stop['min_delta']
            print(f'Early stopping enabled. Patience: {patience}. Min Delta: {min_delta}.')
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
            
        if self.scheduler is not None:
            print(f'Learning rate scheduler enabled. Patience: {self.scheduler.patience}. Factor: {self.scheduler.factor}.')
            prev_lr = self.optimizer.param_groups[0]['lr']
            print(f'Initial learning rate: {prev_lr}')
    
        for epoch in range(0, epochs):
            
            self.train()
            
            tracc, trlss, trsens, trspec, trf1 = self.test(self.train_loader)
            
            train_accuracy.append(tracc)
            train_loss.append(trlss)
            train_sensitivity.append(trsens)
            train_specificity.append(trspec)
            train_f1.append(trf1)
            
            teacc, telss, tesens, tespec, tef1 = self.test(self.test_loader)
            
            test_accuracy.append(teacc)
            test_loss.append(telss)
            test_sensitivity.append(tesens)
            test_specificity.append(tespec)
            test_f1.append(tef1)
            
            if use_wandb:
                wandb.log({
                    "Train Accuracy": tracc, 
                    "Train Loss": trlss,
                    "Train Sensitivity": trsens,
                    "Train Specificity": trspec,
                    "Train F1": trf1,
                    "Test Accuracy": teacc, 
                    "Test Loss": telss,
                    "Test Sensitivity": tesens,
                    "Test Specificity": tespec,
                    "Test F1": tef1
                    })
            
            if tef1 > best_test_f1:
                best_test_f1 = tef1
                if path:
                    print('saving model')
                    torch.save(self.model, f"{path}/{best_test_f1:.3f}_{epoch}.pth")
                    torch.save(self.model.state_dict(), f"{path}/{best_test_f1:.3f}_{epoch}_dict.pth")

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Train Acc: {tracc:.4f}, Test Acc: {teacc:.4f}, Train Loss: {trlss:.4f}, Test Loss: {telss:.4f}')

            if self.scheduler is not None:
                self.scheduler.step(telss)
                current_lr = self.optimizer.param_groups[0]['lr']
                if current_lr != prev_lr:
                    print(f'Epoch: {epoch:03d}, Learning rate changed from {prev_lr} to {current_lr}')
                    prev_lr = current_lr
                
            if abort_on_thresh:
                if teacc > abort_on_thresh:
                    if epoch % 10 != 0:
                        print(f'Epoch: {epoch:03d}, Train Acc: {tracc:.4f}, Test Acc: {teacc:.4f}, Train Loss: {trlss:.4f}, Test Loss: {telss:.4f}')
                    print(f"Accuracy threshold of {abort_on_thresh} reached. Stopping training.")
                    break
                
            if early_stop:
                early_stopping(telss)
                if early_stopping.early_stop:
                    print(f"{patience} epochs passed without {min_delta} test loss improvement. \nEarly stopping triggered.")
                    break

        if use_wandb:
            wandb.finish()

        if abort_on_thresh:
            return train_accuracy, test_accuracy, train_loss, test_loss, epoch
        else:
            return train_accuracy, test_accuracy, train_loss, test_loss


def load(dataset, 
         batch_size,
         shuffle_dataset=True, 
         train_split:int = 0.7,
         test_split:int = 0.15,
         val_split:int = 0.15
         ):
    
    dataset_copy = dataset.copy()
    
    assert train_split + test_split + val_split == 1, "Split values must sum to 1"
    
    train_cutoff = int(len(dataset) * train_split)
    test_cutoff = int(len(dataset) * (train_split + test_split))
    
    # todo: will need to edit to keep a conistent val_set, one which isn't subject to random shuffle
    # need this to keep consistent with model selection
    
    if shuffle_dataset:
        random.shuffle(dataset_copy)
    
    train_dataset = dataset_copy[:train_cutoff]
    test_dataset = dataset_copy[train_cutoff:test_cutoff]
    val_dataset = dataset_copy[test_cutoff:]
    
    dataset_split = {'train': train_dataset,
                     'test': test_dataset,
                     'val': val_dataset}
    
    # print('Train dataset length:', len(train_dataset))
    # print('Test dataset length:', len(test_dataset))
    # print('Validation dataset length:', len(val_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader, dataset_split