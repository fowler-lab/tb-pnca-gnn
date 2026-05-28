import os
import random

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import confusion_matrix, f1_score
import wandb
import e3nn
from e3nn import o3
from e3nn.math import soft_one_hot_linspace

from src.model_helpers import EarlyStopping


class EGNNConvToy(nn.Module):
    def __init__(self, irreps_in: str, irreps_mid: str, irreps_out: str, irreps_sh: str, num_basis: int = 10, max_radius: float = 1.0):
        super(EGNNConvToy, self).__init__()
        
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_mid = o3.Irreps(irreps_mid)
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_sh = o3.Irreps(irreps_sh)
        
        self.num_basis = num_basis
        self.max_radius = max_radius
    
        self.conv1 = ConvLayer(self.irreps_in, self.irreps_mid, self.irreps_sh, self.num_basis, self.max_radius)
        self.conv2 = ConvLayer(self.irreps_mid, self.irreps_out, self.irreps_sh, self.num_basis, self.max_radius)
        self.out = nn.Linear(self.irreps_out.dim, 2)
        
    def forward(self, f_in, node_positions, batch=None):
        
        x = self.conv1(f_in, node_positions, batch=batch)
        x = self.conv2(x, node_positions, batch=batch)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_mean_pool(x, batch)
        x = self.out(x)
        
        return x

class ConvLayer(nn.Module):
    def __init__(self, irreps_in: str, irreps_out: str, irreps_sh: str, num_basis: int = 10, max_radius: float = 1.0):
        super(ConvLayer, self).__init__()
        
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_sh = irreps_sh
        
        self.num_basis = num_basis
        self.max_radius = max_radius
        
        self.tp = o3.FullyConnectedTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out, shared_weights=False)

        # self.fc = e3nn.nn.FullyConnected(self.num_basis, self.tp.weight_numel)
        self.fc = nn.Linear(self.num_basis, self.tp.weight_numel)
    
    def conv(self, f_in, node_positions, batch=None):
    
        # get edge source and target nodes
        edge_src, edge_dst = radius_graph(
            node_positions,
            self.max_radius,
            batch=batch,
            max_num_neighbors=len(node_positions) - 1,
        )
        
        # get edge vectors (source -> target)
        edge_vec = node_positions[edge_dst] - node_positions[edge_src]
        
        # compute spherical harmonics
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        
        # use basis function to embed edge lengths
        emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, self.max_radius, self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)
        # and pass these into MLP
        emb = self.fc(emb)
        
        # do tensor product of x, y and w
        tp_result = self.tp(f_in[edge_src], sh, emb)
        
        # sum per target node and normalise
        num_neighbours = len(edge_src) / len(node_positions)
        f_out = scatter(tp_result, edge_dst, dim=0, dim_size=len(node_positions)).div(num_neighbours**0.5)
        
        return f_out
    
    def forward(self, f_in, node_positions, batch=None):
        return self.conv(f_in, node_positions, batch=batch)
    
class EGNNTrainer:
    def __init__(
        self,
        model,
        loss_func,
        optimizer,
        train_loader,
        test_loader,
        scheduler=None,
        output_dim=2,
        node_positions=None,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.node_positions = node_positions
        # self.output_dim = output_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        self.model.train()
        for data in self.train_loader:

            data = data.to(self.device)

            node_positions = self._get_node_positions(data)
            batch = data.batch if hasattr(data, "batch") else None
            out = self.model(data.x, node_positions, batch=batch)
            out = self._maybe_pool_graph_output(out, data)
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

        with torch.no_grad():  # improves efficiency ? during evaluation gradients do not need to be computed
            for data in loader:
                data = data.to(self.device)

                node_positions = self._get_node_positions(data)
                batch = data.batch if hasattr(data, "batch") else None
                out = self.model(data.x, node_positions, batch=batch)
                out = self._maybe_pool_graph_output(out, data)

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
        average_loss = total_loss / len(
            loader
        )  # give average for whole test set rather than just the batch

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Sensitivity - true positive rate
        sensitivity = tp / (tp + fn)
        # Specificity - true negative rate
        specificity = tn / (tn + fp)
        # f1 score
        f1 = f1_score(y_true, y_pred)

        return accuracy, average_loss, sensitivity, specificity, f1

    @staticmethod
    def _maybe_pool_graph_output(out, data):
        y_size = data.y.size(0) if data.y.dim() > 0 else 1
        out_features = out.unsqueeze(1) if out.dim() == 1 else out

        if out_features.size(0) != y_size:
            if hasattr(data, "batch") and data.batch is not None:
                batch = data.batch
            else:
                batch = torch.zeros(
                    out_features.size(0), dtype=torch.long, device=out_features.device
                )
            out_features = global_mean_pool(out_features, batch)

        return out_features.squeeze(1) if out.dim() == 1 else out_features

    def _get_node_positions(self, data):
        if hasattr(data, "pos") and data.pos is not None:
            return data.pos
        if hasattr(data, "node_positions") and data.node_positions is not None:
            return data.node_positions
        if self.node_positions is not None:
            return torch.as_tensor(
                self.node_positions,
                dtype=data.x.dtype,
                device=data.x.device,
            )
        raise AttributeError(
            "EGNNTrainer expects data.pos or data.node_positions for node coordinates."
        )

    def run(
        self,
        epochs,
        use_wandb=False,
        path: str = None,
        early_stop={"patience": 20, "min_delta": 0},
        abort_on_thresh=0,
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
        best_model_path = None

        if early_stop:
            patience = early_stop["patience"]
            min_delta = early_stop["min_delta"]
            print(
                f"Early stopping enabled. Patience: {patience}. Min Delta: {min_delta}."
            )
            early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

        # if self.scheduler is not None:
        #     print(f'Learning rate scheduler enabled. Patience: {self.scheduler.patience}. Factor: {self.scheduler.factor}.')
        #     prev_lr = self.optimizer.param_groups[0]['lr']
        #     print(f'Initial learning rate: {prev_lr}')

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
                wandb.log(
                    {
                        "Train Accuracy": tracc,
                        "Train Loss": trlss,
                        "Train Sensitivity": trsens,
                        "Train Specificity": trspec,
                        "Train F1": trf1,
                        "Test Accuracy": teacc,
                        "Test Loss": telss,
                        "Test Sensitivity": tesens,
                        "Test Specificity": tespec,
                        "Test F1": tef1,
                    }
                )

            if tef1 > best_test_f1:

                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                    os.remove(f"{best_model_path}".replace(".pth", "_dict.pth"))

                best_test_f1 = tef1
                if path:
                    # print('saving model')
                    os.makedirs(f"{path}", exist_ok=True)

                    best_model_path = f"{path}/F1={best_test_f1:.3f}_epoch={epoch}.pth"

                    torch.save(
                        self.model, f"{path}/F1={best_test_f1:.3f}_epoch={epoch}.pth"
                    )
                    torch.save(
                        self.model.state_dict(),
                        f"{path}/F1={best_test_f1:.3f}_epoch={epoch}_dict.pth",
                    )

            if epoch % 10 == 0:
                print(
                    f"Epoch: {epoch:03d}, Train Acc: {tracc:.4f}, Test Acc: {teacc:.4f}, Train Loss: {trlss:.4f}, Test Loss: {telss:.4f}"
                )

            if self.scheduler is not None:
                # self.scheduler.step(telss)
                self.scheduler.step()
                # current_lr = self.optimizer.param_groups[0]['lr']
                # if current_lr != prev_lr:
                #     print(f'Epoch: {epoch:03d}, Learning rate changed from {prev_lr} to {current_lr}')
                #     prev_lr = current_lr

            if abort_on_thresh:
                if teacc > abort_on_thresh:
                    if epoch % 10 != 0:
                        print(
                            f"Epoch: {epoch:03d}, Train Acc: {tracc:.4f}, Test Acc: {teacc:.4f}, Train Loss: {trlss:.4f}, Test Loss: {telss:.4f}"
                        )
                    print(
                        f"Accuracy threshold of {abort_on_thresh} reached. Stopping training."
                    )
                    break

            if early_stop:
                early_stopping(telss)
                if early_stopping.early_stop:
                    print(
                        f"{patience} epochs passed without {min_delta} test loss improvement. \nEarly stopping triggered."
                    )
                    break

        if use_wandb:
            wandb.finish()

        if abort_on_thresh:
            return train_accuracy, test_accuracy, train_loss, test_loss, epoch
        else:
            return train_accuracy, test_accuracy, train_loss, test_loss


def load(
    dataset,
    batch_size,
    shuffle_dataset=True,
    train_split: int = 0.7,
    test_split: int = 0.3,
    val_split: int = 0,
):

    dataset_copy = dataset.copy()

    assert train_split + test_split + val_split == 1, "Split values must sum to 1"

    train_cutoff = int(len(dataset) * train_split)
    test_cutoff = int(len(dataset) * (train_split + test_split))

    if shuffle_dataset:
        random.shuffle(dataset_copy)

    train_dataset = dataset_copy[:train_cutoff]
    test_dataset = dataset_copy[train_cutoff:test_cutoff]
    val_dataset = dataset_copy[test_cutoff:]

    dataset_split = {"train": train_dataset, "test": test_dataset, "val": val_dataset}

    # print('Train dataset length:', len(train_dataset))
    # print('Test dataset length:', len(test_dataset))
    # print('Validation dataset length:', len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader, dataset_split