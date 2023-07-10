import torch
import numpy as np
import mlflow.pytorch
from dataset_creation import Cifar10_graphs
from sklearn.metrics import  accuracy_score
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from model import Loop_net 
torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)

NUM_CLASSES=10
BATCH_SIZE=64
EPOCHS=1000

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Experiment 1")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def val(epoch, model, val_loader, loss_fn, device):
    all_preds=[]
    all_labels=[]
    model.eval()
    for batch in val_loader:
        batch.to(device)
        pred = model(batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch)
        loss = loss_fn(pred, batch.y.reshape(len(batch),10))
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(np.argmax(batch.y.reshape(len(batch),10).cpu().detach().numpy(), axis=1))
    all_preds=np.concatenate(all_preds).ravel()
    all_labels=np.concatenate(all_labels).ravel()
    accu = calculate_metrics(all_preds, all_labels, epoch, "val")
    return loss, accu

def calculate_metrics (y_pred , y_true , epoch, type) :
    print (f"{type} Accuracy:{accuracy_score(y_pred, y_true)}")
    mlflow.log_metric(key=f"Accuracy-{type}", value=float(accuracy_score(y_pred, y_true)), step=epoch)
    return accuracy_score(y_pred, y_true)

def train(epoch, train_loader, device, optimizer, model, loss_fn):
    all_preds = []
    all_labels = []
    for batch in train_loader:
        batch.to(device)
        optimizer.zero_grad()
        pred=model(batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch)

        loss = loss_fn(pred, batch.y.reshape(len(batch),10))
        loss.backward()
        optimizer.step()
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(np.argmax(batch.y.reshape(len(batch),10).cpu().detach().numpy(), axis=1))
    all_preds=np.concatenate(all_preds).ravel()
    all_labels=np.concatenate(all_labels).ravel()
    accu = calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss, accu, optimizer

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = Cifar10_graphs(root="data/", nodes=20, k_neighbors=8, use_knn=True, complete_graph=True)#LOAD THE DATASET  
    
    train_indices=np.load("train_indices.npy").tolist()
    val_indices=np.load("val_indices.npy").tolist()
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    

    train_loader = DataLoader(data, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(data, batch_size=int(BATCH_SIZE), sampler=valid_sampler)
    
    model = Loop_net(node_feature_size=data[0].x.shape[1], edge_feature_size=data[0].edge_attr.shape[1], num_classes=data[0].y.shape[0], embedding_size=70)
    model = model.to(device)    
    print(f"Number of parameters: {count_parameters(model)}") 
    
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)                                                                               
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, min_lr=1e-6, factor=0.5, verbose=True)
    best_accu=0
    print("-----------------------------------------------------------")
    with mlflow.start_run() as run:
        for epoch in range(EPOCHS):
            model.train()
            loss, accu, optimizer = train(epoch=epoch, train_loader=train_loader, device=device, optimizer=optimizer, model=model, loss_fn=loss_fn)           
            print(f"Epoch {epoch+1} | Train Loss {loss}")
            mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
            print("--")
            
            ##### Validation
            model.eval()
            epoch_val_loss, epoch_val_accu = val(epoch=epoch, model=model, val_loader=val_loader, device=device, loss_fn=loss_fn)
            print(f"Epoch {epoch+1} | Val Loss {epoch_val_loss}")
            print("-----------------------------------------------------------")
            mlflow.log_metric(key="val loss", value=float(epoch_val_loss), step=epoch)
            if(epoch_val_accu >= best_accu):
                best_accu = epoch_val_accu
                torch.save(model.state_dict(), "weights/hierarchy_adj.pth")
            scheduler.step(epoch_val_accu)

    print("Done")



if __name__ == "__main__":
    main()