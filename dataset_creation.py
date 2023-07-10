import torch
import torch_geometric
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor, Lambda
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
import graph_utils
from tqdm import tqdm
# torch.manual_seed(1)
# np.random.seed(1)


print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class Cifar10_graphs(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, test=False, k_neighbors=8,  use_knn=False, nodes=100, complete_graph=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.k_neighbors = k_neighbors
        self.use_knn = use_knn
        self.nodes = nodes
        self.complete = complete_graph
        
        if(not test):
            self.data = CIFAR10("data/raw/", download=True, transform=graph_utils.be_np, 
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
        else:
            self.data = CIFAR10("data/raw", download=True, transform=graph_utils.be_np, train=False,
                target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

        super(Cifar10_graphs, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return 'cifar-10-python.tar.gz'

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        if self.test:
            return [f'superpixel_hierarchy_data_test_{i}.pt' for i in range(len(self.data))]
        else:
            return [f'superpixel_hierarchy_data_{i}.pt' for i in range(len(self.data))]
            
        
    def download(self):
        pass

    def process(self):

        
        if(self.use_knn):
            if(self.complete):
                print("USING SUPER PIXEL HIERARCHY COMPLETE GRAPH")
            else:
                print("USING SUPER PIXEL HIERARCHY "+ str(self.k_neighbors) + "-NN ADJACENCY")
        else:
                print("USING SUPER PIXEL HIERARCHY WITH HIERARCHY ADJACENCY")
            
        if(self.test):
            for i in tqdm(range(len(self.data))):
                #print(i)
                node_features, coo, edge_features, pos = graph_utils.superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, knn=self.use_knn,
                                                                                        k_neighbors=self.k_neighbors,
                                                                                        complete=self.complete)
                data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=torch.argmax(self.data[i][1]),#self.data[i][1],
                            pos=torch.from_numpy(pos))
                path = os.path.join('data/processed/', f'superpixel_hierarchy_data_test_{i}.pt')
                torch.save(data, path)
        else:
            for i in tqdm(range(len(self.data))):
                #print(i)
                node_features, coo, edge_features, pos = graph_utils.superpixel_hierarchy(self.data[i][0], n_nodes=self.nodes, knn=self.use_knn,
                                                                                        k_neighbors=self.k_neighbors,
                                                                                        complete=self.complete)
                data = Data(x= torch.from_numpy(node_features).type(torch.FloatTensor),
                            edge_index=torch.from_numpy(coo).type(torch.LongTensor),
                            edge_attr=torch.from_numpy(edge_features),
                            y=torch.argmax(self.data[i][1]),#self.data[i][1],
                            pos=torch.from_numpy(pos))
                path = os.path.join('data/processed/', f'superpixel_hierarchy_data_{i}.pt')
                torch.save(data, path)
           
    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return len(self.data)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'superpixel_hierarchy_data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'superpixel_hierarchy_data_{idx}.pt'))        
        return data
            