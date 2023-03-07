import torch
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import Dataset 


class BatchData():

    # def __init__(self, idx, idx2):
    #     super().__init__()
    #     self.path = Path("../data/batch_1_{}/".format(idx))
    #     self.graphs = list(self.path.glob("*.pt"))
    #     print(len(self.graphs))
    
    def __init__(self, idx1, idx2):
        super().__init__()
        self.graphs = []
        print("Loading:")
        for i in range(idx1, idx2):
            self.path = Path("../data/batch_1_{}/".format(i))
            print(self.path)
            self.graphs += list(self.path.glob("*.pt")) 

    def __getitem__(self, idx):
        return torch.load(self.graphs[idx])
    
    def __len__(self) -> int:
        return len(self.graphs)
    

    
    





