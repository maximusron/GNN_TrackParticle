# GNN Tracking

My solutions and submission to the qualification task of the GSoC 2023 “GNN Tracking” project under CERN, IRIS-HEP


## TODO for Machine learning & statistics 

Deliverable: A Jupyter Notebook completing the task below.

**Task: Edge classification with pytorch geometric**

Train an Edge Classifier Graph Neural Network to classify the edges (given by the edge_index) as true or false (given by the array y). 

**Inputs** for training/inference :
  - x (the node features)                       
  - edge_features.
  
**Basic Plan**
- Use binary cross-entropy as loss function for the classification. 

- Look into other loss functions

- First use basic accuracy to evaluate the model

- Modify the model evalutation technique to acknowledge the class imbalance 

**Note:** 
- The basic structure of the code -> similar to last section of pytorch geometric tutorial.


**Bonuses:**

- Extract relevant reusable code (models etc.) to a python library/python files.

- Include the model in the gnn_tracking/models/ module of the forked gnn_tracking repository (see instructions for the software engineering tasks) and find good locations for other reusable components that you have created.

**If time allows**
- Use GraphGym to illustrate experiments and performance of different architecturesand losses

- Use LinkLoader for negative sampling

- VGAE


it works correctly with >> symmetrize_edge_weights(T([[1, 2], [2, 1]]), T([1, 3]))
T([2, 2])
But i am getting this error:
RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton dimension 0 
while giiving input symmetrize_edge_weights(T([[1, 2], [3, 4], [2, 1]]), T([1, 2, 3]))

fix:
use long
def symmetrize_edge_weights(edge_indices: T, edge_weights: T) -> T:
    # create a dictionary to store the symmetric edge weights
    symmetric_weights = {}
    for i, (a1, a2) in enumerate(edge_indices.long().t()):
        # check if the symmetric edge exists
        if (a2, a1) in symmetric_weights:
            # calculate the average of the two edge weights
            avg_weight = (symmetric_weights[(a2, a1)] + edge_weights[i]) / 2
            # update the symmetric edge weights
            symmetric_weights[(a2, a1)] = avg_weight
            symmetric_weights[(a1, a2)] = avg_weight
        else:
            # add the edge weight to the dictionary
            symmetric_weights[(a1, a2)] = edge_weights[i]
    
    # create an array to store the symmetric edge weights
    symmetric_edge_weights = T([symmetric_weights[(a1, a2)] for a1, a2 in edge_indices.long().t()])
    return symmetric_edge_weights
