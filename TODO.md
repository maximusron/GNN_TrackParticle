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