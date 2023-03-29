## Graph Neural Networks: A Variational Graph Autoencoder model for link prediction.

The utils are the files required to set up a framework to develop the architecture further.

#### Handling class imbalance :

- So as expected the negative samples (i.e., pairs of nodes that are not connected) are greater than the positive samples (i.e., pairs of nodes that are connected)

    - Stratified sampling that can be used to ensure that the training set contains an equal number of positive and negative samples.

    - Random selection of a subset of the negative samples for training, instead of using all of them. This follows the idea of reducing the degree of class imbalance intead of trying to balance the classes directly

    -  To address the fact that the negative edges will overlap with the positive edges and negative edges in other sets. The sample is discarded from the training set to emulate a disjoint sampling algorithm

    Result:
    ```bash
    **Percentage of positive edges:  17.673395558091762**
    **Percentage of negative edges:  82.32660444190823**
    ```

    - Ensemble methods: This goes under future prospective improvement directions. multiple models with different negative sampling strategies and combine their predictions to obtain a more balanced and accurate model.
 
The new architecture learns the node embeddings by aggregating and transforming neighboring node features in the graph. These embeddings are representative of the graph as a whole and should improve the performance of edge classification.
 
The node embeddings are learnt via the following architecture:

- Linear Transformation: first apply a linear transformation to the input node features using a weight matrix. 
Why? the learned weight matrix is basically a unique representation for each node.

- Message Passing: perform message passing aggregated from the neighboring nodes of each node in the graph(using ```torch.geomtric.MessagePassing```). 
   
    - Algorithm:
        - taking a weighted_sum(features of neighboring nodes)   *(where the weights are learned using another weight matrix)*. 
        - original node features += weighted_sum, *this updates the node representation to include info from the neighboring nodes.*

- Non-linearity: pass new node features through non-linear activation function, which introduces non-linearity to the layer and allows it to learn complex representations.

- stack GCN layers, end with ReLu, should learn complex and informative representations of the graph. These node embeddings capture both local and global information from the graph 

The aggregation emulates learned representations using features of its neighbors in the graph instead of just the nodes. This should allow the GCN layer to capture structural dependencies between nodes.

The similarity score of two node embeddings is used to decide whether they should be connected.

**Experiments**

Different decoders are tried, varying from a dot product model for reduction to a symmetric mirror of the encoder that can be reduced to probabilities

The embeddings are extracted from the layer using:
- MLP with two linear layers and a ReLU activation function between them. 
- That takes the latent representation of each node as input and generates a vector of the same dimension as the input features.

Net model:
```
VGAE forward -> encoder->latent space -> decoder-> embeddings-> dot product->sigmoid-> link probability
```

The model is now updated so that the latent representation makes predictions about missing edges.

Experimenting with both dense adjacency matrices and sparse edge representation matrices resulted in the following observations:
- `edge_index` contains only essential info about the graph, which makes it easier to learn meaningful patterns. 
- The dense adjacency matrix contains a lot of redundant information which leads to overfitting
- Computationally, sparse edge representations win without a doubt, considering only the edges that are present in the graph instead of all combinatorial  pairs. With a class imbalance in this dataset, we're looking at over 80% of node pairs that don't have edges.

Note: GCNConv only supports one-dimensional edge attributes so a replacement layer(or aggregation network) has to be utilised to encode the `edge_attr`
- This can also be achieved using a custom message passing layer that performs an of aggregation of `x` and `edge_attr`
- A simpler solution is the passing of `edge_attr` into GCN by concatenation with x.
- The model is tested which shows that 76% edge attributes are actually informative while the rest are noisy.

**Improvements** :

The architecture is migrated from:
```python
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
```

to

```python
class Encoder(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, latent_dims, num_layers):
        super(Encoder, self).__init__()
        self.linear = torch.nn.Linear(num_node_features, hidden_channels)
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers-2):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels)) # GCN layers 2 to num_layers-1
        self.conv_layers.append(GCNConv(hidden_channels, latent_dims)) # GCN layer num_layers
        self.num_layers = num_layers

    def forward(self, x, edge_index):
    # Linear transformation
        x = self.linear(x)
        x = F.relu(x)
        # Message passing
        for i in range(self.num_layers-1):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
        # Non-linearity
        x = self.conv_layers[-1](x, edge_index)
        return x
```

### LOSSES

I have utilised ROC and AP:

- ROC curve evaluates the link prediction by checking TPR vs FPR at different link probs(*low priority: add a confusion matrix*). The area under the ROC measures performance in distinguishing between positive and negative samples, independent of the link probability threshold. 

- Average precision (AP) is a really good metric here since the positive class is rare. AP measures the area under the precision-recall curve, which plots the precision against the recall at different thresholds of predicted link probabilities. so it evaluates the model's ability to rank pairs of nodes based on their likelihood of being linked. 

**Hence, they both provide different(but complementary) information about the performance of the model in classifying if the edges exist. The ROC curve measures the ability to distinguish between positive and negative samples,  AP measures the ability to rank node pairs according to their probability of being linked**

To further deal with class imbalance, I have experimented with weighted and focal loss functions.
    - Can be improved with a dynamically learned weight through the decoder alone, that assigns higher weights to misclassified samples or focus on hard-to-classify samples.

### GPU Optimization

Now that everything is working smoothly(not blowing vram anymore), I have wrapped the code to use "cuda" as the torch device.
Accordingly, setting the flag USE_GPU ensures the following:
- torch.utils.data.DataLoader uses pin_memory = True
- num_workers = 4*num_gpus which in my case is 4
- the model is moved to the GPU
- in the train function, x, edge_index, edge_attr and y are moved to the GPU
- accumulated loss and output is deleted
- stored loss history is type converted from tensor to float
gpu usage seems to settle at a constant 733 mb per epoch

### TODO

- [x] Add real-time metrics during the train function so I can plot AUC and ROC score graphs.(done)
- [x] migrate encoder and decoder network to torch.nn.module

### SUMMARY:

The room for improvement and further studies are in the following directions:

#### Model Improvement:

- The VGAE can be experimented with passing the node features through normal autoencoders to obtain latent low-dimensional representations at different layers and propagate them through the model using skip connections. This should increase/preserve the node information better.

-  A custom message passing layer that performs some sort of aggregation of x and edge_attr

#### Data:

- Advanced sampling strategies can provide a more balanced representation of the data(negative sampling but in a disjoint way would be most optimal for the vgae)
 - There is room to test advanced decoders, trying the symmetric architecture definitely hurt the model in terms of the information it retained.
 - Other loss functions between the encoder and the decoder-generated distribution that can determine the similarity between the probability distributions(compute the loss between two nodes using something other than kl divergence)

#### Miscellaneous:

- The ensemble of experiments with the losses, aggregation, model architecture, and degree of encoder bottleneck can be migrated to proper automated tests using GraphGym

- Another intriguing approach was to run the link predition as a time series to achieve online edge classification keeping in mind the eventual goal of reconstructing a trajectory in real time.

- This goes hand in hand with the idea to improve upon the sampling strategies since online edge classification can be evaluated by replacing the sampled edges with generated negative edges for every node that is continuously added to the graph. This forms a neat testbench to test online link prediction, especially using the data model of particle hits through a detector.

This can be tested on two types of datasets:
- varying the particle density will increase the number of nodes
- varying the number of particles to provide more edges.
This should help the model generalize better.

### Guide:

All my experiments are stored with outputs in the notebooks folder. 
