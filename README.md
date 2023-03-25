My solutions and submission to the qualification task of the GSoC 2023 “GNN Tracking”project, under CERN, IRIS-HEP

My initial approach to classifying whether the edges exist or not is to construct a basic model that uses a batch of graphs.
I will start all the utils required to set up a framework to develop the architecture further.

- Data model
 To make using the data simple and abstracted I have expanded the provided dataset import class to do the following functions:
 check load_data.py I have defined the same class that uses just the index of the batch that I currently want to load. Decide dataset split. *tr,val,test(85 : 5 : 10)?*

    - *TODO:* move the display statistics and visualise functions into load_data.py[low priority]

    - *TODO:* get model summary working, for some reason the module is missing in my conda env [DONE]


### handling class imbalance :

**Sampling strats**

- So as expected the negative samples (i.e., pairs of nodes that are not connected) are greater than the positive samples (i.e., pairs of nodes that are connected).

    *How will you address this?*

    - stratified sampling can be used to ensure that the training set contains an equal number of positive and negative samples.

    - the preliminary idea is to just sample the same number of negative samples for each set.

    - Ok, looks like that helps but now I have less data. I think it would make sense to distribute the same number of negative samples for each set.

    - If time allows, try randomly selecting a subset of the negative samples for training, instead of using all of them. 
    (instead of trying to balance the class distribution, reduce the degree of imbalance? :) )

    - (MAYBE) If I do the above step, one concern I have is that the negative edges will overlap with the positive edges and negative edges in other sets. Try discarding the sample from the training set to emulate some sort of disjoint sampling algo(reminder: figure out math first plz)


    <insert code segment displaying the imbalance>

    Result:
    ```bash
    **Percentage of positive edges:  17.673395558091762**
    **Percentage of negative edges:  82.32660444190823**
    ```

    - Ensemble methods: don't have enough time so this goes under future prospective improvement directions. multiple models with different negative sampling strategies and combine their predictions to obtain a more balanced and accurate model.

    - modifying loss function can also deal with class imbalance
#### TODO:
- Experiment with different approaches and evaluate their performance to determine the most effective method for your specific use case.(Maybe using GraphGym?)

Update: 
- went through basic architecture and standard layer functions. Implement one basic model using a few graphconvs and one Linear for binary classification
- Obviousy not the best approach. 
- gg found a parallel to VAE's, personally the most interesting approach.
- a lot of it is similar including using KL divergence (obviously they are both probability distributions at the end of the day)

- Encoder should learn the node embeddings. 
 
*TODO today:*
 
- try making this architecture to learn node embeddings by aggregating and transforming neighboring node features in the graph.
- these embeddings are representative of the graph as a whole and should improve the performance of edge classification.
 
 How are node embeddings learned?

Try this as a basic architecture(link the research paper later)

- Linear Transformation: first apply a linear transformation to the input node features using a weight matrix. 
Why? the learned weight matrix is basically a unique representation for each node.

- Message Passing: perform message passing aggregated from the neighboring nodes of each node in the graph. 
   
    - Rough algo:
        - taking a weighted_sum(features of neighboring nodes)   *(where the weights are learned using another weight matrix)*. 
        - original node features += weighted_sum, *this updates the node representation to include info from the neighboring nodes.*

- Non-linearity: pass new node features through non-linear activation function(ReLU?), which introduces non-linearity to the layer and allows it to learn complex representations.

- stack GCN layers, end with ReLu, should learn complex and informative representations of the graph. These node embeddings capture both local and global information from the graph 

learned using features of its neighbors in the graph instead of just the nodes. This should allow the GCN layer to capture structural dependencies between nodes.

*TODO for evening*  try and make the GAE first. (done)

- Now that the embeddings part is done, use the similarity score of two node embeddings to decide whether they should be connected.

**Experiment if time allows**
types of decoders, dot product?
okay so extract the embeddings from the latent representation using:

an MLP with two linear layers and a ReLU activation function between them. That takes the latent representation of each node as input and generates a vector of the same dimension as the input features 


so all in all:
the vgae forward -> encoder->latent space -> decoder->embeddings-> dot product->sigmoid-> which gives probabilty

**Update day 2**
the learned latent representation makes predictions about missing edges.
the research paper describes the architecture using a dense adjacency matrix, but on experimenting with both, I think the sparse edge representation is better.
1. edge_index contains only essential info about the graph, which makes it easier to learn meaningful patterns. the dense adjacency matrix contains a lot of redundant information(which may lead to overfitting too?)

2. computation-wise, not even a doubt, considering only the edges that are present in the graph, instead of all possible pairs of nodes. with a class imbalance in this dataset, we're looking at over 80% of node pairs that don't have edges.

**ERROR** So GCNConv only supports one-dimensional edge attributes so will have to find a replacement layer
**Approaches**:
- If time had allowed, I would try implementing a custom message passing layer that performs some sort of aggregation of x and edge_attr
- Since I am on a time crunch I have decided to solve this by going ahead with incorporating the edge-attributes into GCN by concatenation with x.
- I will experiment with the same model with and without to help me decide if edge attributes are noisy or actually informative.

**Improvements** :
changed architecture from 
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

- going ahead with area under the ROC and average precision (AP) scores

ROC curve evaluates the link prediction by checking TPR vs FPR at different link probs(*low priority: add a confusion matrix*). The area under the ROC measures performance in distinguishing between positive and negative samples, independent of the link probability threshold. 

Average precision (AP) is a really good metric here since the positive class is rare. AP measures the area under the precision-recall curve, which plots the precision against the recall at different thresholds of predicted link probabilities. so it evaluates the model's ability to rank pairs of nodes based on their likelihood of being linked. 

**so they both provide different(but complementary) information about the performance of the model in classifying if the edges exist. The ROC curve measures the ability to distinguish between positive and negative samples,  AP measures the ability to rank node pairs according to their probability of being linked**

if time allows:
- To deal with **class imbalance** can we try weighted or ```focal loss functions```?
    - Look into loss functions that assign higher weights to misclassified samples or focus on hard-to-classify samples.

**update**
*started writing basic focal loss function. not enough time to retrain
*TODO*: add to basic autoencoder and compare the performance

### GPU Optimization

Now that everything is working smoothly(not blowing vram anymore gg/why is my laptop more choppy while exporting a jupyter notebook as a html file ;) ), I have wrapped the code to use "cuda" as the torch device.
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

### software engineering task
tried function
it works correctly with >> symmetrize_edge_weights(T([[1, 2], [2, 1]]), T([1, 3]))
T([2, 2])
**update*8
indexing was wrong,

while giving input 
```python
>>symmetrize_edge_weights(T([[1, 2], [3, 4], [2, 1]]), T([1, 2, 3]))
```
```RuntimeError: The size of tensor a (3) must match the size of tensor b (2) at non-singleton ```
fix this.is the type a problem?

fix:
use long
```python
symmetric_edge_weights = T([symmetric_weights[(a1, a2)] for a1, a2 in edge_indices.long().()])
```
fixed(nvm was not indexing for both cases). add tests if possible



### SUMMARY:

Apologies if this documentation is too informal, it was used for me to personally track what I was doing over the last 48 hours. Since I haven't written down so many other small  improvements like using validation or hyperparam tuning it is definitely not as long or detailed as it should be but, here's a quick summary and a few ideas I would have liked to pursue.

The room for improvement and further studies I would like to do are in the following directions:
immediate improvement on model:
- on the variational graph autoencoder I would like to experiment with passing the node features through normal autoencoders to obtain latent low-
dimensional representations at different layers and propagate them through the model using skip connections. This should increase/preserve the node information better.

- implementing a custom message passing layer that performs some sort of aggregation of x and edge_attr

Data:
- Implementing some of the sampling strategies as mentioned above that provide a more balanced representation of the data(negative sampling but in a disjoint way would be most optimal for the vgae)
 - The decoder itself I have not experimented with too much. I have tried using a symmetric decoder but it definitely hurt the model in terms of the information it retained. 
 - Try other functions that can determine the similarity between two probability distributions(compute the loss between two nodes using something other than kl divergence?)

A lot of hurried experiments with the losses, aggregation, model architecture, degree of encoder bottleneck, and others can be migrated to proper automated tests using GraphGym

Definitely, the most intriguing approach I would like to pursue was the online edge classification keeping in mind the eventual goal of reconstructing a trajectory.
This goes hand in hand with the idea to improve upon the sampling strategies since I would like to test online edge classification by replacing the sampled edges with generated negative edges for every node that is continuously added to the graph. This forms a neat testbench to test online link prediction, especially using the data model of particle hits through a detector.

This can be tested on two types of datasets:
- varying the particle density will increase the number of nodes
- varying the number of particles to provide more edges.
This should help the model generalize better.

### Guide:

All my experiments are stored with outputs in the notebooks folder. 

 The models folder conatins the vgae since it performed the best, although I was not able to convert it into a model plugin for the [gnn_tracking repository](https://github.com/gnn-tracking/gnn_tracking)

Finally the links:

https://www.researchgate.net/figure/a-Architecture-of-the-variational-graph-auto-encoder-VGAE-the-first-part-of_fig2_343526318
 
https://arxiv.org/pdf/1611.07308.pdf
 
https://arxiv.org/pdf/1902.07987.pdf
 
https://arxiv.org/pdf/2103.11414.pdf
 
https://arxiv.org/abs/1901.01343
 
https://graph-neural-networks.github.io/static/file/chapter10.pdf
 
https://indico.cern.ch/event/955026/contributions/4012924/attachments/2127589/3582355/IRIS-Topical_GNN_Tracking.pdf
