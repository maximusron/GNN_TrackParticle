My initial approach to classifying whether the edges exist or not is to construct a basic model that uses a batch of the graphs.
I will start all the utils required to setup a framework to develop the architecture further.

- Data model
 To make using the data simple and abstracted I have expanded the provided dataset import class to do the following functions:
 check load_data.py I have defined the same class that uses just the index of the batch that I currently want to load. Decide dataset split. *tr,val,test(85 : 5 : 10)?*

    - *TODO:* move the display statistics and visualise functions into load_data.py[low priority]

    - *TODO:* get model summary working, for some reason the module is missing in my conda env [DONE]


### handling class imbalance :

**Sampling strats**

- So as expected the negative samples (i.e., pairs of nodes that are not connected) are greater the positive samples (i.e., pairs of nodes that are connected).

    *How will you address this?*

    - stratified sampling can be used to ensure that the training set contains an equal number of positive and negative samples.

    - preliminary idea is to just sample the same number of negative samples for each set.

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

    - Ensemble methods: don't have enough time sothis goes under future prospective improvement directions. multiple models with different negative sampling strategies and combine their predictions to obtain a more balanced and accurate model.

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
 
 How are node embeddings learnt?

Try this as a basic architecture(link the research paper later)

- Linear Transformation: first apply a linear transformation to the input node features using a weight matrix. 
Why? the learned weight matrix is basically a unique representation for each node.

- Message Passing: perform message passing aggregated from the neighboring nodes of each node in the graph. 
   
    - Rough algo:
        - taking a weighted_sum(features of neighboring nodes)   *(where the weights are learned using another weight matrix)*. 
        - original node features += weighted_sum, *this updates the node representation to include info from the neighboring nodes.*

- Non-linearity: pass new node features through non-linear activation function(ReLU?), which introduces non-linearity to the layer and allows it to learn complex representations.

- stack GCN layers, end with ReLu, should learn complex and informative representations of the graph. These node embeddings capture both local and global information from the graph 

learnt using features of its neighbors in the graph instead of just the nodes. This should allow the GCN layer to capture structural dependencies between nodes.

*TODO for evening*  try and make the GAE first.

- Now that the embeddings part is done, use similarity score of two node embeddings to decide whether they should be connected.

### LOSSES

- going ahead with area under the ROC curve (AUC) and average precision (AP) scores

ROC curve evaluates the link prediction by checking TPR vs FPR at different link probs(*low priority: add a confusion matrix*). The area under the ROC measures performance in distinguishing between positive and negative samples, independent of the link probabilty threshold. 

Average precision (AP) is a really good metric here since the positive class is rare. AP measures the area under the precision-recall curve, which plots the precision against the recall at different thresholds of predicted link probabilities. so it evaluates the models ability to rank pairs of nodes based on their likelihood of being linked. 

**so they both provide different(but complementary) information about the performance of the model in classifying if the edges exist. The ROC curve measures the ability to distinguish between positive and negative samples,  AP measures the ability of to rank node pairs according to their probability of being linked**

if time allows:
- To deal with **class imbalance** can we try weighted or focal loss functions?
    - Look into loss functions that assign higher weights to misclassified samples or focus on hard-to-classify samples.

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