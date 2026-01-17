# Node Label Prediction using Relational Classifiers on the OGBN-Products Graph

1. Objective of the Project

The objective of this project is to perform node label prediction on a large real-world graph using a message-passing Graph Neural Network (GNN). The task involves learning node representations that incorporate both node features and graph structure, and then using these representations to classify each node into one of several product categories. The project uses the OGBN-Products dataset, which represents an Amazon co-purchase network, and applies a Graph Convolutional Network (GCN) implemented using PyTorch Geometric.

2. Dataset Description and Graph Characteristics

The OGBN-Products dataset consists of a single, very large graph with millions of nodes and edges. Each node represents a product, and edges indicate co-purchase relationships between products. Nodes are associated with feature vectors and a single class label corresponding to a product category. Due to the size of the dataset, direct full-graph training and visualization are not feasible within limited computational resources, which motivates the use of subgraph sampling throughout the project.

3. Data Loading and Preprocessing

The dataset is loaded using the OGB Node Property Prediction interface provided by PyTorch Geometric. The graph is converted to an undirected version to allow symmetric message passing between connected nodes. Standardized train, validation, and test splits provided by OGB are used to ensure fair evaluation. These splits define which nodes contribute to training loss, validation tuning, and final performance reporting.

4. Graph Convolutional Network Architecture

The model used for node classification is a three-layer Graph Convolutional Network. Each GCN layer performs neighborhood aggregation followed by layer normalization and ReLU activation. Dropout is applied after each layer to reduce overfitting. The final node embeddings are passed through a linear layer to produce class logits. This architecture allows information to propagate up to three hops in the graph, enabling each node to incorporate context from its local neighborhood.

5. Model Design Considerations

Layer normalization is used instead of batch normalization to handle variable neighborhood sizes more robustly. A relatively large hidden dimension is chosen to capture the rich structural and feature information present in the graph. Dropout is used consistently across layers to stabilize training. The model also exposes an encoding function that allows extraction of intermediate node embeddings for further analysis.

6. Subgraph Construction for Scalable Training

Because the full OGBN-Products graph is too large to train on directly, a random node-induced subgraph is created using 50,000 nodes. A node-induced subgraph preserves all edges that connect selected nodes while discarding edges that involve nodes outside the subset. This approach retains realistic local connectivity patterns while reducing memory and computation requirements to a manageable level.

7. Mapping Dataset Splits to the Subgraph

The original train, validation, and test node indices are mapped to the subgraph index space. Nodes that do not appear in the sampled subgraph are discarded from the split sets. This ensures that training and evaluation are performed only on nodes present in the subgraph, while still respecting the original dataset splits.

8. Training and Evaluation Strategy

The model is trained using cross-entropy loss computed on the training nodes of the subgraph. Optimization is performed using the Adam optimizer with weight decay to regularize the model. Performance is evaluated using the official OGB evaluator, which computes classification accuracy. Metrics are reported separately for training, validation, and test nodes to monitor generalization behavior during training.

9. Training Behavior and Results

Over multiple training epochs, the model shows steady improvement in training accuracy, followed by stabilization of validation and test accuracy. This behavior indicates that the GCN is effectively learning meaningful node representations without severe overfitting. The final evaluation on the subgraph demonstrates reasonable predictive performance given the reduced training size and limited neighborhood context.

10. Graph Statistics and Structural Analysis

To better understand the sampled graph, several structural properties are computed. These include the number of nodes and edges, the global clustering coefficient, and the approximate diameter of the largest connected component. The clustering coefficient provides insight into the local density of connections, while the diameter reflects the graph’s overall connectivity and reachability. These statistics help contextualize the learning task and explain why multi-hop message passing is necessary.

11. Visualization of Sampled Subgraphs

For visualization purposes, a much smaller node-induced subgraph with 2,000 nodes is created. This subgraph is visualized using a spring layout, with nodes colored according to their class labels. The visualization highlights the relational nature of the problem, as nodes with similar labels often appear close to each other, suggesting that label information is correlated with graph structure.

12. Node-Induced Subgraph Concept

A node-induced subgraph is formed by selecting a subset of nodes and retaining all edges whose endpoints lie entirely within that subset. This method preserves the original graph’s connectivity patterns among selected nodes without introducing artificial edges. In this project, node-induced subgraphs are essential for both scalable training and meaningful visualization, as they maintain the relational properties of the original dataset.

13. Two-Hop Node Embedding Extraction

To analyze the learned representations, explicit two-hop node embeddings are generated by forwarding node features through the first two GCN layers of the trained model. These embeddings encode information from a node’s immediate neighbors as well as neighbors of neighbors. Examining these embeddings helps illustrate how relational information is aggregated and how structural context influences node representations.

14. Diameter Computation on Sampled Subgraphs

The diameter of the largest connected component of a sampled subgraph is computed to understand the graph’s effective depth. A relatively small diameter suggests that information can propagate across the graph in only a few message-passing steps, supporting the use of a shallow GCN architecture. This structural insight aligns well with the design choice of using three GCN layers.

15. Observations and Practical Insights

The experiments confirm that relational classifiers such as GCNs are well-suited for node classification tasks where label information is correlated with graph structure. Subgraph sampling proves to be an effective strategy for scaling training and analysis without losing essential structural properties. Visualization and embedding analysis further reinforce the intuition that message passing enables nodes to learn from their neighbors in a meaningful way.

16. Limitations of the Approach

While subgraph training makes the problem tractable, it limits the receptive field of each node compared to full-graph training. The sampled subgraph may omit important long-range connections present in the original graph. Additionally, the model uses a relatively simple GCN architecture and does not incorporate more advanced techniques such as attention, sampling-based mini-batching, or label propagation.

17. Possible Extensions and Future Work

Future work could explore training on larger subgraphs or using scalable mini-batch methods such as GraphSAGE or Cluster-GCN. More expressive architectures, such as Graph Attention Networks, could also be evaluated. Incorporating node embedding visualization techniques or downstream tasks such as link prediction would further deepen the analysis.

18. Conclusion

This project demonstrates a complete pipeline for node label prediction using relational learning on large graphs. By combining Graph Convolutional Networks with node-induced subgraph sampling, it becomes possible to train, evaluate, and analyze models on otherwise intractable datasets. The results highlight the importance of graph structure in node classification and show how message-passing models effectively leverage relational information in real-world networks.
