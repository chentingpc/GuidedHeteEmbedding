Introduction
-----------------
This is a CPU-based multi-thread implementation of Task-guided and Path-augmented Heterogeneous Network Embedding.

Data format for network embedding
-----------------
**network file**

each line can be either (homogeneous network):
```
src_node dst_node weight
```
or (heterogeneous network)
```
src_node dst_node weight edge_type
```

_Note_:

1. separator allowed: tab or space (don't mix them in names).
2. each edge_type can only be associated at most two types of nodes.

**node_type file (optional - required for heterogeneous network)**

each line should be
```
node node_type
```

_Note_:

1. separator allowed: tab or space (don't mix them in names).
2. nodes of different types should have different names.
3. nodes should also include features used in supervised task.

**path_conf_file (optional)**

this file specify the configuration for heterogeneous network embedding.

each line includes:

```
edge_type edge_type_weight direction_conf proximity_conf sampling_ratio_power base_degree
```

Where

```edge_type_weight```: float in [0, inf). edge_type is weighted by random sampling, less weight means less chances being sampled. when tuning, let all edge type weights sum to some constant.

```direction_conf```: normal/reverse/bidirection. normal means only consider given direction (negative is drawn in destination); reverse means only consider reverse direction; bidirection means half-half for both given and reverse direction (negative is drawn in both source and destination).

```proximity_conf```: single/context. single is use one vector for each node; context is to use additional context vector for target node.

```sampling_ratio_power```: float in (0, 1], used to scale degree of a node under the path for negative node sampling.

```base_degree```: added to every node (of targeted type) when computing noise node sampler, note the number is in multiple of targeted min non-zero degree.

_Note_:

1. if this file is given, edge types not present in the file will not be used in network embedding.
2. if this file is not given, default option is: all edge types have weight sum of 100000, bidirection, single embedding vector, sampling ratio 0.75 with 1 base degree.
3. separator allowed: space.

Data format for supervised ranking task (optional)
-----------------
if you have no supervised task and are only interested at the network embedding, simply provide NO train/test files, the supervised model will be skipped.

**train_target.txt & test_target.txt**

for each line,

```
id target label
```

in the future, it will support (similar to libsvm format)

```
id target_1:label target_2:label ...
```

_Note_:

1. target should be the node in network embedding, id does not need to be. (will change in the future)
2. if only positive labels are given, negative samples are generated randomly.
3. if negative labels are also given, negative samples are generated proportional to the magnitude of negative scores. (currently not supported)

**train_feature.txt & test_feature.txt**

```
id feature weight
```

in the future, it will support

```
id feature_1:weight feature_2:weight ...
```

_Note_:

1. feature should be node in network embedding, id does not need to be. (will change in the future)

Key parameters
---------------

**omega**: network embedding task sampling rate, when given; if not, tasks are trained independently as per number of threads.

**lr_emb, reg_emb**: learning rate and regularization for embedding (set same for both network embedding and supervised task).

**net_loss**: network embedding loss function, 0 for NCE/skip-gram objective, 1 for max-margin ranking objective. default 0.

**supf_loss**: supervised task loss function, 0 for max-margin ranking objective, 1 for Bayesian Personalized ranking objective, 2 for NCE/skip-gram loss with 1 negative sample. default 0.

**path_conf_file**: see above, a file of configurations for network embedding.

**supf_negative_by_sample, supf_neg_sampling_pow, supf_neg_base_deg**: first term specifies if to use negative sampling or directly use the negative candidate set (if prepared). When the first term is 0, will use prepared negative samples, and the last two terms don't matter. When the first term is 1, the latter two terms are similar to sampling_ratio_power and base_degree in network embedding. default: 1, 0, 1 (uniform dist over all authors).

**path_normalization, row_reweighting (optional)**: path_normalization set true when you want to keep all network of same weight sum (except specified by path_conf_file); row_reweighting can be used to smooth the neighbor distribution. default: 1, 0 (each path adjacency matrix will be normalized to a constant, and edge sampling is proportional to edge weight in each adjacency matrix).

**supf_dropout**: randomly dropout rate for features. default 0.

**supf_ignore_feat_weight**: if to ignore the weights in input features. default 0.

**Detailed learning rates and regularizations (unnecessary for most cases)**:

network embedding: *lr_net_emb, lr_net_w, lr_net_etype_bias, reg_net_emb*

supervised task: *lr_supf_emb, lr_supf_ntype_w, lr_supf_nbias, reg_supf_emb*

By default with setting lr_emb and reg_emb:
1. lr_emb == lr_net_emb == lr_supf_emb, reg_emb == reg_net_emb == reg_supf_emb.
2. lr_net_etype_bias == 1/100ish lr_net_emb, lr_supf_ntype_w \approx lr_supf_nbias = 1/100ish lr_supf_emb.
3. lr_net_w = 0.
4. reg_net_emb and reg_supf_emb start from 0.

Others
----------------------
1. some predefined parameters, mainly related to hash table sizes, can be modified in common.h if needed (e.g. to reduce memory usage for small data).
2. about load and save, only main emb can be load/saved, biases and node type weights will not.

Requirements
-----------------
The code can be run under Linux using Makefile for compilation.

Also, the GSL package is used and can be downloaded at
http://www.gnu.org/software/gsl/
