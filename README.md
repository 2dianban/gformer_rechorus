# Graph Transformer for Recommendation
This is the PyTorch implementation for GFormer model proposed in this paper:

 
## Introduction
This paper presents a novel approach to representation learning in recommender systems by integrating generative self-supervised learning with graph transformer architecture. We highlight the importance of high-quality data augmentation with relevant self-supervised pretext tasks for improving performance. Towards this end, we propose a new approach that automates the self-supervision augmentation process through a rationale-aware generative SSL that distills informative user-item interaction patterns. The proposed recommender with Graph Transformer (GFormer) that offers parameterized collaborative rationale discovery for selective augmentation while preserving global-aware user-item relationships. In GFormer, we allow the rationale-aware SSL to inspire graph collaborative filtering with task-adaptive invariant rationalization in graph transformer. The experimental results reveal that our GFormer has the capability to consistently improve the performance over baselines on different datasets. Several in-depth experiments further investigate the invariant rationale-aware augmentation from various aspects.

<img src='fig\framework.jpg'>

## Environment
The codes of GFormer are implemented and tested under the following development environment:
  
<p>PyTorch:

</p>
<ul>
<li>python=3.8.13</li>
<li>torch=1.9.1</li>
<li>numpy=1.19.2</li>
<li>scipy=1.9.0</li>
<li>networkx = 2.8.6</li>
</ul>
  
## Datasets
| Dataset |
|:-------:|
|Grocery_and_Gourmet_Food|
|MovieLens_1M |



## How to Run the Code

<ul>
<li><code>python gmain.py --data dataset --reg 1e-4 --ssl_reg 1 --gcn 3 --ctra 1e-3 --b2 1 --pnn 1<code></li></ul></body></html>
