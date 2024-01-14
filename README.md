# two_tower_models
This is companion repository to [two tower models](https://recsysml.substack.com/p/two-tower-models-for-retrieval-of).

1. This shows a sample implementation of two tower models in PyTorch.
2. Then we go on to show how to implement the ranking module described in [Revisting neural accelerators](https://arxiv.org/abs/2306.04039).
3. Then we extend this to adding knowledge distillation from ranking models similar to the approach described in [How to reduce the cost of ranking by knowledge distillation](https://recsysml.substack.com/p/how-to-reduce-cost-of-ranking-by)
4. In this we add a further layer of funnel consistency where, inspired by [RLHF](https://arxiv.org/abs/1909.08593), we use the ranking model as a "reward model" and learn how to make the retrieval more aligned with the ranking model.
