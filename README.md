# two_tower_models

This is companion repository to [two tower models](https://recsysml.substack.com/p/two-tower-models-for-retrieval-of), a commonly used approach to retrieval / candidate generation in recommender systems. The goal of this repository is to show how to increase the alignment of retrieval with ranking.

## [two_tower_base_retrieval.py](./src/two_tower_base_retrieval.py)

This shows a sample implementation of two tower models in PyTorch.

## [two_tower_plus_light_ranker.py](./src/two_tower_plus_light_ranker.py)

Then we go on to show how to implement the ranking module described in [Revisting neural accelerators](https://arxiv.org/abs/2306.04039). We chose this since implementing a light ranking in the retrieval system is an intuitive way to increase consistency with main ranking.

## [two_tower_plus_light_ranker_plus_main_ranker_kd.py](./src/two_tower_plus_light_ranker_plus_main_ranker_kd.py)

Then we extend this to adding knowledge distillation from ranking models. This is similar to the approach described in [How to reduce the cost of ranking by knowledge distillation](https://recsysml.substack.com/p/how-to-reduce-cost-of-ranking-by)

## [two_tower_base_plus_main_ranker_reward_model.py](./src/two_tower_base_plus_main_ranker_reward_model.py)

In this step we add a further layer of funnel consistency where, inspired by [RLHF](https://arxiv.org/abs/1909.08593), we use the ranking model as a "reward model" and learn how to make the retrieval more aligned with the ranking model.

------

_Disclaimer:_ These are the personal creations/opinions of the author(s). Any artifacts, opinions stated here are theirs and not representative of their current or prior employer(s). Apart from publicly available information, any other information here is not claimed to refer to any company including ones the author(s) may have worked in or been associated with.
