https://arxiv.org/abs/1611.03530?utm_source=chatgpt.com
https://proceedings.mlr.press/v97/du19c.html
https://arxiv.org/html/2101.02429
https://en.wikipedia.org/wiki/Generalization_(learning)

Generalization is the ability to adapt existing knowledge to a new setting.

https://en.wikipedia.org/wiki/Anti-unification

Generalize through algorithmic stability - we want our model to not depend on a particular set of data. We can optimize for that by minimizing discrepancies between the models trained independently on different subsets of the data. Note that there is a limit to this approach, as dataset may be too small to reliably derive actual relation accurately, so we must enforce a minimum size of the subsets we compare to be sufficient for the identifiability of the solution. 

After training on a specific dataset, a specific solution the model settles on encodes some assumption of how the dataset should be extrapolated into unseen dataset. Moreover there might be some solutions that extrapolate well in one region, but badly in another. That is often possible to solve through ensemble of models such as mixture of experts, gradient boosting

A source of poor generalization may be a latent state, unobservable through the dataset, but required to determine the true relation between the points.

Ensemble learning
https://youtu.be/sQ3JCxcQ-Gs?si=gOPbap7fWX0v3bfE

and Mixture of experts

[generalizetion definition and statistical approach](https://chatgpt.com/c/6af1dd52-5f68-83ed-b8c4-842e4537ea17)