criticality of neural networks allows for maximizing the computational efficiency. It is characterized by a branching ration - how many neurons does any specific neuron activate, when it was active by itself. In modern neural nets it is forced through layer normalization - by normalizing outputs of neurons to be around 1 we make them behave as if a single neuron was active, without overblowing or decaying information. So besides numerical stability it is also the informational stability that we get.

losses
https://youtu.be/jNhnxr5yCWQ?si=4qrxRw3CgyAeiWWz
## Continual learning

Suppose we have an objective $L$ and the ability to sample it at any point. We can formulate continual learning as an indefinitely running process that samples the objective, adjusts its representation to account for the error and repeat.

One of the fundamental issues of this approach is catastrophic forgetting - as we observe new inputs, accuracy at earlier learned terms degrades. To avoid that we must regularly refresh old knowledge to preserve accuracy, while also exploring other areas where we didnt have enough data to be able to represent the domain. ideally we want to minimize expected error on future observations, while also minimizing error on already observed points.

bayesian networks
https://youtu.be/8UE0IdCyg6Y?si=E8zxz0RXNQGqmS9S

[training invariance](https://chatgpt.com/c/6af218d0-1558-83eb-952c-a30628e6a9c4)

Can Scale Save Us From Plasticity Loss in Large Language Models? [https://arxiv.org/abs/2606.24752](https://arxiv.org/abs/2606.24752) 
In-Place Test-Time Training [https://arxiv.org/abs/2604.06169](https://arxiv.org/abs/2604.06169) 
Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference [https://arxiv.org/abs/2605.26099](https://arxiv.org/abs/2605.26099) 
Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories [https://arxiv.org/abs/2606.03979](https://arxiv.org/abs/2606.03979)