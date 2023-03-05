# DQN-pytorch
Parallel DQN for an Inverted Pendulum with Image Data implemented in PyTorch and OpenAI Gym

The codes for DQN are adapted from [Kchu](<https://github.com/Kchu/DeepRL_PyTorch>) 

The codes for the classical control game Pendulum are excerpted from OpenAI baseline Gym, and an new image data output API is added to this game by defining a new observation wrapper in wrappers.py.

The default number of parallel threads is set to 1 because the percentage of the sampling period in all traning time is very small. In other words, the sampling costs very little time, and the optimization space is very narrow through employing parallel coding technique.

Always up for a chat -- shoot me an email (wgj@buaa.edu.cn) if you'd like to discuss anything.

## Installing Dependency: 

```
conda env create -f environment.yml
```

## How to use

```
git clone https://github.com/buaawgj/DQN-pytorch.git
cd DQN-pytorch
python train.py
```

When you run these codes, it can automatically create two subdirectories under the current directory: ./data/model/ & ./data/plots/. These two directories are used to store the models and the results.

After training, you can plot the results by running result_show.py with appropriate parameters.

## References:

1. Human-level control through deep reinforcement learning (DQN)   [[Paper](https://www.nature.com/articles/nature14236)]   [[Code](https://github.com/buaawgj/DQN-pytorch/dqn.py)]
