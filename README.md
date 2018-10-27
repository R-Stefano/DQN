# Playing Atari with Deep Reinforcement Learning

## About
Replicating Google Deepmind's paper "Playing Atari with Deep Reinforcement Learning"
[Full article on the paper here](http://demiledge.com/artificialIntelligence/DQN.php)

## Dependencies
* numpy
* tensorflow
* matplotlib
* gym
## Getting started
The network architecture is in `DQN.py`. The class `replayMemory.py` stores and manages the transitions created during training. The file `main.py` is used to run the whole program callable using
`python3 main.py`

The network is saved in the folder **myModel** while the tensorboard's file in **results**

## Result

## Sources
* [arXiv](https://arxiv.org/abs/1312.5602) by Deepmind
* [Nature paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) by Deepmind
* [Full article about the paper](http://demiledge.com/artificialIntelligence/DQN.php)
