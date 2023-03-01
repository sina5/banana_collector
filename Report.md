# Solve Banana Collector Unity Environment with DQN

## DQN Model Details

In this project, I used a two-layer fully-connected network for DQN model.

Model details:

- Layer one
    - Units: 64
    - Activation: ReLU
- Layer two
    - Units: 64
    - Activation: ReLU

- Hyperparameters:
    - Replay buffer size: 1e5
    - Minibatch size: 64
    - Discount factor: 0.99
    - Interpolation parameter: 1e-3
    - Learning rate: 5e-4
    - DQN network update interval: 4


## Results

On my machine, the environment was solved in 481 episodes:

```
Episode 100     Average Score: 1.065
Episode 200     Average Score: 4.18
Episode 300     Average Score: 7.93
Episode 400     Average Score: 11.20
Episode 481     Average Score: 13.07
Environment solved in 481 episodes!     Average Score: 13.07
Trained model weights saved to: checkpoint_481.pth
```

![Trained Model Scores](images/train_scores.png)

## Future Work

In future, I plan to update the repo to add Double DQN, Dueling DQN, and prioritized experience replay. 

## Author
  - **Sina Fathi-Kazerooni** - 
    [Website](https://sinafathi.com)
