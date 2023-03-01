# Banana Collector Unity Environment
In this project, I used a DQN model to train an agent to play the Unity food collector environment. 

This environment has 37 states with 4 actions:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

This environment is episodic, and to solve it, the agent must get an average score of +13 over 100 consecutive episodes.

## Summary

  - [Getting Started](#getting-started)
  - [Runing the scripts](#running-the-scripts)
  - [Author](#author)
  - [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Install Python
I have tested this repo with Python 3.9 and 3.10. To continue, install either of these versions on your local machine. With Python installed, I suggest you create a virtual environment to install required libraries:

```bash
python -m venv desired_path_for_env
```
Activate this environment before moving to next step. For addirional help, [check Python documentation here](https://docs.python.org/3/library/venv.html).

### Install PIP Packages

The required packages for this project are listed in [requirements file](requirements.txt). To install these libraries, from the repo folder, run the following command in your virtual env:

```bash
python -m pip install -r requirements.txt
```


### Download Unity Banana Collector
The already built Unity environment for this project is accessible from following links:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- MacOS: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Decompress (unzip) the downloaded file and copy it to the repo folder.

## Running the scripts

The training and testing scripts are located in [scripts](scripts) folder.

### Training

To train the model, use [train_agent.py](scripts/train_agent.py) script. This script accepts the following arguments:

- Path to downloaded Unity App: --unity-app
- Target Score to save trained model: --target-score

```bash
python train_agent.py --unity-app Banana.app --target-score 13
```

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

[Saved Trained Checkpoint](checkpoints/checkpoint_481.pth)

![Trained Model Scores](images/train_scores.png)

### Testing

To compare a trained agent with a untrained one, use [test_agent.py] script. This script accepts the following arguments: 

- Path to downloaded Unity App: --unity-app
- Path to saved model checkpoint: --checkpoint-file

```bash
python test_agent.py --unity-app Banana.app --checkpoint-file ../checkpoints/checkpoint_481.pth
```

## Author
  - **Sina Fathi-Kazerooni** - 
    [Website](https://sinafathi.com)


## License

This project is open source under MIT License and free to use. It is for educational purposes only and provided as is.

I have used parts of scripts in [Udacity DRL](https://github.com/udacity/deep-reinforcement-learning/) repo under [MIT License](https://github.com/udacity/deep-reinforcement-learning/blob/master/LICENSE). Scripts in [dqn](dqn) and [mlagents](mlagents) are based on [Udacity DRL](https://github.com/udacity/deep-reinforcement-learning/) repo with minor modifications.
