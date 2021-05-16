[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Getting Started
1. Follow instruction here [click here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to set up your Python environment with unityagents.  

2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
   

3. Clone this GitHub repository, and add the environment files in the data folder.

### Instructions  

Training
```
$ python training.py
```

### Results
**(Training)** Mean scores every 100 episodes ([traning.py](./traning.py)) 
```
Episode 100	Average Score: 0.03690	Max Score: 0.30000 in 100 episodes
Episode 200	Average Score: 0.37900	Max Score: 2.70000 in 100 episodes
Episode 206	Average Score: 0.51610	Max Score: 2.70000 in 100 episodes
Environment solved in 206 episodes!	Average Score: 0.51610
```
![train](./tennis_scores.png)


**(Play games)** Mean scores every episode ([play.py](./play.py))   
Note: set training_mode = True to speed-up a game play and visualise results
```
Episode 1	 Average Score: 2.60
Episode 2	 Average Score: 1.35
Episode 3	 Average Score: 1.77
Episode 4	 Average Score: 1.35
Episode 5	 Average Score: 1.60
Episode 6	 Average Score: 1.35
Episode 7	 Average Score: 1.53
Episode 8	 Average Score: 1.66
Episode 9	 Average Score: 1.78
Episode 10	 Average Score: 1.61
Episode 11	 Average Score: 1.70
Episode 12	 Average Score: 1.78
Episode 13	 Average Score: 1.85
Episode 14	 Average Score: 1.90
Episode 15	 Average Score: 1.95
Episode 16	 Average Score: 1.99
Episode 17	 Average Score: 2.04
Episode 18	 Average Score: 2.07
Episode 19	 Average Score: 2.10
Episode 20	 Average Score: 2.13
Episode 21	 Average Score: 2.15
Episode 22	 Average Score: 2.17
Episode 23	 Average Score: 2.19
Episode 24	 Average Score: 2.21
Episode 25	 Average Score: 2.23
Episode 26	 Average Score: 2.25
Episode 27	 Average Score: 2.26
Episode 28	 Average Score: 2.28
Episode 29	 Average Score: 2.29
Episode 30	 Average Score: 2.30
```
[Video](./tennis.md) 

### DDPG Implementation Details

Further information regarding to the implementation can be found in [Report.md](./Report.md)   



