## Project Description 

The environment in this repository consist of single 2-link robot arm and a target sphere.
The agent controlling the robot receives a reward of 0.1 for every time step that the end effector is withn the boundary of the target sphere.

The robot has a 33 dimensional state space consisting of position, velocity, angular rotation and velocity of all the joints. The action space is instead 4 dimensional space representing all the torques applicable on the joints.

The environment is considered solved with a score of 30 averaged over 100 consecutive episodes. 

## Getting Started

The environment dependencies can be found in the [Udacity Deep Reinforcement Learning Github](https://github.com/udacity/deep-reinforcement-learning#dependencies).
By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to complete the project.

In order to run the environment a pre built simulation has to be installed according to the specific OS.
The link are reported below:
- Linux: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OX: [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)


## Instructions

After installing the local simulator fill in the path to *Reacher.exe* file on the first line of the *main.py*
The agent can then be trained running the *main.py* file in the repo.

The set of pre-train weights of the agent can be found in the *TrainedAgent* folder. 
A simulation of the agent running the parameters can be checked running *test_agent.py* file. Also in this case, make sure the directory of the executable file is defined in the first line of the file.

