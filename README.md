# Hyperparameter search for playing StarCraft II with DQN with non-spatial features
This is my source code of my research project in my masters degree of computer science and engineering. You might find it helpful. 

## Used components
* [**StarCraft II binary and mini-games maps** ](https://github.com/Blizzard/s2client-proto#downloads)
* [**PythonSC2 Framework**](https://github.com/Dentosal/python-sc2) for non-spatial features and atomic functions 
* [**Ray/RLlib**](https://github.com/ray-project/ray/tree/master/rllib) for parallelization

Our source code is a mostly a team project, find it in folder bot. In /bot/JAS/Agents/produce_marines.py you can find the agent code for the RL scenario. In /bot/JAS/game_commander.py you can see a on_step method that will be called every time step. In /bot/ray_server you can find the ray/rllib configs for DQN and Ape-X. My code for the plots can be found in /bot/scripts. 

## Scenario: BuildMarines
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")


## The neural network architecture
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

## The result of tuned DQN with human reference
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")
