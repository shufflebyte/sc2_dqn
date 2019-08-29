# Hyperparameter search for playing StarCraft II with DQN with non-spatial features
This is the source code of my research project (worth 10 ECTS) in my masters degree of computer science and engineering. It is my first scientific elaboration, after having done only engineering projects before. You might find it helpful. I do not recommend to use this code (write it at your own, here is a lot of quick and dirty in it). But you might find my report helpful.

## Topic

My report deals with the DQN algorithm family and SC2. I started with a hyperparameter search for DQN and applied it to the scenario "BuildMarines". I investigated values for training_batch_size, learn_rate and the discount_factor. In order to spare calculation power and time, we decided to not use spatial features and instead use non-spatial features (python-sc2). In order to parallelize the whole learning, we used Ray/RLlib. 

After finding the best values for DQN (in the investigated interval) I applied it to DDQN (Double DQN) and Ape-X DQN (which is a distribution framework in which DDQN is used). You can see my results at later in this document. 

## Help and Theory 

Find my final [presentation](https://github.com/shufflebyte/sc2_dqn/doc/sc2_dqn_presentation.pdf) here and my [report](https://github.com/shufflebyte/sc2_dqn/doc/sc2_dqn_report.pdf) here. References to the papers I read and cited can be found in my report. Start your study with "Playing Atari with Deep Reinforcement Learning" by Mnih et al. and "Reinforcement Learning: An Introduction, Second edition in progress" by Sutton and Barto (in the newest draft there is an own chapter about the Atari experiments and the DQN algorithm). If you are totally new to Reinforcement Learning, start with Youtube and articles :-). Especially David Silver gave a very good [course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) which helped me a lot. 

## Used components
* [**StarCraft II binary and mini-games maps** ](https://github.com/Blizzard/s2client-proto#downloads)
* [**PythonSC2 Framework**](https://github.com/Dentosal/python-sc2) for non-spatial features and atomic functions 
* [**Ray/RLlib**](https://github.com/ray-project/ray/tree/master/rllib) for parallelization

| Software            | Version                                                   |
| ------------------- | --------------------------------------------------------- |
| anaconda            | 4.6.11 (this is not important, you can use what you want) |
| python              | 3.6.8                                                     |
| python-sc2          | 0.11.1                                                    |
| ray/rllib           | 0.7.0.dev2 (worked also with 0.7.0)                       |
| StarCraft II binary | B70326 (SC2.2018Season4)                                  |
| s2clientprotocol    | 4.9.1.74456.0                                             |
| tensorflow          | 1.12.0                                                    |

**Be prepared**: If you use python-sc2 with Windows or MacOS, you are using the Battle.Net Version of SC2. Since python-sc2 is not synchronised with the newest versions of SC2, you will run into issues. If you are using Linux, you should be fine, because you can choose, which version of SC2 should be installed. 

In my case I could run the training on the Linux server, but I could not let the agent run locally on my machine. You can also run the agent on your Linux machine und view the replay file with [pysc2](https://github.com/deepmind/pysc2). 

If you want to use spatial features or you want your agent to play the game more like a human (or both) you migth find it better to use pysc2. Moreover you wont have these problems regarding asynchronous updating of framework and SC2 API changes. 

## Source Code

Our source code is mostly a team project, find it in folder bot. In /bot/JAS/Agents/produce_marines.py you can find the agent code for the RL scenario. In /bot/JAS/game_commander.py you can see a on_step method that will be called every time step. Here you may see, that we initally designed a structure where you could put in a number of sub-agents, for specific tasks. You might insert some there. In /bot/ray_server you can find the ray/rllib configs for DQN and Ape-X. My code for the plots can be found in /bot/scripts. At some points the code is quick and dirty, because the reinforcement learning was the subject, not the nicest code ;-)

In case you cannot get the code working, here some hints: Some Python packages may be not present, here is a bunch of packages I installed by reading the error messages: 

```python
pip install async_timeout aiohttp logbook gym bitarray sklearn
pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.7.0.dev2-cp36-cp36m-manylinux1_x86_64.whl

pip install opencv-python
pip install psutil
pip install lz4
pip install s2clientprotocol
```

You might need to install more than that. I will not support you by installing or get this thing run. Sorry ;-) 

If you have no NVIDIA GPU present, you need to configure that in ray_server.py. If you have a shared server, you might want to only use one GPU. Try this: ```CUDA_VISIBLE_DEVICES=1 python yoursrcript.py```.

**Screen** ist your friend, if you want to start mutliple workers while not being logged in via SSH. 

**nvidia-smi** is a tool to show the workload on your NVIDIA GPUs, **top** shows the workload on your CPUs and RAM. 

## Scenario: BuildMarines
![scenario](https://github.com/shufflebyte/sc2_dqn/doc/scenario_snipped.png "Scenario BuildMarines")


## The neural network architecture
![neural network](https://github.com/shufflebyte/sc2_dqn/doc/neural_network.pdf "Neural Netowork")

## The result of tuned DQN with human reference
![neural network](https://github.com/shufflebyte/sc2_dqn/doc/dqn_result.png "Result of tuned DQN")
