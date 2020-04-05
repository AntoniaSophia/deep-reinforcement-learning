# Install environment for Windows

## Visual studio code
* Just normal installation for Windows 10 - X86_64
* Installed python extension

## Anaconda Python Environment
* Installed x86_64 version for python3.7 --> https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
* Opened a Anaconda Prompt from start menu
* Created a *drl* environment ```conda create -n drl python=3.7```
* Created a Shortcut on my desktop to start anaconda prompt with activated drl environment

## Fork of udacity deep-reinforcement-repository
* To save my code and progress in executing the course I generated a fork of the original udaity repo for this course on my github account https://github.com/AntoniaSophia/deep-reinforcement-learning
* Created a branch called *antoniasophia* to save my work
* Added the the gym repository https://github.com/openai/gym as submodule to my repo 
  

## Install python dependencies for the DRL course
* Openend prompt with drl environemnt, than:
```
cd gym
pip install -e .
pip install -e ".[all]"
```
I Ignored the errors because of missing mujoco-py
* Installed jupyter and other dependencies
```
conda install jupyter matplotlib seaborn
```
