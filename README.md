# Deep Turtles
This repository includes executable programs and tutorials for teaching turtles how to play variations of "tag" with Deep Q-Networks, users can modify network configuration, hyperparameters, in addition to the game enviroment to anaylize the preformance of your turtle agent(s) in realtime. The intent of this project is study MDPs, state-action space defintions, sampling techniques, multi-agent, and adversariale approaches to training Deep Q-networks. In addition a series of visualization tools are included to allow the user to examine how the weights of an agent's Deep Q-Network change during the training process. These visualization tools aim to provide a more intutative assessment of nerual-networks and help identify optimal learning parameters, reward functions, state-action space defintions, and model configurations. Our research aims to highlight and leverage the presense of behaviroal and enviroment symetery to reduce agent training time and modify agent policies with direct matrix operations. Before procceding please install packages listed under system requirments. Under (Executables and Verision Descriptions) users can select different executable versions of Deep_Turtle to run on their local machine, each verison consitist of different agent(s) configurations, game conditions, and enviroments. Discriptions of topoloical enviroments in which your turtle agents can train in are listed under Turtle Enviroments.



## System Requirments 
- pip install turtles
- pip install Keras
- pip install numpy
- pip install matplotlib
- pip install palettable


## Turtle Enviroments 

###### Bounded Plane - Turtles are restricted to a bounded rectangular plane 
 
###### **Klein bottle** - Edges of plane are "glued together", turtle which passes through one side of the plane appears will reappear on the opposing side.

###### Real projective plane - Add description

###### Boy's surface - Add description


## Executables and Descriptions

###### Deep_Turtle_V1.py - Trains a single blue turtle agent in Bounded Plane two avoid being tagged by 2 red turtles 

###### Deep_Turtle_V2.py - Trains a single blue turtle agent in Klein bottle two avoid being tagged by 2 red turtles 

###### Adversarial_Turtle_Tag.py - Two turtle agents compete aginst one another in a game of tag as they simulatniously learn how to play the game 





# Tutorials for Building Custom Executables

- Version I

- Version II

- Version III

