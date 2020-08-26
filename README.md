# Deep Turtles
This repository includes executable programs and tutorials for teaching turtles how to play variations of "tag" with Deep Q-Networks, users can modify network configuration, hyperparameters, in addition to the game enviroment to anaylize the preformance of your turtle agent(s) in realtime. The repository also includes a series of visualization tools to allow the user to examine how the weights of an agent's Deep Q-Network change during the training process. These visualization tools aim to provide a more intutative assessment of nerual-networks, help identify training efficency, optimal learning parameters, reward functions, state-action space defintions, model configurations, ect. 

 The intent of this project is study MDPs, state-action space defintions, sampling techniques, multi-agent, and adversariale approaches to training Deep Q-networks. Our research aims to highlight and leverage the presense of behaviroal and enviroment symetery to reduce agent training time and modify agent policies with direct matrix operations.

Before procceding please install packages listed under system requirments. Under (Executables and Verision Descriptions) users can select different executable versions of Deep_Turtle to run on their local machine, each verison consitist of different agent(s) configurations, game conditions, and enviroments. Discriptions of topoloical enviroments in which your turtle agents can train in are listed under Turtle Enviroments.


## System Requirments 
To install all required libraries, download the repo to your local machine, open cmd and navigate to folder, then enter

    pip install -r requirements.txt
     pip install turtles
     pip install Keras
     pip install numpy
     pip install matplotlib
     pip install palettable


## Learning the Not It Policy 
In the first turtle tag seniro we train a blue turtle agent to avoid being caught by two red turtles.
The agents state is defined as [xb,yb,x1r,y1r,x2r,y2r] where xb,yb is the blue turtle coordinates, and x1r,yr1,x2r,yr2, are the red turtle positions. The blue turtle recieves a reward of +1 for each timestep it is not caught by another red turtle and a reward of -100 for being caught. The players get randomly positioned in the enviroment at the start of training and each time the blue turtle is caught. A penalty of -2 is assigned to the blue turtle reward if it attempts to move outside the Bounded_Plane. The Blue turtle actions consist of a selection of a relative heading being either N, NE, E, SE, S , SW, W, NW in combination with to speed options 10 or 20. The red turtles move at a speed of 10 and are programed to always have a relative heading pointing towards the blue turtle. 


## Guided Policy Implentation 
Applying a priority experience replay drastically enhances training efficency. By priority I mean having the model fit to experiences where it losses the game or attempts to run out of the playing area. For example we add the condition below to our experience replay function.

    if reward < 0:  
            for j in range(100):     
                Q_network.train_on_batch(state, updated_Qs)
     
Next we set  reward = (reward + penalty + r) , where r = 100 for all user actions.(We hijack the turtles action selection), use user input rather than action predicted by network. Penalty is -100 for hitting boundary or being taged, reward = 1 for each timestep. For each graident decent step we have, where updated Qs are caluated as current_Q + learning_rate * (reward + discountfactor * target_Q - current_Q)

     if User_Input == True:
        learning_rate = 1
        for j in range(1000):     
            Q_network.train_on_batch(state, updated_Qs) $ Overfit user input 
            
However, for experince replay we have the latter implentation, sampling from minibatch. 


          Update_Frequency = 16,
          Batch_Size = 6,
          Learning_Rate = .2,
          Discount_Factor = 0.9


The image below represent 1000 guided training iterations, you can notice the user input by the delay in the video feed, 
32x16 tanh network with lr.001 and Adadelta optimizer, clipnorm=1



![Policy Guidance For Accelerated Training](https://github.com/Jesse-Redford/Deep_Turtles/blob/master/G_Policy.gif?raw=true)



## Turtle Enviroments 

###### Bounded Plane - Turtles are restricted to a bounded rectangular plane 
###### Klein bottle - Edges of plane are "glued together", turtle which passes through one side of the plane appears will reappear on the opposing side.
###### Real projective plane - Add description
###### Boy's surface - Add description


## Executables and Descriptions

###### Adversarial_Turtle_Tag.py - Two turtle agents compete aginst one another in a game of tag as they simulatniously learn how to play the game 

###### Deep_Not_It_Turtle_Bounded_Plane.py - Trains a single blue turtle agent in Bounded Plane two avoid being tagged by 2 red turtles 

###### Deep_It_Turtle_Bounded_Plane_V1.py - Trains a red/black turtle agent with a red turtle teammate in Bounded Plane to catch a trained blue turtle
###### Deep_It_Turtle_Bounded_Plane_V2.py -  Same as Version 1 but turtle learns w/o awarness of teamates location

###### Deep_Not_It_Turtle_Klein_Bottle.py - Trains a single blue turtle agent in Klein bottle two avoid being tagged by 2 red turtles 


## Tutorials for Building Custom Executables

    import Deep_Turtles as DT
    
    """ Create a Turtle Agent """
    
    # define action space
    min_speed = 10
    max_speed = 20
    speed_resolution = 10
    heading_resolution = 45

     Create_Turtle_Agent(Model_Name = 'Not it Turtle.h5', 
                         Game_Type ='Not It', 
                         Version = 0,  # 
                         Agent_Speeds = range(min_speed,max_speed,speed_resolution),            
                         Agent_Headings = range(0,271,heading_resolution), 
                         Model_Layers = [30,'tanh',15,'tanh'],
                         Learning_Rate = .1)
                    
                        
                        
    """ Train or Evaluate Turtle Agent """
    
    Deep_Turtles_Trainer(Train = True,
                     Inspect_Model = True,
                     Turtle_Agent = 'Not it Turtle.h5',
                     #Turtle_Actions = 'actions.json',
                     Turtle_Experiences = 'blue_experiences.json',
                     Game_Type = 'Not It',          # Not It, It
                     Topology = 'Bounded Plane',    # Bounded Plane Klein bottle
                     Enviroment = [400,400,1],          # x,y,resolution
                     Reward_Type = 'Non-Sparce',            # Non-Sparce, Sparce, Non-negative, Non-positive, 
                     Epslion_Scheme = 'linear_decay',  # linear_decay, expo_decay, cos
                     Enforce_Penalties = True,
                     Training_Steps = 5000,
                     Update_Frequency = 32,
                     Batch_Size = 64,
                     Learning_Rate = 0.2,
                     Discount_Factor = 0.99)



## Examples, Benchmarks, and PreTrained Turtles

###### Add gif images of policy Rollout, NN visualizer, and links to pretrain models with description
