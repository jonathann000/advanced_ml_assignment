# Reinforcement Learning assignment

The goal of this assignment is to create your own reinforcement learning assignments.
This github repo contains both the instructions and the files needed for the assignment.
The assignment will use the [OpenAI gym](https://gym.openai.com/) structure, 
I suggest you familiarize yourselves with this before getting started.

**agent.py** is a template agent for you to fill in, 
make sure you follow the format to be able to participate in the competition (more on that later).

**riverswim.py** contains an environment that you can experiment with, but can also serve as a template for making your own
environments.

**run_experiment.py** serves as a template for how to run your experiments, it allows you to load different agents and 
environments you might create by running. You can either call the existing gym environments or the ones you have created.

## General instructions
You will need the gym and numpy libraries. If you want to use torch/tensorflow for question 4 that is also ok 
(but definitely not necessary, make sure your code runs on the latest version of the library). 
Ask Emilio if you want to use any other library.

You should implement the algorithms yourselves, not using implementations by anybody else. 
****
## Tasks
Briefly describe each algorithm and how it works. 
For all methods, run your experiments on NChain-v0 and FrozenLake-v0.  
Plot the moving average of the rewards, averaged over 5 runs. 
Include error-bars (or something similar) indicating the 95% confidence intervals.

### 1. Q-learning
Implement a Q-learning agent with 95% epsilon greedy and run it on the 
environments. Visualize the Q-functions for each state. 
   
What are your conclusions?

### 2. Policy based RL
Implement the A2C algorithm and run on the environments. Since we are in a tabular environment, 
you should parametrize the policy (and value function.) independently for each state (no function approximation). 
You can either calculate the gradient updates yourselves or use torch, tensorflow etc. It will probably be neccesary to do the updates in batch and with a suitable learning rate.
Visualize the policy and the value-function of the agent.

### 3. Compare the methods
Make a list describing the conceptual differences between Q-learning and A2C, make sure to mention on/off-policy, exploration etc.


### 4. Time for competition. 
   Which group can make an agent that learns the best on an unknown environment?
   Be creative and use what you have learned in the course and come up with something that works well.
   Include experiments (as in Q1-2) with your agent and describe the agent in the report. Argue for why this is a reasonable approach. 
   It can also be a good idea to describe some approaches you have tried but that were not successful.
   Make sure to visualize the behaviour of your agent.

Some suggestions of things to try: smarter exploration, model based RL, distributional methods instead of only expectations, ensemble methods.
Anything that you think seems reasonable is ok (but you are only allowed to interact with the environment through the step command as well as reset on episode end)

Your grade is not decided by how well you perform, but performing well can obviously help show that you have understood.

#### Competition rules
* Your submission to the competition is the file competitionagent.py
* The algorithm can not be "unreasonably slow" (Discuss with Emilio if neccessary). Anything comparable with Q-learning and A2C is definately ok.
* The goal is to perform as well as possible (highest total reward in T time steps or N episodes depending on if the environment "finishes") on some unknown environments which 
   we will run after submission.
* We will make a leaderboard of the top 20% of submissions 
   (others will only be notified directly). 
* The leaderboard will be created by the average ranking for a team in each of the environments.
****

##  Submission
Upload your submission to canvas as usual. 
Include a report, up to 2 pages pdf (including figures, not references).
Include an agent named competitionagent.py (make sure this runs with default run_experiment.py)
Include the rest of the agents and neccessary code also.

## Grading
The grading will be based on:
1. **Your report**, how well you describe, analyse and visualize your results and how clear your text is.
2. **Correctness**, how well your solutions perform as expected, and to some degree how well you manage to create a good algorithm in Q3.
3. **Creativity**, coming up something good for Q3. Also make sure it is sufficiently challenging 
   (not just tuning an epsilon greedy parameter in Q-learning...) 

## Getting input
To get input on your ideas for Q4 you can very briefly describe your ideas (1/3 page should be enough) and submit under the assignment "Proposal". 
You should get feedback within a few days.
See the deadline for proposal on canvas. This is voluntary but strongly recommended.
****
## Tips
Note that some environments have a **done** flag to indicate that the episode is finished. 
Make sure you take this into account. 
For example, you do not want any future rewards from the terminal state and no transition from terminal state to starting state. 
