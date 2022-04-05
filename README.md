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
Plot the moving average of the rewards while your agent learns, possibly by episode, averaged over 5 runs (i.e. restarting the training). 
Include error-bars (or something similar) indicating the 95% confidence intervals calculated from your variance of the 5 runs.

### Algorithms
Implement Q-learning, Double Q-learning and SARSA agents with 95% epsilon greedy and run it on the 
environments. Visualize the Q-functions for each state. Does it reach the optimal policy? Does it depend on the initialization of Q?
   
What are your conclusions? How do the agents perform? Discuss if the behaviour is expected based on what you have learned in the course.

## Grading
The grading will be based on:
1. **Your report**, how well you describe, analyse and visualize your results and how clear your text is.
2. **Correctness**, how well your solutions perform as expected, and to some degree how well you manage to create a good algorithm in Q4.
3. **Creativity**, coming up something good for Q4. Also make sure it is sufficiently challenging 
   (not just tuning an epsilon greedy parameter in Q-learning...) 


## Tips
Note that some environments have a **done** flag to indicate that the episode is finished. 
Make sure you take this into account. 
For example, you do not want any future rewards from the terminal state and no transition from terminal state to starting state. 
