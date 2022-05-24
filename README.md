# Reinforcement Learning assignment

The goal of this assignment is to create your own reinforcement learning agents.
This github repo contains both the instructions and the files needed for the assignment.
The assignment will use the [gym](https://www.gymlibrary.ml/) structure, 
I suggest you familiarize yourselves with this before getting started.
You might need to do '''pip install gym[toy_text]''' to install the FrozenLake environment.

**agent.py** is a template agent for you to fill in.

**riverswim.py** contains an environment that you can experiment with, but can also serve as a template for making your own
environments. '''python3 run_experiment.py --env riverswim''' will run this environment.

**run_experiment.py** serves as a template for how to run your experiments, it allows you to load different agents and 
environments you might create by running. You can either call the existing gym environments or the ones you have created.

## General instructions
You will need the gym and numpy libraries. 

You should implement the algorithms yourselves, not using implementations by anybody else. 
****
## Tasks
Briefly describe each algorithm and how it works. 
For all methods, run your experiments on riverswim (included locally in this repository) and FrozenLake-v1 (which can be found in the gym library, you might need to install).  
Plot the moving average of the rewards while your agent learns, possibly by episode, averaged over 5 runs (i.e. restarting the training). 
Include error-bars (or something similar) indicating the 95% confidence intervals calculated from your variance of the 5 runs.

### Algorithms
Implement Q-learning, Double Q-learning and SARSA agents with 5% epsilon greedy (95% greedy action) and run it on the 
environments. Set gamma=0.95. Visualize the Q-functions for each state (make sure it is easy to interpret, i.e. not just a Q-table for FrozenLake). Draw/visualize the greedy policy obtained from the Q-function. Does it reach the optimal policy? Does it depend on the initialization of Q?
   
What are your conclusions? How do the agents perform? Discuss if the behaviour is expected based on what you have learned in the course.

## Grading
The grading will be based on:
1. **Your report**, how well you describe, analyse and visualize your results and how clear your text is.
2. **Correctness**, how well your solutions perform as expected.


## Tips
- Note that some environments have a **done** flag to indicate that the episode is finished. Make sure you take this into account. 
   - For example, you do not want any future rewards from the terminal state and no transition from terminal state to starting state. 
- In practice it is often good to initialize  Q-tables optimistically (larger than you expect the true table to be, not with zeroes). It can be interesting to think about why this is (but not necessary for the assignment).
- The explorative policy might look bad due to the epsilon-exploration, good to check if the Q-table seems to give a good policy. You can always compare with the Bellman equation.
- Since FrozenLake is slippery, it might look like good policy is sub-optimal. There are some subtle dynamics coming from that the agent can only slip perpendicularly to the direction of the action that can be worth thinking about.
- The amount of steps in the run_experiments is just an example, you will probably need more steps.
- You will need to have alpha "sufficiently small" otherwise there will be high variance (alpha=1 would mean that you ignore the old value and overwrite it).

## Submission
Make sure to include both your code and your report when you submit your assignment. 
