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
You will need the gym and numpy libraries. If you want to use torch/tensorflow for question 3 that is also ok 
(but definitely not necessary, make sure your code runs on the latest version of the library). 
Ask Emilio if you want to use any other library.

You should implement the algorithms yourselves, not using implementations by anybody else.
****
## Tasks

### 1. Q-learning
Implement a Q-learning agent with 95% epsilon greedy and run it on the 
NChain-v0 and FrozenLake-v0.  Plot the moving average of the rewards, averaged over 10 runs.
   
What are your conclusions?

### 2. Policy based RL
#### **Policy Gradient???**

### 3. We will have a competition. 
   Which group can make an agent that learns the best on an unknown environment?
   Be creative and use what you have learned in the course and come up with something that works well.
Include experiments (as in Q1) with your agent and describe the agent in the report. Argue for why this is a reasonable approach. 
   It can also be a good idea to describe some approaches you have tried but that were not successful.

Some suggestions of things to try: smarter exploration, model based RL, distributional methods instead of only expectations, ensemble methods.
Anything that you think seems resonable is ok (but you are only allowed to interact with the environment through the step command as well as reset on episode end)

Your grade is not decided by how well you perform, but performing well can obviously help show that you have understood.
#### Competition rules
* Your submission to the competition is the file competitionagent.py
* The goal is to perform as well as possible (highest total reward in T time steps or N episodes depending on if the environment "finishes") on some unknown environments which 
   we will run after submission.
* We will make a leaderboard of the top 20% of submissions 
   (others will only be notified directly). 
* The leaderboard will be created by the average ranking for a team in each of the environments.
****

##  Submission
Upload your submission to canvas as usual. 
Include a report, up to 1.5 pages pdf (including figures, not references).
Include an agent named competitionagent.py (make sure this runs with default run_experiment.py)
Include the rest of the agents and neccessary code also.

## Grading
The grading will be based on:
1. **Your report**, how well you describe and analyse your results and how clear your text is.
2. **Correctness**, how well your solutions perform as expected, and to some degree how well you manage to create a good algorithm in Q3.
3. **Creativity**, coming up something good for Q3. Also make sure it is sufficiently challenging 
   (not just tuning an epsilon greedy parameter in Q-learning...) 

## Getting input
To get input on your ideas for Q3 you can very briefly describe your ideas (1/3 page should be enough) and submit under the assignment "Proposal". 
You should get feedback within a few days.
See the deadline for proposal on canvas. This is voluntary but strongly recommended.
****
## Tips
Note that some environments have a **done** flag to indicate that the episode is finished. 
Make sure you take this into account. 
For example, you do not want any future rewards from the terminal state and no transition from terminal state to starting state. 