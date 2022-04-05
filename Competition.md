

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
* The goal is to perform as well as possible (highest total reward in T time steps or N episodes depending on if the environment is episodic (i.e finishes) on some unknown environments which 
   we will run after submission. The environments will follow the gym format and can be up to 300 states, 8 actions and rewards in [-10,10].
* We will make a leaderboard of the top 25% of submissions 
   (others teams will only be notified directly). 
* The leaderboard will be created by the average ranking for a team in each of the environments.
****

##  Submission
Upload your submission to canvas as usual. 
Include a report, up to 4 pages pdf (including figures, not references). 
Include an agent named competitionagent.py (make sure this runs with default run_experiment.py and withouth depending on any extra files)
Include the rest of the agents and neccessary code also.
