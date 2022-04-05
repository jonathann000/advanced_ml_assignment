

## RL competition. 
   Which group can make an agent that learns the best on an unknown environment?
   Be creative and use what you have learned in the course and come up with something that works well.
   

Some suggestions of things to try: smarter exploration, model based RL, distributional methods instead of only expectations, ensemble methods.
Previously it has been successful to make sure that you use the data you seen more than just once (as you would with regular Q-learning for example).
Anything that you think seems reasonable is ok (but you are only allowed to interact with the environment through the step command as well as reset on episode end)


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
Include a report, around 0.5 page describing what you have done with your agent.
Include an agent named competitionagent.py (make sure this runs with default run_experiment.py and withouth depending on any extra files)
