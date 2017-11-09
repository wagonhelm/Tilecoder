# Tilecoding RL
This implementation of tilecoding is an old attempt of mine and it does not generalize well to the ranges of different dimensions.  A better implementation can be found [here](https://github.com/MeepMoop/tilecoding).

# Mountain Car (Q - Learning Works Best)
Solve mountain car problem using tilecoding and a Q or Double Q Algorithim. The tile coding method is a computationally simple linear method for representing small dimensional states and can solve the problem (average reward of -110 over 100 episodes) much faster than many other attempts by people on openai's gym. On average the Q learning algorithim solves in under 2000 episodes and probably could solve it even faster by tweaking the tilecoder and learning rate.

# CartPole (DQ - Learning Works Best)
Double Q seems to perform better probably due to the fact it randomly initializes two values for theta and also prevents one policy with goodish results from being exploited.  This example is allready beginning to show the limitations of Tilecoding as I had to drastically increase the number of tilings and the amount of the tiles per tiling.  Had to overide max and min values for observations because gym defaults were extremely high numbers which threw my tilecoder off.

![alt tag](https://github.com/wagonhelm/RL/blob/master/tileCoderFinal.gif)
