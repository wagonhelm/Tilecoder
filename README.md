# Tilecoding RL

# Mountain Car (Q - Learning Works Best)
Solve mountain car problem using tilecoding and a Q or Double Q Algorithim. The tile coding method is a computationally simple linear method for representing states and can solve the problem (average reward of -110 over 100 episodes) much faster than many other attempts by people on openai's gym. On average the Q learning algorithim solves in under 2000 episodes and probably could solve it even faster by tweaking the tilecoder and learning rate.

# CartPole (DQ - Learning Works Best)

I feel like a huge part of solving CartPole quickly is lucking out with a good random theta initialization.  Double Q seems to perform a little better.  This example is allready beginning to show the limitations of Tilecoding as I had to drastically increase the number of tilings and the size of the tiles.  Have'nt expirimented with tweaking values to much.  Had to overide max and min values for cart and pole velocity because gym defaults where extremely high numbers which threw my tilecoder off.

![alt tag](https://github.com/wagonhelm/mountainCar/blob/master/tileCoderFinal.gif)
