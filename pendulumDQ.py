## APPEARS TO BE LEARNING VERY SLOWLY 
## ACTION CHOICES AND TILECODER NEED TO BE TWEAKED

import gym
import numpy as np
env = gym.make("Pendulum-v0")
import time
import random as rand

class tilecoder:
	
	def __init__(self, numTilings, tilesPerTiling):
		self.maxIn = env.observation_space.high
		self.minIn = env.observation_space.low
		self.numTilings = numTilings
		self.tilesPerTiling = tilesPerTiling
		self.dim = len(self.maxIn)
		self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
		self.actions = 11
		self.n = self.numTiles * self.actions
		self.tileSize = np.divide(np.subtract(self.maxIn,self.minIn), self.tilesPerTiling-1)
		
	def getFeatures(self, variables):
		### ENSURES LOWEST POSSIBLE INPUT IS ALWAYS 0
		self.variables = np.subtract(variables, self.minIn)
		tileIndices = np.zeros(self.numTilings)
		matrix = np.zeros([self.numTilings,self.dim])
		for i in range(self.numTilings):
			for i2 in range(self.dim):
				matrix[i,i2] = int(self.variables[i2] / self.tileSize[i2] \
					+ i / self.numTilings)
		for i in range(1,self.dim):
			matrix[:,i] *= self.tilesPerTiling**i
		for i in range(self.numTilings):
			tileIndices[i] = (i * (self.tilesPerTiling**self.dim) \
				+ sum(matrix[i,:])) 
		return tileIndices

	def oneHotVector(self, features, action):
		oneHot = np.zeros(self.n)
		for i in features:
			index = int(i + (self.numTiles*action))
			oneHot[index] = 1
		return oneHot

	def getVal(self, theta, features, action):
		val = 0 
		for i in features:
			index = int(i + (self.numTiles*action))
			val += theta[index]
		return val

	def getQ(self, features, theta):
		Q = np.zeros(self.actions)
		for i in range(self.actions):
			Q[i] = tile.getVal(theta, features, i)
		return Q

	def action2float(self, a):
		if a == 0:
			return([rand.uniform(-2,-1.5)])
		if a == 1:
			return([rand.uniform(-1.5,-1)])
		if a == 2:
			return([rand.uniform(-1,-0.75)])
		if a == 3:
			return([rand.uniform(-0.75, -0.5)])
		if a == 4:
			return([rand.uniform(-0.5,-0.25)])
		if a == 5:
			return([0])
		if a == 6:
			return([rand.uniform(0.25, 0.5)])
		if a == 7:
			return([rand.uniform(0.5, 0.75)])
		if a == 8:
			return([rand.uniform(0.75, 1)])
		if a == 9:
			return([rand.uniform(1,1.5)])
		if a == 10:
			return([rand.uniform(1.5,2)])


if __name__ == "__main__":

	tile = tilecoder(4,20)
	theta1 = np.random.uniform(-0.001, 0, size=(tile.n))
	theta2 = np.random.uniform(-0.001, 0, size=(tile.n))
	alpha = .1/ tile.numTilings * 2
	numEpisodes = 100000
	episodeSum = 0
	rewardtracker = []

	for episodeNum in range(1,numEpisodes+1):
		G = 0
		state = env.reset()
		for steps in range(10000):
			F = tile.getFeatures(state)
			Q1 = tile.getQ(F, theta1)
			Q2 = tile.getQ(F, theta2)
			action = np.argmax(Q1+Q2)
			
			state2, reward, done, info = env.step(tile.action2float(action))
			G += reward
			
			delta1 = reward - Q1[action]
			delta2 = reward - Q2[action]
		
			Q1_ = tile.getQ(tile.getFeatures(state2), theta1)
			Q2_ = tile.getQ(tile.getFeatures(state2), theta2)
			delta1 = reward + gamma*(Q2_[np.argmax(Q1_)] - Q1[action])
			delta2 = reward + gamma*(Q1_[np.argmax(Q2_)] - Q2[action])

			if np.random.rand() > 0.5:
				theta1 += np.multiply((alpha*delta1), tile.oneHotVector(F,action))
			else:
				theta2 += np.multiply((alpha*delta2), tile.oneHotVector(F,action))

			state = state2

		print("Episode Total Reward = {}".format(G))
		rewardtracker.append(G)
		print("Total Average = {}".format(sum(rewardtracker)/episodeNum))
			
