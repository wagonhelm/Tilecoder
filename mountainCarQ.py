import gym
import numpy as np
env = gym.make("CartPole-v0")

class tilecoder:
	
	def __init__(self, numTilings, tilesPerTiling):
		self.maxIn = env.observation_space.high
		self.minIn = env.observation_space.low
		self.numTilings = numTilings
		self.tilesPerTiling = tilesPerTiling
		self.dim = len(self.maxIn)
		self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
		self.actions = env.action_space.n
		self.n = self.numTiles * self.actions
		self.tileSize = np.divide(np.subtract(self.maxIn,self.minIn), self.tilesPerTiling-1)
		print(self.maxIn, self.minIn)	
	
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


if __name__ == "__main__":

	tile = tilecoder(4,18)
	theta = np.random.uniform(-0.001, 0, size=(tile.n))
	alpha = (.1/ tile.numTilings)*3.2
	gamma = 1
	numEpisodes = 100000
	rewardTracker = []
	episodeSum = 0
	counter = 0 

	for episodeNum in range(1,numEpisodes+1):
		G = 0
		state = env.reset()
		while True:
			env.render()
			F = tile.getFeatures(state)
			Q = tile.getQ(F, theta)
			action = np.argmax(Q)
			state2, reward, done, info = env.step(action)
			G += reward
			delta = reward - Q[action]
			if done == True:
				theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))
				episodeSum += G
				rewardTracker.append(G)
				if episodeNum %100 == 0:
					print('Average reward = {}'.format(episodeSum / 100))
					episodeSum = 0
				break
			Q = tile.getQ(tile.getFeatures(state2), theta)
			delta += gamma*np.max(Q)
			theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))
			state = state2

		# if episodeNum > 100:
		# 	if sum(rewardTracker[episodeNum-100:episodeNum])/100 >= -110:
		# 		print('Solve in {} Episodes'.format(episodeNum))
		# 		break
