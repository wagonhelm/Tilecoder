import gym
import numpy as np
env = gym.make("CartPole-v0")

class tilecoder:
	
	def __init__(self, numTilings, tilesPerTiling):
		self.maxIn = env.observation_space.high
		self.maxIn[0] = 3
		self.maxIn[1] = 4
		self.maxIn[3] = 4
		self.minIn = env.observation_space.low
		self.minIn[0] = -3
		self.minIn[1] = -4
		self.minIn[3] = -4
		self.numTilings = numTilings
		self.tilesPerTiling = tilesPerTiling
		self.dim = len(self.maxIn)
		self.numTiles = (self.tilesPerTiling**self.dim) * self.numTilings
		self.actions = env.action_space.n
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


if __name__ == "__main__":

	tile = tilecoder(4,22)
	theta = np.random.uniform(-0.001, 0, size=(tile.n))
	alpha = (.1/ tile.numTilings)
	gamma = 1
	numEpisodes = 100000
	rewardTracker = []
	epsilon = 0.5

	for episodeNum in range(1,numEpisodes+1):
		G = 0
		step = 0
		state = env.reset()
		while True:
			#env.render()
			F = tile.getFeatures(state)
			Q = tile.getQ(F, theta)
			
			if np.random.rand() > epsilon:
				action = env.action_space.sample()
				epsilon += epsilon * 0.0001
			else:
				action = np.argmax(Q)

			state2, reward, done, info = env.step(action)
			G += reward
			delta = G*(gamma**step) - Q[action]
			step += 1

			if done == True:
				theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))
				rewardTracker.append(G)
				break
			Q = tile.getQ(tile.getFeatures(state2), theta)
			delta += gamma*np.max(Q)
			theta += np.multiply((alpha*delta), tile.oneHotVector(F,action))
			state = state2

		print(G)
		if episodeNum % 25 == 0:
			print ('Epsilon = {}'.format(epsilon))
			print('Average Total Reward = {}'.format((sum(rewardTracker)/episodeNum)))
		if sum(rewardTracker[episodeNum-100:episodeNum])/100 >= 195:
			print('Solve in {} Episodes'.format(episodeNum))
			break
