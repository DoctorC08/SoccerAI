# # PPO.py
# # by christophermao
# # 2/7/24
#
# import time
#
# import torch
#
# import torch.nn as nn
# from torch.optim import Adam
# from torch.distributions import MultivariateNormal
#
# import numpy as np
#
# class PPO:
#     def __init__(self, actor_policy, critic_policy, Env, obs_dim, act_dim): #TODO: Create an actual env class to run everything from
#
#         # Initialize hyperparamaters
#         self._init_hyperparamaters()
#
#         # Extract env info
#         self.env = Env
#         # self.obs_dim = env.observation_space.shape[0]
#         # self.act_dim = env.action_space.shape[0]
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#
#         # Define Actor Critic Methods
#         self.actor = actor_policy(self.obs_dim, self.act_dim)
#         self.critic = critic_policy(self.obs_dim, 1)
#
#         # Define Optimizers for actor and critic
#         self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
#         self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
#
#         # Create variables for matrix
#         # fill_value is std_dev
#         self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
#
#         self.cov_mat = torch.diag(self.cov_var)
#
#         self.logger = {
#             'delta_t': time.time_ns(),
#             't_so_far': 0,                  # Timestems so far
#             'i_so_far': 0,                  # Iterations so far
#             'batch_lens': [],               # episodic lengths in batch
#             'batch_rews': [],               # episodic returns in batch
#             'actor_losses': [],             # losses of actor network in current iteration
#             'lr': 0,
#         }
#
#
#
#
#     def get_action(self, obs):
#         # Query the actor network for a mean action
#         # Same thing as calling self.actor.forward(obs)
#         mean = self.actor(obs)
#
#         # Create our Multivariate Normal Distribution
#         dist = MultivariateNormal(mean, self.cov_mat)
#
#         # Sample an action from the distribution and get its log prob
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#
#         # Sampling should only be used for training as exploration
#         # If testing just return deterministic action
#         if self.deterministic:
#             return mean.detach().numpy(), 1
#
#         # Return the sampled action and the log prob of that action
#         # Note that calling detach() because the action and _log prob are just tensors with computation graphs, so I want
#         # to get rid of the graph and just convert the action to numpy array. Log prob as tensor is fine. Our computation
#         # graph will start later down the line.
#         return action.detach().numpy(), log_prob.detach()
#
#     def compute_rtgs(self, batch_rews):
#         # The rewards-to-go (rtg) per episode per batch to return. The shape will be (num timesteps per episode)
#         batch_rtgs = []
#
#         # Iterate through each episode backwards to maintain same order in batch_rtgs
#         for ep_rews in reversed(batch_rews):
#             discounted_reward = 0 # Discounted reward so far
#
#             for rew in reversed(ep_rews):
#                 discounted_reward = rew + discounted_reward*self.gamma
#                 batch_rtgs.insert(0, discounted_reward)
#
#         # Convert the rewards-to-go into a tensor
#         batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
#
#         return batch_rtgs
#
#     def evaluate(self, batch_obs, batch_acts):
#         # Query critic network for a value V for each obs in batch_obs
#         V = self.critic(batch_obs).squeeze()
#
#         # Calculate the log probs of batch actions using most recent actor networks.
#         # This segment of code is similar to get action()
#         mean = self.actor(batch_obs)
#         dist = MultivariateNormal(mean, self.cov_mat)
#         log_probs = dist.log_prob(batch_acts)
#
#         return V, log_probs, dist.entropy()
#
#     def calculate_gae(self, rewards, values, dones):
#         # List to store computed advantages for each timestep
#         batch_advantages = []
#         print("Dones sample:", dones[:10])
#         print("value sample:", values[:10])
#         print("reward sample:", rewards[:10])
#
#         for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
#             advantages = []
#             last_advantage = 0 # Initialize the last computed advantage
#
#             # Starts from end and goes backward. GAE calculates advantages by the difference between observed rewards
#             # and estimate values of the current and future timesteps. This difference is change by gamma and lambda (lam)
#             # Which helps balance the influence of short-term and long-term rewards.
#             # NOTE: Dones is essential because it lets the algorithm know when an episode ends.
#             for t in reversed(range(len(ep_rews))):
#                 if t + 1 < len(ep_rews):
#                     # Calculate TD (Temporal Difference) error for the current timestep
#                     delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
#                 else:
#                     # Special case at last timestep
#                     delta = ep_rews[t] - ep_vals[t]
#
#                 # Calculate GAE (Generalized Advantage Estimation) for the current timestep
#                 advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
#                 last_advantage = advantage # Update the last advantage for next timestep
#                 advantages.insert(0, advantage) # Insert advantage at the beginning of the list
#
#             # Extend the batch_advantages list with advantages computed for current episode
#             batch_advantages.extend(advantages)
#         print("sample batch advantages:", batch_advantages[:10])
#         print("batch advantages shape:", len(batch_advantages), len(batch_advantages[0]))
#         return torch.stack(batch_advantages, dim=0)
#
#
#     def _init_hyperparamaters(self):
#         # Default values for hyperparamaters
#         self.timesteps_per_batch = 2_000             # Timesteps per batch
#         self.max_timesteps_per_episode = 20           # Timesteps per episode
#         self.n_updates_per_iteration = 5                # Number of times to udpate actor.critic per iteration
#         self.lr = 0.005                                 # Learning rate of actor optimizer
#         self.gamma = 0.95                               # Discount factor when calculating rewards-to-go
#         self.clip = 0.2                                 # Reccomended 0.2, helsp defin eht threshold to clip the ratio during SGA
#         self.lam = 0.98                                 # Lambda Parameter for GAE
#         self.num_minibatches = 6                        # Number of mini-batches for mini-batch update
#         self.ent_coef = 0                               # Entropy coefficient for Entropy Regularization
#         self.target_kl = 0.02                           # KL Divergence threshold
#         self.max_grad_norm = 0.5                        # Gradient Cliping threshold
#
#         # Miscellaneous paramaters
#         self.save_freq = 10                             # How often to save in number of iterations
#         self.deterministic = False                      # If testing don't sample new actions
#         self.seed = None                                # Sets the seed of our program if we want to reproduce outcomes
#
#         # # Change any defauld values to vucustom values for specificed hyperparamaters
#         # for param, val in meters.items():
#         #     exec('self.' + param + '=' + str(val))
#         #
#         #     # Set the seed
#         #     torch.manual_seed(self.seed)
#         #     print(f"Successfully set seed to {self.seed}")
#
#     def _log_summary(self):
#         # Calculate logging values. I use a few python shortcuts to calculate each value
#         # without explaining since it's not too important to PPO; feel free to look it over,
#         # and if you have any questions you can email me (look at bottom of README)
#         delta_t = self.logger['delta_t']
#         self.logger['delta_t'] = time.time_ns()
#         delta_t = (self.logger['delta_t'] - delta_t) / 1e9
#         delta_t = str(round(delta_t, 2))
#
#         t_so_far = self.logger['t_so_far']
#         i_so_far = self.logger['i_so_far']
#         lr = self.logger['lr']
#         avg_ep_lens = np.mean(self.logger['batch_lens'])
#         avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
#         avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
#
#         # Round decimal places for more aesthetic logging messages
#         avg_ep_lens = str(round(avg_ep_lens, 2))
#         avg_ep_rews = str(round(avg_ep_rews, 2))
#         avg_actor_loss = str(round(avg_actor_loss, 5))
#
#         # Print logging statements
#         print(flush=True)
#         print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
#         print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
#         print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
#         print(f"Average Loss: {avg_actor_loss}", flush=True)
#         print(f"Timesteps So Far: {t_so_far}", flush=True)
#         print(f"Iteration took: {delta_t} secs", flush=True)
#         print(f"Learning rate: {lr}", flush=True)
#         print(f"------------------------------------------------------", flush=True)
#         print(flush=True)
#
#         # Reset batch-specific logging data
#         self.logger['batch_lens'] = []
#         self.logger['batch_rews'] = []
#         self.logger['actor_losses'] = []
#
#
#
#     def learn(self, total_timesteps):
#         print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode {self.timesteps_per_batch} timeteps per batch for a total of {total_timesteps} timesteps" )
#
#         t_so_far = 0 # Timesteps simulated so far
#         i_so_far = 0 # Iterations ran so far
#
#         while t_so_far < total_timesteps:
#
#             # Increment t_so_far somewhere below:
#             batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones = self.rollout()
#
#             # Calculate advantage using GAE
#             A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)
#             V = self.critic(batch_obs).squeeze()
#             print("A k shape", A_k.squeeze().shape)
#             print("A k sample:", A_k[:10])
#             print("V shape", V.shape)
#             print("sample of V:", V[:20])
#             batch_rtgs = A_k.squeeze() + V.detach()
#             print("batch rtgs shape:", batch_rtgs.shape)
#             print("batch rtgs sample:", batch_rtgs[:5])
#
#             # Calculate how many timesteps we collected this batch
#             t_so_far += np.sum(batch_lens)
#
#             # Increment number of iterations
#             i_so_far += 1
#
#             self.logger['t_so_far'] = t_so_far
#             self.logger['i_so_far'] = i_so_far
#
#             # Normalize advantages
#             A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
#
#
#             step = batch_obs.size(0)
#             inds = np.arange(step)
#             minibatch_size = step // self.num_minibatches
#             loss = []
#
#             for _ in range(self.n_updates_per_iteration):
#                 # Update Learning Rate as training goes on
#                 # This stabalizes the learning process making it more effective
#
#                 # Learning rate annealing
#                 frac = (t_so_far - 1.0) / total_timesteps
#                 new_lr = self.lr * (1.0-frac)
#
#                 # Make sure frac isn't less than 0
#                 new_lr = max(new_lr, 0.0)
#                 self.actor_optim.param_groups[0]["lr"] = new_lr
#                 self.critic_optim.param_groups[0]["lr"] = new_lr
#
#                 self.logger['lr'] = new_lr
#
#                 # Shufflethe index
#                 np.random.shuffle(inds)
#                 # Mini-batch Update
#                 for start in range(0, step, minibatch_size):
#                     end = start + minibatch_size
#                     idx = inds[start:end]
#
#                     # Extract data from sampled indices
#                     mini_obs = batch_obs[idx]
#                     mini_acts = batch_acts[idx]
#                     mini_log_prob = batch_log_probs[idx]
#                     mini_advantage = A_k[idx]
#                     mini_rtgs = batch_rtgs[idx]
#
#                     # Calculate V_phi and ph_theta(a_t | s_t) and entropy
#                     V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)
#
#                     # Calculate the ratio ph_theta / pi_theta_k
#                     # Subtract logs which is same as dividing the values and then canceling the log with e^log
#                     logratios = curr_log_probs - mini_log_prob
#                     ratios = torch.exp(logratios)
#                     approx_kl = ((ratios - 1) - logratios).mean()
#
#                     # Calculate surrogate losses
#                     surr1 = ratios * mini_advantage
#                     surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
#
#                     # Calculate actor and critic losses
#                     # NOTE: Taking negative min of surrogate because we're trying maximize and Adam optimizer tries to minimize
#                     # so those cancel out
#                     actor_loss = (-torch.min(surr1, surr2)).mean()
#                     print("v shape:", V.shape)
#                     print("mini_rtgs shape", mini_rtgs.shape)
#                     print("mini rtgs sample", mini_rtgs[:5])
#                     critic_loss = nn.MSELoss()(V, mini_rtgs)
#
#                     # Entropy Regularization
#                     entropy_loss = entropy.mean()
#                     # Discount entropy loss by given coefficient
#                     actor_loss = actor_loss - self.ent_coef * entropy_loss
#
#                     # Calculate gradients and perform backward propagation for actor network
#                     # nn.utils.clip_grad_norm calculates the L2 norm of the gradiests and scales them down if the norm surpasses
#                     # the given threshold which helps maintain stable gradient updates and smoother convergance Normally set to 0.5
#                     self.actor_optim.zero_grad()
#                     actor_loss.backward()
#                     nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
#                     self.actor_optim.step()
#
#                     # Calculate gradients and perform backward propagation for critic network
#                     self.critic_optim.zero_grad()
#                     critic_loss.backward()
#                     nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
#                     self.critic_optim.step()
#
#                     loss.append(actor_loss.detach())
#                 # Approximating KL Divergence
#                 if approx_kl > self.target_kl:
#                     break
#             # Log actor loss
#             avg_loss = sum(loss) / len(loss)
#             self.logger['actor_losses'].append(avg_loss)
#
#             # Print a summary of training
#             self._log_summary()
#
#             if i_so_far % self.save_freq == 0:
#                 torch.save(self.actor.state_dict('~/Desktop/RLModels/PPO'))
#                 torch.save(self.critic.state_dict('~/Desktop/RLModels/PPO'))
#
#
#
#     def rollout(self):
#         # Batch data
#         # observations: (number of timesteps per batch, dimension of observation)
#         # actions: (number of timesteps per batch, dimension of action)
#         # log probabilities: (number of timesteps per batch)
#         # rewards: (number of episodes, number of timesteps per episode)
#         # reward-to-go’s: (number of timesteps per batch)
#         # batch lengths: (number of episodes)
#
#         batch_obs = []          # Batch observations
#         batch_acts = []         # Batch actions
#         batch_log_probs = []    # Batch log probability of each action
#         batch_rews = []         # Batch rewards
#         batch_lens = []         # Episode lengths in batch
#         batch_vals = []
#         batch_dones = []        # Episode done or not
#
#         # Episodic data and will be reset for every new episode
#         ep_rews = []
#         ep_vals = []
#         ep_dones = []
#
#         t = 0
#
#         while t < self.timesteps_per_batch:
#             # Episode Reward
#             ep_rews = []
#             ep_vals = []
#             ep_dones = []
#
#             obs, agent_obs, _ = self.env.reset()
#             done = False
#
#             for ep_t in range(self.max_timesteps_per_episode):
#
#                 ep_dones.append(done)
#
#                 t += 1  # Add timestep ran in batch
#                 batch_obs.append(agent_obs) # Add observations
#
#                 # Calculate action and take a step in env
#                 action, log_prob = self.get_action(agent_obs)
#                 val = self.critic(agent_obs)
#
#                 obs, agent_obs, rew, terminated, truncated = self.env.step(obs, [np.argmax(action)], ep_t, render_mode=False)
#                 done = terminated or truncated
#
#                 # Collect reward, action, and log prob
#                 ep_rews.append(rew)
#                 ep_vals.append(val.flatten())
#                 batch_acts.append(action)
#                 batch_log_probs.append(log_prob)
#
#                 # Define previous observations for next timestep because updating state will require previous states
#                 if done:
#                     break
#
#             # Collect Eposodic Rewards and Length
#             batch_lens.append(ep_t + 1)# Add one because it starts at 0
#             batch_rews.append(ep_rews)
#             batch_vals.append(ep_vals)
#             batch_dones.append(ep_dones)
#
#         # Reshape data as tensors in the shape specified before returning
#         batch_obs = torch.stack(batch_obs, dim=0)
#         batch_acts = torch.tensor(batch_acts, dtype=torch.float)
#         batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
#         print(batch_obs.shape)
#         print(batch_acts.shape)
#         print(batch_log_probs.shape)
#
#         self.logger['batch_rews'] = batch_rews
#         self.logger['batch_lens'] = batch_lens
#
#         return batch_obs, batch_acts, batch_log_probs, batch_rews, batch_lens, batch_vals, batch_dones
#
#


"""
	The file contains the PPO class to train with.
	NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
			It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
	"""
		This is the PPO class we will use as our model in main.py
	"""
	def __init__(self, policy_class, env, **hyperparameters):
		"""
			Initializes the PPO model, including hyperparameters.

			Parameters:
				policy_class - the policy class to use for our actor/critic networks.
				env - the environment to train on.
				hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

			Returns:
				None
		"""
		# Make sure the environment is compatible with our code
		assert(type(env.observation_space) == gym.spaces.Box)
		assert(type(env.action_space) == gym.spaces.Box)

		# Initialize hyperparameters for training with PPO
		self._init_hyperparameters(hyperparameters)

		# Extract environment information
		self.env = env
		self.obs_dim = env.observation_space.shape[0]
		self.act_dim = env.action_space.shape[0]

		 # Initialize actor and critic networks
		self.actor = policy_class(self.obs_dim, self.act_dim)                                                   # ALG STEP 1
		self.critic = policy_class(self.obs_dim, 1)

		# Initialize optimizers for actor and critic
		self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		# Initialize the covariance matrix used to query the actor for actions
		self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
		self.cov_mat = torch.diag(self.cov_var)

		# This logger will help us with printing out summaries of each iteration
		self.logger = {
			'delta_t': time.time_ns(),
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}

	def learn(self, total_timesteps):
		"""
			Train the actor and critic networks. Here is where the main PPO algorithm resides.

			Parameters:
				total_timesteps - the total number of timesteps to train for

			Return:
				None
		"""
		print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
		print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
		t_so_far = 0 # Timesteps simulated so far
		i_so_far = 0 # Iterations ran so far
		while t_so_far < total_timesteps:                                                                       # ALG STEP 2
			# Autobots, roll out (just kidding, we're collecting our batch simulations here)
			batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()                     # ALG STEP 3

			# Calculate how many timesteps we collected this batch
			t_so_far += np.sum(batch_lens)

			# Increment the number of iterations
			i_so_far += 1

			# Logging timesteps so far and iterations so far
			self.logger['t_so_far'] = t_so_far
			self.logger['i_so_far'] = i_so_far

			# Calculate advantage at k-th iteration
			V, _ = self.evaluate(batch_obs, batch_acts)
			A_k = batch_rtgs - V.detach()                                                                       # ALG STEP 5

			# One of the only tricks I use that isn't in the pseudocode. Normalizing advantages
			# isn't theoretically necessary, but in practice it decreases the variance of
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
			A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

			# This is the loop where we update our network for some n epochs
			for _ in range(self.n_updates_per_iteration):                                                       # ALG STEP 6 & 7
				# Calculate V_phi and pi_theta(a_t | s_t)
				V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

				# Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
				# NOTE: we just subtract the logs, which is the same as
				# dividing the values and then canceling the log with e^log.
				# For why we use log probabilities instead of actual probabilities,
				# here's a great explanation:
				# https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
				# TL;DR makes gradient ascent easier behind the scenes.
				ratios = torch.exp(curr_log_probs - batch_log_probs)

				# Calculate surrogate losses.
				surr1 = ratios * A_k
				surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

				# Calculate actor and critic losses.
				# NOTE: we take the negative min of the surrogate losses because we're trying to maximize
				# the performance function, but Adam minimizes the loss. So minimizing the negative
				# performance function maximizes it.
				actor_loss = (-torch.min(surr1, surr2)).mean()
				critic_loss = nn.MSELoss()(V, batch_rtgs)

				# Calculate gradients and perform backward propagation for actor network
				self.actor_optim.zero_grad()
				actor_loss.backward(retain_graph=True)
				self.actor_optim.step()

				# Calculate gradients and perform backward propagation for critic network
				self.critic_optim.zero_grad()
				critic_loss.backward()
				self.critic_optim.step()

				# Log actor loss
				self.logger['actor_losses'].append(actor_loss.detach())

			# Print a summary of our training so far
			self._log_summary()

			# Save our model if it's time
			if i_so_far % self.save_freq == 0:
				torch.save(self.actor.state_dict(), './ppo_actor.pth')
				torch.save(self.critic.state_dict(), './ppo_critic.pth')

	def rollout(self):
		"""
			Too many transformers references, I'm sorry. This is where we collect the batch of data
			from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
			of data each time we iterate the actor/critic networks.

			Parameters:
				None

			Return:
				batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
				batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
				batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
				batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
				batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
		"""
		# Batch data. For more details, check function header.
		batch_obs = []
		batch_acts = []
		batch_log_probs = []
		batch_rews = []
		batch_rtgs = []
		batch_lens = []

		# Episodic data. Keeps track of rewards per episode, will get cleared
		# upon each new episode
		ep_rews = []

		t = 0 # Keeps track of how many timesteps we've run so far this batch

		# Keep simulating until we've run more than or equal to specified timesteps per batch
		while t < self.timesteps_per_batch:
			ep_rews = [] # rewards collected per episode

			# Reset the environment. sNote that obs is short for observation.
			obs, _ = self.env.reset()
			done = False

			# Run an episode for a maximum of max_timesteps_per_episode timesteps
			for ep_t in range(self.max_timesteps_per_episode):
				# If render is specified, render the environment
				if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
					self.env.render()

				t += 1 # Increment timesteps ran this batch so far

				# Track observations in this batch
				batch_obs.append(obs)

				# Calculate action and make a step in the env.
				# Note that rew is short for reward.
				action, log_prob = self.get_action(obs)
				obs, rew, truncated, terminated, _ = self.env.step(action)
				done = truncated or terminated
				# Track recent reward, action, and action log probability
				ep_rews.append(rew)
				batch_acts.append(action)
				batch_log_probs.append(log_prob)

				# If the environment tells us the episode is terminated, break
				if done:
					break

			# Track episodic lengths and rewards
			batch_lens.append(ep_t + 1)
			batch_rews.append(ep_rews)

		# Reshape data as tensors in the shape specified in function description, before returning
		batch_obs = torch.tensor(batch_obs, dtype=torch.float)
		batch_acts = torch.tensor(batch_acts, dtype=torch.float)
		batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
		batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

		# Log the episodic returns and episodic lengths in this batch.
		self.logger['batch_rews'] = batch_rews
		self.logger['batch_lens'] = batch_lens

		return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

	def compute_rtgs(self, batch_rews):
		"""
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
		batch_rtgs = []

		# Iterate through each episode
		for ep_rews in reversed(batch_rews):

			discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
			for rew in reversed(ep_rews):
				discounted_reward = rew + discounted_reward * self.gamma
				batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
		batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

		return batch_rtgs

	def get_action(self, obs):
		"""
			Queries an action from the actor network, should be called from rollout.

			Parameters:
				obs - the observation at the current timestep

			Return:
				action - the action to take, as a numpy array
				log_prob - the log probability of the selected action in the distribution
		"""
		# Query the actor network for a mean action
		mean = self.actor(obs)

		# Create a distribution with the mean action and std from the covariance matrix above.
		# For more information on how this distribution works, check out Andrew Ng's lecture on it:
		# https://www.youtube.com/watch?v=JjB58InuTqM
		dist = MultivariateNormal(mean, self.cov_mat)

		# Sample an action from the distribution
		action = dist.sample()

		# Calculate the log probability for that action
		log_prob = dist.log_prob(action)

		# Return the sampled action and the log probability of that action in our distribution
		return action.detach().numpy(), log_prob.detach()

	def evaluate(self, batch_obs, batch_acts):
		"""
			Estimate the values of each observation, and the log probs of
			each action in the most recent batch with the most recent
			iteration of the actor network. Should be called from learn.

			Parameters:
				batch_obs - the observations from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of observation)
				batch_acts - the actions from the most recently collected batch as a tensor.
							Shape: (number of timesteps in batch, dimension of action)

			Return:
				V - the predicted values of batch_obs
				log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
		"""
		# Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
		V = self.critic(batch_obs).squeeze()

		# Calculate the log probabilities of batch actions using most recent actor network.
		# This segment of code is similar to that in get_action()
		mean = self.actor(batch_obs)
		dist = MultivariateNormal(mean, self.cov_mat)
		log_probs = dist.log_prob(batch_acts)

		# Return the value vector V of each observation in the batch
		# and log probabilities log_probs of each action in the batch
		return V, log_probs

	def _init_hyperparameters(self, hyperparameters):
		"""
			Initialize default and custom values for hyperparameters

			Parameters:
				hyperparameters - the extra arguments included when creating the PPO model, should only include
									hyperparameters defined below with custom values.

			Return:
				None
		"""
		# Initialize default values for hyperparameters
		# Algorithm hyperparameters
		self.timesteps_per_batch = 4800                 # Number of timesteps to run per batch
		self.max_timesteps_per_episode = 1600           # Max number of timesteps per episode
		self.n_updates_per_iteration = 5                # Number of times to update actor/critic per iteration
		self.lr = 0.005                                 # Learning rate of actor optimizer
		self.gamma = 0.95                               # Discount factor to be applied when calculating Rewards-To-Go
		self.clip = 0.2                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA

		# Miscellaneous parameters
		self.render = True                              # If we should render during rollout
		self.render_every_i = 10                        # Only render every n iterations
		self.save_freq = 10                             # How often we save in number of iterations
		self.seed = None                                # Sets the seed of our program, used for reproducibility of results

		# Change any default values to custom values for specified hyperparameters
		for param, val in hyperparameters.items():
			exec('self.' + param + ' = ' + str(val))

		# Sets the seed if specified
		if self.seed != None:
			# Check if our seed is valid first
			assert(type(self.seed) == int)

			# Set the seed
			torch.manual_seed(self.seed)
			print(f"Successfully set seed to {self.seed}")

	def _log_summary(self):
		"""
			Print to stdout what we've logged so far in the most recent batch.

			Parameters:
				None

			Return:
				None
		"""
		# Calculate logging values. I use a few python shortcuts to calculate each value
		# without explaining since it's not too important to PPO; feel free to look it over,
		# and if you have any questions you can email me (look at bottom of README)
		delta_t = self.logger['delta_t']
		self.logger['delta_t'] = time.time_ns()
		delta_t = (self.logger['delta_t'] - delta_t) / 1e9
		delta_t = str(round(delta_t, 2))

		t_so_far = self.logger['t_so_far']
		i_so_far = self.logger['i_so_far']
		avg_ep_lens = np.mean(self.logger['batch_lens'])
		avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
		avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])

		# Round decimal places for more aesthetic logging messages
		avg_ep_lens = str(round(avg_ep_lens, 2))
		avg_ep_rews = str(round(avg_ep_rews, 2))
		avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
		print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
		print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
		print(f"Average Loss: {avg_actor_loss}", flush=True)
		print(f"Timesteps So Far: {t_so_far}", flush=True)
		print(f"Iteration took: {delta_t} secs", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

		# Reset batch-specific logging data
		self.logger['batch_lens'] = []
		self.logger['batch_rews'] = []
		self.logger['actor_losses'] = []
