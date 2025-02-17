import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, varpar=False):
        super(ActorCritic, self).__init__()
        self.varpar = varpar
        if varpar:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim),
            )
            self.variance = torch.nn.Parameter(torch.ones(action_dim, dtype=torch.float32))
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim * 2),
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self.action_dim = action_dim
        self.state_dim = state_dim

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        if self.varpar:
            action_mean = self.actor(state)
            action_var = self.variance
        else:
            action_mean, action_var = torch.split(self.actor(state), self.action_dim, dim=-1)
        cov_mat = torch.diag_embed(F.softplus(action_var))

        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def sample(self, state, mean=False):
        with torch.no_grad():
            if self.varpar:
                action_mean = self.actor(state)
                action_var = self.variance
            else:
                action_mean, action_var = torch.split(self.actor(state), self.action_dim, dim=-1)
            cov_mat = torch.diag_embed(F.softplus(action_var))

            dist = MultivariateNormal(action_mean, cov_mat)
            if mean:
                action = action_mean
            else:
                action = dist.sample()

        return action.detach()

    def evaluate(self, state, action):
        if self.varpar:
            action_mean = self.actor(state)
            action_var = self.variance
        else:
            action_mean, action_var = torch.split(self.actor(state), self.action_dim, dim=-1)
        action_var = F.softplus(action_var)

        #         action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class GruAC(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(GruAC, self).__init__()
        # self.hidden_dim = 256
        self.hidden_dim = 200
        self.actor_gru = nn.GRU(
            input_size=state_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=False,
        )
        self.actor_linear = nn.Linear(self.hidden_dim, action_dim * 2)
        self.actor_state = None

        # self.critic_gru = nn.GRU(
        #     input_size=state_dim,
        #     hidden_size=self.hidden_dim,
        #     num_layers=1,
        #     batch_first=False,
        # )
        self.critic_linear = nn.Linear(self.hidden_dim, 1)

        self.action_dim = action_dim
        self.state_dim = state_dim

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        batch_size, state_dim = state.shape
        # assert state_dim == self.state_dim
        state = state.reshape(1, batch_size, state_dim)

        if self.actor_state is None:
            h, self.actor_state = self.actor_gru(state)
        else:
            h, self.actor_state = self.actor_gru(state, self.actor_state)

        h = h.reshape(batch_size, self.hidden_dim)
        action_mean, action_var = torch.split(self.actor_linear(h), self.action_dim, dim=-1)
        cov_mat = torch.diag_embed(F.softplus(action_var))

        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state.reshape(batch_size, state_dim))
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        seq_len, batch_size, state_dim = state.shape
        assert state_dim == self.state_dim
        h, _ = self.actor_gru(state)
        action_mean, action_var = torch.split(self.actor_linear(h), self.action_dim, dim=-1)
        action_var = F.softplus(action_var)

        #         action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # h, _ = self.critic_gru(state)
        state_value = self.critic_linear(h)

        self.actor_state = None
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, K_epochs, eps_clip, entropy, varpar=False, recurrent=False):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        if recurrent:
            self.policy = GruAC(state_dim, action_dim).to(device)
        else:
            self.policy = ActorCritic(state_dim, action_dim, varpar).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.MseLoss = nn.MSELoss()
        self.entropy_bonus = entropy

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.stack(rewards).squeeze(-1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_bonus * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


    def save(self, filename):
        torch.save(
            {
                "actor_critic": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
