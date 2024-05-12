import gymnasium as gym
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import numpy as np
from collections import namedtuple, deque


# environment setup
env = gym.make("CartPole-v1", render_mode="human")


# replay memory setup
import random

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

class Replay_Memory(object):  # object (?)
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
replay_memory = Replay_Memory(10000)


# Q-network
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_observations, 128),
            nn.ReLU(0.1),
            nn.Linear(128, 128),
            nn.ReLU(0.1),
            nn.Linear(128, num_actions))

    def forward(self, x):
        return self.layers(x)
    

# training
replay_memory_batch_size = 128
gamma = 0.99  # discount

# epsilon-greedy
epsilon_initial = 0.9
epsilon_final = 0.05
epsilon_decay_rate = 1000

tau = 0.005  # target network update rate
learning_rate = 1E-4

# network setup
num_actions = env.action_space.n

observation, observation_info = env.reset()
num_observations = len(observation)

print(num_actions, num_observations)

policy_net = DQN(num_observations, num_actions).to(device)
target_net = DQN(num_observations, num_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())  # (?)

optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, amsgrad=True)

steps_done = 0
def select_action(state):
    """select action based on epsilon-greedy."""
    global steps_done
    random_number = random.random()

    epsilon_threshold = epsilon_final + (epsilon_initial - epsilon_final)*math.exp(-1.*steps_done/epsilon_decay_rate)

    steps_done += 1

    if random_number > epsilon_threshold:
        with torch.no_grad():  # (!)            
            return policy_net(state).max(1).indices.view(1, 1)  # (?)

    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
# plot setup
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'iframe'

import dash
from dash import Dash, dcc, html, Input, Output

episode_durations = []

app = Dash(__name__)

app.layout = html.Div([
    html.H3("Duration VS Episode"),
    dcc.Graph(id="duration_vs_episode"),
    dcc.Interval(
        id="interval_manager",
        interval=1500,  # [ms] callback interval
        n_intervals=0)])  # interval count

@app.callback(Output(component_id="duration_vs_episode", component_property="figure"),
    Input(component_id="interval_manager", component_property="n_intervals"))
def update_figure(n):
    global episode_durations
    durations_tensor = torch.tensor(episode_durations, dtype=torch.float)
    fig = px.scatter(y=durations_tensor.numpy())

    # take average of m episodes and juxtapose
    m = 20  #100
    if len(durations_tensor) >= m:  
        mean_durations = durations_tensor.unfold(0, m, 1).mean(1).view(-1)
        mean_durations = torch.cat((torch.zeros(m - 1), mean_durations))
        fig_mean = px.line(y=mean_durations.numpy())
        fig_mean.data[0].line.color = "#e02a19"
        fig.add_trace(fig_mean.data[0])

    # add axis labels
    fig.update_layout(
        xaxis_title="<b>episode</b>",
        yaxis_title="<b>duration</b>")
    
    return fig

# method
def optimize_model():
    if len(replay_memory) < replay_memory_batch_size:
        return  # (?)

    transitions = replay_memory.sample(replay_memory_batch_size)
    transition_batch = Transition(*zip(*transitions))  # (?)

    # compute a mask of non-final states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, transition_batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in transition_batch.next_state if s is not None])
    state_batch = torch.cat(transition_batch.state)
    action_batch = torch.cat(transition_batch.action)
    reward_batch = torch.cat(transition_batch.reward)

    # Q(s_{t}, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1})
    next_state_values = torch.zeros(replay_memory_batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = reward_batch + gamma*next_state_values

    # compute Huber loss
    loss_function = nn.SmoothL1Loss()  # (?)
    loss = loss_function(state_action_values, expected_state_action_values.unsqueeze(1))

    # optimize
    optimizer.zero_grad() 
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # in-place gradient clipping
    optimizer.step()

# loop
from itertools import count
import math

if torch.cuda.is_available():
    num_episodes = 200  #600
else:
    num_episodes = 50  #50
print("num_episodes:", num_episodes)

# run live plot app
# if __name__ == '__main__':
#     app.run(debug=True, port=8051)
# print("app running.")

for episode in range(num_episodes):
    observation, observation_info = env.reset()
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    print("episode:", episode)

    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _= env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        # test
        env.render()
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # store transition in replay memory
        replay_memory.push(state, action, reward, next_state)

        # step forward
        state = next_state

        # optimize
        optimize_model()

        # soft update of target network weights (?)
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = tau*policy_net_state_dict[key] + (1 - tau)*target_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

        if done:
            # plot
            episode_durations.append(t + 1)
            break 

print("training: complete.")
