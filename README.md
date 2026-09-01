# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the Q-Learning control algorithm using the Gymnasium FrozenLake-v1 environment and train an agent to learn an effective policy for reaching the goal while avoiding hole states.

## Software Requirements

Software Requirements
Python 3.x
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab

## Environment Description
Text
```

FrozenLake-v1 is a reinforcement learning environment represented as a grid. In this experiment, a custom 4×4 FrozenLake environment is used.

The environment contains 16 states and 4 possible actions.

The actions are:

0 → Left
1 → Down
2 → Right
3 → Up

The custom map used is:

F F F F
F S H F
F H G F
F F F F

The state numbering is:

 0   1   2   3
 4   5   6   7
 8   9  10  11
12  13  14  15

The agent receives a reward when it reaches the goal and receives no positive reward for ordinary movements.
```
## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

1. Create the custom FrozenLake-v1 environment with start state 5 and goal state 10.
2. Initialize the Q-table with zeros and set the learning parameters α, γ, ε, ε_min, and ε_decay.
3. Reset the environment and obtain the current state S.
4. Select an action A using the epsilon-greedy policy.
5. Execute action A and observe the reward R and next state S'.
6. Update Q(S,A) using the maximum Q-value of the next state.
7. Set S = S' and repeat the process until the episode terminates.
8. Decrease epsilon using `ε = max(ε_min, ε × ε_decay)`.
9. Repeat the training process for all episodes and store the rewards.
10. Obtain the final Q-table, state-value function, learned policy, and learning curves.


## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

for episode in range(num_episodes):
    state, info = env.reset()

    # FrozenLake starts at S = state 5 in the custom map
    assert state == START_STATE

    epsilon_history.append(epsilon)
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Select action using epsilon-greedy policy
        action = epsilon_greedy_action(state, epsilon)

        # Take action and observe the result
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-Learning uses the maximum Q-value of the next state
        if terminated or truncated:
            td_target = reward
        else:
            td_target = reward + gamma * np.max(Q[next_state])

        # Update Q-value
        Q[state, action] += alpha * (
            td_target - Q[state, action]
        )

        total_reward += reward

        if terminated or truncated:
            break

        state = next_state

    episode_rewards.append(total_reward)

    # Variable epsilon: reduce exploration after each episode
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

print("Training completed.")
print("Initial epsilon:", epsilon_history[0])
print("Final epsilon:", epsilon)






```
---

## Output

Final Q-table:

<img width="266" height="312" alt="image" src="https://github.com/user-attachments/assets/b1175a77-b51b-4183-8123-bda72a122716" />




Estimated State-Value Function:


<img width="282" height="112" alt="image" src="https://github.com/user-attachments/assets/83c8a58f-7a79-450d-a8bf-de61b3af7efd" />




Learned Policy:

<img width="190" height="102" alt="image" src="https://github.com/user-attachments/assets/2c64bbb6-bcba-4913-bfb8-7bef7310c59a" />



Average reward over last 1000 episodes: 
<img width="377" height="35" alt="image" src="https://github.com/user-attachments/assets/8c97d668-ec73-4bf6-86db-36dd37b0a5cd" />

<img width="692" height="465" alt="image" src="https://github.com/user-attachments/assets/320bab22-a6c5-4f6f-8986-0dae607c79ff" />

<img width="690" height="468" alt="image" src="https://github.com/user-attachments/assets/f6c7f308-a366-446a-80b0-27a874ce5fff" />


## Result

The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned an action-value function through repeated interaction with the environment. The learned Q-table was used to obtain the state-value function and learned policy for navigating from the specified starting state to the goal while avoiding holes.

## Inference

The experiment demonstrates that Q-Learning can learn an effective policy using an epsilon-greedy exploration strategy. Initially, the agent explores different actions using a high epsilon value. As epsilon decreases, the agent increasingly exploits actions with higher Q-values. After sufficient training, the Q-table represents the learned action values and can be used to select suitable actions for reaching the goal.

