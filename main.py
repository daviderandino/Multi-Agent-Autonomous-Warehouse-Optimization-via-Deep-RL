import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import random
import os

# ==========================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ==========================================
GRID_SIZE = 5  # 5x5 Grid
NUM_EPISODES = 10
MAX_STEPS = 50
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95  # Gamma: Importance of future rewards
EPSILON_START = 1.0     # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999

# Output directory for the GIF
OUTPUT_DIR = "rl_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. THE ENVIRONMENT (The Warehouse)
# ==========================================
class WarehouseEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        # Agent 1 (Blue) Start & Goal
        self.start_1 = (0, 0)
        self.goal_1 = (4, 4)

        # Agent 2 (Red) Start & Goal
        self.start_2 = (0, 4)
        self.goal_2 = (4, 0)

        self.reset()

    def reset(self):
        """Resets the environment to starting positions."""
        self.agent_1_pos = self.start_1
        self.agent_2_pos = self.start_2
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        State representation: (A1_x, A1_y, A2_x, A2_y)
        We include BOTH agents' positions in the state so they can 'see' each other.
        """
        return (self.agent_1_pos, self.agent_2_pos)

    def step(self, action_1, action_2):
        """
        Moves both agents based on their chosen actions.
        Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        """
        moves = {
            0: (-1, 0), 1: (1, 0),  # Up, Down
            2: (0, -1), 3: (0, 1),  # Left, Right
            4: (0, 0)               # Stay
        }

        # Calculate proposed new positions
        new_pos_1 = self._move(self.agent_1_pos, moves[action_1])
        new_pos_2 = self._move(self.agent_2_pos, moves[action_2])

        reward_1 = -0.1 # Step penalty (encourage speed)
        reward_2 = -0.1

        # --- COLLISION LOGIC (The "Twist") ---
        # 1. Collision with each other (Swapping places or landing on same spot)
        collision = False
        if new_pos_1 == new_pos_2:
            collision = True
        if new_pos_1 == self.agent_2_pos and new_pos_2 == self.agent_1_pos:
            collision = True # Crossed paths directly

        if collision:
            # Heavy penalty for crashing
            reward_1 -= 10
            reward_2 -= 10
            # Agents stay in previous spot (bounce back)
            new_pos_1 = self.agent_1_pos
            new_pos_2 = self.agent_2_pos
        else:
            # Update positions
            self.agent_1_pos = new_pos_1
            self.agent_2_pos = new_pos_2

        # --- GOAL LOGIC ---
        # Check if Agent 1 reached goal
        if self.agent_1_pos == self.goal_1:
            reward_1 += 20 # Big reward

        # Check if Agent 2 reached goal
        if self.agent_2_pos == self.goal_2:
            reward_2 += 20

        # Episode ends if both reach goals or max steps (handled in loop)
        if self.agent_1_pos == self.goal_1 and self.agent_2_pos == self.goal_2:
            self.done = True

        return self.get_state(), (reward_1, reward_2), self.done

    def _move(self, pos, move):
        """Helper to calculate new position checking boundaries."""
        new_r = max(0, min(self.grid_size - 1, pos[0] + move[0]))
        new_c = max(0, min(self.grid_size - 1, pos[1] + move[1]))
        return (new_r, new_c)

# ==========================================
# 3. THE AGENT (Q-Learning)
# ==========================================
class QLearningAgent:
    def __init__(self, action_space_size=5):
        self.q_table = {} # Using a dictionary for sparse state storage
        self.action_space_size = action_space_size
        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START

    def get_q(self, state, action):
        """Helper to get Q-value, defaults to 0.0"""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Epsilon-Greedy Strategy"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space_size - 1) # Explore
        else:
            # Exploit: Choose best action
            q_values = [self.get_q(state, a) for a in range(self.action_space_size)]
            max_q = max(q_values)
            # Handle ties randomly to prevent getting stuck
            actions_with_max_q = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state):
        """Bellman Equation Update"""
        current_q = self.get_q(state, action)

        # Max Q for next state
        next_q_values = [self.get_q(next_state, a) for a in range(self.action_space_size)]
        max_next_q = max(next_q_values)

        # Update Q-value
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    env = WarehouseEnv()
    agent_1 = QLearningAgent()
    agent_2 = QLearningAgent()

    print("🤖 Training Started... (This may take 10-20 seconds)")

    history = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward_1 = 0
        total_reward_2 = 0

        for step in range(MAX_STEPS):
            # 1. Choose Actions
            # Note: Each agent sees the full state (Coordination requires knowing where the other is)
            action_1 = agent_1.choose_action(state)
            action_2 = agent_2.choose_action(state)

            # 2. Act in Environment
            next_state, rewards, done = env.step(action_1, action_2)

            # 3. Learn
            agent_1.learn(state, action_1, rewards[0], next_state)
            agent_2.learn(state, action_2, rewards[1], next_state)

            state = next_state
            total_reward_1 += rewards[0]
            total_reward_2 += rewards[1]

            if done:
                break

        # Decay exploration rate
        agent_1.decay_epsilon()
        agent_2.decay_epsilon()

        if episode % 500 == 0:
            print(f"Episode {episode}: Total Reward ({total_reward_1:.1f}, {total_reward_2:.1f}) | Epsilon: {agent_1.epsilon:.2f}")

    print("✅ Training Complete!")
    return agent_1, agent_2

# ==========================================
# 5. VISUALIZATION & DEMO
# ==========================================
def run_demo(agent_1, agent_2):
    print("🎥 Generating Demo GIF...")
    env = WarehouseEnv()
    state = env.reset()

    # Set epsilon to 0 to observe learned behavior (Pure Exploitation)
    agent_1.epsilon = 0
    agent_2.epsilon = 0

    frames_data = []

    # Run one episode
    for _ in range(MAX_STEPS):
        frames_data.append(state)
        action_1 = agent_1.choose_action(state)
        action_2 = agent_2.choose_action(state)
        state, _, done = env.step(action_1, action_2)
        if done:
            frames_data.append(state) # Capture final state
            break

    # --- MATPLOTLIB ANIMATION ---
    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_grid(state_data):
        ax.clear()
        # Draw Grid Lines
        ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
        ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
        ax.grid(True, color='black', linestyle='-', linewidth=1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.invert_yaxis() # Coordinate system (0,0) top-left

        # Extract positions
        p1, p2 = state_data

        # Draw Goals (Squares)
        # Goal 1 (Blue Target)
        rect_g1 = patches.Rectangle((env.goal_1[1]-0.4, env.goal_1[0]-0.4), 0.8, 0.8, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(rect_g1)
        ax.text(env.goal_1[1], env.goal_1[0], "G1", ha='center', va='center', color='blue')

        # Goal 2 (Red Target)
        rect_g2 = patches.Rectangle((env.goal_2[1]-0.4, env.goal_2[0]-0.4), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax.add_patch(rect_g2)
        ax.text(env.goal_2[1], env.goal_2[0], "G2", ha='center', va='center', color='red')

        # Draw Agents (Circles)
        # Agent 1 (Blue)
        circle_1 = patches.Circle((p1[1], p1[0]), 0.3, color='blue', label='Agent 1')
        ax.add_patch(circle_1)

        # Agent 2 (Red)
        circle_2 = patches.Circle((p2[1], p2[0]), 0.3, color='red', label='Agent 2')
        ax.add_patch(circle_2)

        ax.set_title("Multi-Agent Warehouse: Collision Avoidance")

    # Create Animation
    ani = animation.FuncAnimation(fig, draw_grid, frames=frames_data, interval=300, repeat=False)

    # Save
    save_path = os.path.join(OUTPUT_DIR, "warehouse_agents.gif")
    try:
        ani.save(save_path, writer='pillow')
        print(f"✨ GIF saved successfully: {save_path}")
        print("Open the GIF to see your agents avoiding collision!")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        print("Try installing pillow: pip install pillow")

    plt.close()

if __name__ == "__main__":
    # 1. Train
    trained_a1, trained_a2 = train()

    # 2. Visualize
    run_demo(trained_a1, trained_a2)
