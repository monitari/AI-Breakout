# AI Breakout Game with Reinforcement Learning

A modern implementation of the classic Breakout game featuring AI integration using Deep Q-Learning (DQN), complete with a 3-life system and progressive difficulty levels.

![Game Screenshot](https://github.com/user-attachments/assets/7c30a081-645c-45c7-b60b-91e377ff8379)

## Key Features

🎮 **Gameplay Essentials**
- Classic Breakout mechanics with modern physics
- 3-life system with visual indicators
- Progressive difficulty levels
- Dynamic score tracking
- Responsive paddle controls

🤖 **AI Integration**
- Deep Q-Network (DQN) reinforcement learning
- Adaptive difficulty scaling
- State normalization for efficient learning
- Experience replay buffer
- Epsilon-greedy exploration strategy

🛠 **Technical Highlights**
- Configurable parameters via centralized Config class
- Smooth paddle-ball interaction mechanics
- Precision collision detection
- Real-time performance metrics
- Dual rendering modes (training/playing)

## Installation

**Requirements:**
- Python 3.8+
- Pygame 2.1+
- PyTorch 2.0+

```bash
# Clone repository
git clone https://github.com/monitari/AI-Breakout.git

# Install dependencies
pip install pygame torch