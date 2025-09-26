# Physical Simulation Datasets and Simulators for GateL0RD

This document outlines recommended datasets and simulators for training GateL0RD models on time series prediction tasks with discrete events embedded in continuous data streams.

## Recommended Datasets

### 1. Robot Manipulation Tasks

#### FetchPickAndPlace (Already Available)
- **Description**: Robot arm picking and placing objects with discrete grasping events
- **Data Format**: Observations, actions, and next observations
- **Discrete Events**: Grasping, releasing objects
- **Continuous Stream**: Joint positions, velocities, gripper state
- **Files**: `data/FetchPickAndPlace/`

#### Robot Pushing Tasks
- **Description**: Robot pushing objects on a table
- **Discrete Events**: Contact initiation, object sliding starts/stops
- **Continuous Stream**: Robot and object trajectories
- **Source**: Can be generated using PyBullet or MuJoCo

### 2. Physical Dynamics

#### BilliardBall (Already Available)  
- **Description**: Ball trajectories with collision events
- **Data Format**: Position and velocity over time
- **Discrete Events**: Ball collisions, pocket drops
- **Continuous Stream**: Ball positions and velocities
- **Files**: `data/BilliardBall/`

#### Pendulum Systems
- **Description**: Single or double pendulum dynamics
- **Discrete Events**: Direction changes at extremes, chaos onset
- **Continuous Stream**: Angle and angular velocity

#### Spring-Mass-Damper Systems
- **Description**: Oscillating mass systems
- **Discrete Events**: Impact events, resonance points
- **Continuous Stream**: Position, velocity, acceleration

### 3. Fluid Dynamics

#### Particle Flow Systems
- **Description**: Fluid particles in channels with obstacles
- **Discrete Events**: Particle collisions, turbulence onset
- **Continuous Stream**: Particle positions and velocities

#### Liquid Sloshing
- **Description**: Liquid in moving containers
- **Discrete Events**: Splash events, phase transitions
- **Continuous Stream**: Surface height, velocity fields

### 4. Economic/Financial Time Series

#### Market Crashes
- **Description**: Stock price movements with crash events
- **Discrete Events**: Market crashes, policy announcements
- **Continuous Stream**: Price movements, volume, volatility

### 5. Biological Systems

#### Neural Spike Trains
- **Description**: Neuron firing patterns
- **Discrete Events**: Action potentials (spikes)
- **Continuous Stream**: Membrane potential

#### Heartbeat Monitoring
- **Description**: ECG signals with arrhythmia events
- **Discrete Events**: Irregular heartbeats, arrhythmias
- **Continuous Stream**: ECG waveform

## Recommended Simulators

### 1. Physics Simulators

#### PyBullet (Recommended)
```python
# Installation
pip install pybullet

# Example: Robot manipulation
import pybullet as p
import pybullet_data

# Setup
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)

# Load robot and objects
robot_id = p.loadURDF("franka_panda/panda.urdf")
object_id = p.loadURDF("cube_small.urdf", [0.5, 0, 0.5])

# Run simulation and collect data
for i in range(1000):
    p.stepSimulation()
    # Collect positions, velocities, contacts
```

**Advantages**:
- Easy to use Python interface
- Built-in collision detection
- Robot models available
- Good for manipulation tasks

#### MuJoCo
```python
# Installation (requires license for full version)
pip install mujoco-py

# Example usage
import mujoco_py

# Load model and create simulation
model = mujoco_py.load_model_from_path("robot_model.xml")
sim = mujoco_py.MjSim(model)

# Run simulation
for i in range(1000):
    sim.step()
    # Collect data
```

**Advantages**:
- High-fidelity physics
- Excellent for robotics research
- Continuous dynamics with contact events

#### Gazebo (for more complex scenarios)
- ROS integration
- Realistic sensor simulation
- Multi-robot systems

### 2. Simple Physics Simulators (Custom Implementation)

#### Bouncing Ball Simulator
```python
import numpy as np

class BouncingBallSimulator:
    def __init__(self, gravity=9.81, bounce_damping=0.8):
        self.gravity = gravity
        self.bounce_damping = bounce_damping
        
    def simulate(self, initial_pos, initial_vel, timesteps=1000, dt=0.01):
        positions = []
        velocities = []
        collisions = []
        
        pos = initial_pos.copy()
        vel = initial_vel.copy()
        
        for t in range(timesteps):
            # Physics update
            vel[1] -= self.gravity * dt  # gravity in y-direction
            pos += vel * dt
            
            # Collision detection (ground at y=0)
            collision = False
            if pos[1] <= 0 and vel[1] < 0:
                pos[1] = 0
                vel[1] = -vel[1] * self.bounce_damping
                collision = True
                
            positions.append(pos.copy())
            velocities.append(vel.copy())
            collisions.append(collision)
            
        return np.array(positions), np.array(velocities), collisions
```

#### Pendulum Simulator
```python
import numpy as np

class PendulumSimulator:
    def __init__(self, length=1.0, gravity=9.81, damping=0.1):
        self.length = length
        self.gravity = gravity  
        self.damping = damping
        
    def simulate(self, initial_angle, initial_velocity, timesteps=1000, dt=0.01):
        angles = []
        velocities = []
        direction_changes = []
        
        angle = initial_angle
        vel = initial_velocity
        prev_vel = vel
        
        for t in range(timesteps):
            # Pendulum equation: d²θ/dt² = -(g/L)sin(θ) - damping*dθ/dt
            acc = -(self.gravity / self.length) * np.sin(angle) - self.damping * vel
            vel += acc * dt
            angle += vel * dt
            
            # Detect direction changes
            direction_change = (prev_vel * vel) < 0
            
            angles.append(angle)
            velocities.append(vel)
            direction_changes.append(direction_change)
            
            prev_vel = vel
            
        return np.array(angles), np.array(velocities), direction_changes
```

### 3. Specialized Simulators

#### OpenAI Gym Environments
```python
import gym

# Various environments with discrete events
env = gym.make('FetchPickAndPlace-v1')  # Robot manipulation
env = gym.make('HalfCheetah-v3')       # Locomotion with gait events
env = gym.make('Pendulum-v0')          # Pendulum dynamics
```

#### DeepMind Control Suite
```python
from dm_control import suite

# High-quality physics simulations
env = suite.load(domain_name="manipulator", task_name="bring_ball")
env = suite.load(domain_name="pendulum", task_name="swingup")
```

## Data Generation Pipeline

### Example: Generate Robot Manipulation Dataset
```python
def generate_robot_manipulation_data(n_episodes=100, episode_length=200):
    import pybullet as p
    import numpy as np
    
    # Setup simulation
    p.connect(p.DIRECT)  # No GUI for faster generation
    p.setGravity(0, 0, -10)
    
    dataset = []
    
    for episode in range(n_episodes):
        # Reset environment
        robot_id = p.loadURDF("franka_panda/panda.urdf")
        object_id = p.loadURDF("cube.urdf", [0.5, 0, 0.5])
        
        episode_data = []
        
        for step in range(episode_length):
            # Random action
            action = np.random.uniform(-1, 1, 7)  # 7-DOF robot
            
            # Apply action
            p.setJointMotorControlArray(robot_id, range(7), p.POSITION_CONTROL, action)
            p.stepSimulation()
            
            # Collect observations
            robot_state = p.getJointStates(robot_id, range(7))
            object_pos, object_orn = p.getBasePositionAndOrientation(object_id)
            contacts = p.getContactPoints(robot_id, object_id)
            
            # Create observation vector
            obs = []
            obs.extend([state[0] for state in robot_state])  # Joint positions
            obs.extend([state[1] for state in robot_state])  # Joint velocities  
            obs.extend(object_pos)                           # Object position
            obs.append(len(contacts) > 0)                    # Contact flag (discrete event)
            
            episode_data.append(obs)
            
        dataset.append(np.array(episode_data))
        
    return np.concatenate(dataset, axis=0)

# Generate and save data
data = generate_robot_manipulation_data()
np.save('robot_manipulation_dataset.npy', data)
```

## Dataset Characteristics for GateL0RD

For optimal GateL0RD training, datasets should have:

1. **Continuous Variables**: Position, velocity, angles, forces
2. **Discrete Events**: Collisions, grasps, phase transitions
3. **Temporal Dependencies**: Events that affect future behavior
4. **Variable Event Frequency**: Both frequent and rare events
5. **Multi-dimensional**: Multiple interacting variables

## Implementation Notes

- Use the `TimeSeriesDataset` class for loading any of these datasets
- Ensure proper normalization for stable training
- Consider data augmentation (noise, time warping) for robustness
- Balance continuous dynamics with discrete event frequency
- Create validation sets with different dynamics/parameters than training