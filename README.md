# RL Without Forgetting

This repository contains the implementation of **Dynamic Policy Anchoring (DPA)**, an approach extending Proximal Policy Optimization (PPO) to address catastrophic forgetting in reinforcement learning. The project explores methods to retain performance across sequential tasks, inspired by ideas from continual learning.

For more detailed information, please refer to the **`report.pdf`** in the repository.

---

## Setup

### 1. Clone the repository
```
git clone <repository_url>
cd <repository_name>
```

### 2. Install dependencies
Use the `requirements.txt` file to install all necessary Python packages:
```
pip install -r requirements.txt
```

---

## Training

### 1. Create a configuration file
To start training, create a JSON configuration file in the `params/` folder. Use the examples in this folder as references. Ensure your file is named using the experiment ID (e.g., `const_lambda_1.json`).

### 2. Run the training script
```
python3 main.py --expt_id=<experiment_id>
```
For example, if your configuration file is named `const_lambda_1.json`, run:
```
python3 main.py --expt_id=const_lambda_1
```

---

## Project Structure

- **`params/`**: Contains JSON configuration files for different experiments.
- **`xml_files/`**: Contains config information for the environment.
- **`env.py`**: Includes the dynamically changing environment.
- **`ppo.py`**: Main implementation of PPO with the DPA extension.
- **`requirements.txt`**: List of required Python packages for the project.
- **`report.pdf`**: Detailed report on the approach, experiments, and results. Please refer to it for more in-depth information.

---

## Experiments

The experiments focus on alternating tasks using the MuJoCo `HalfCheetahVanilla` and `HalfCheetahBigLeg` environments. Results are tracked using reward per episode over 20 million timesteps. Task changes are hardcoded since automated task detection is not currently functional.
