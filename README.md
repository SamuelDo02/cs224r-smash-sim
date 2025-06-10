# smash-sim
## Creating and emulating Slippi files
### Set up emulator
* Install Slippi Dolphin emulator from [download](https://slippi.gg/downloads)
* Download melee iso from [Google drive link](https://drive.google.com/drive/u/1/folders/1JYTWe0uDXC9w49NOPqWBP2-KFzlJ0Gqj)
    * Request if you don't have access
* Create a Slippi account and log in on the desktop app
* Provide app with downloaded iso file when prompted
    * If you miss this, go to settings and game tab on left sidebar to load iso

### Set up keyboard controls
* Click the gear icon on the top right of the launcher
* Under Settings/Game, press the "Launch Dolphin" button.
* Back on the home screen, press Play
* In Dolphin:
    - Go to `Controllers` on the top right
    - For Port 1, choose Standard Controller, then click Configure
    - In the popup:
        - Set Device to something like `Keyboard/0/Internal Apple Keyboard...`
        - If it doesn't work when you load the game, you might need some trial and error
* Select and launch the game directly in Dolphin

### Get video frames from .slp replay
* The .slp replay should automatically be saved when you finish the match
* Find the folder where your relevant .slp is saved
* Use [this tool](https://github.com/cbartsch/Slippipedia) to extract .slp to .mp4

## Existing datasets
The Slippi machine learning community already has a 200GB dataset of .slp files that we 
can use ([link here](https://drive.google.com/file/d/1ab6ovA46tfiPZ2Y3a_yS1J3k3656yQ8f/view)).
![Slippi ML dataset info](images/slippi_dataset_info.png)

## Setup
### Instance
Use the AWS AMI from HW4: https://us-west-2.console.aws.amazon.com/ec2/home?region=us-west-2#LaunchInstances:ami=ami-0b9cb966e6b0bdcf8

After setting up an instance, make sure to delete the two pre-existing conda environments to make space.

### CS224R-Smash-Sim Repo
```
sudo apt-get install libasound2
sudo apt-get install -y libegl1
conda create -n melee python=3.11
conda activate melee
python -m pip install -r requirements.txt
python -m pip install -e .
pip install "git+https://github.com/vladfi1/libmelee"
```

### Emulator
Get the prebuilt emulator Linux AppImage from vladfi1 here: https://github.com/vladfi1/slippi-Ishiiruka/releases/download/exi-ai-0.1.0/Slippi_Online-x86_64-ExiAI.AppImage 

Run the following to extract the AppImage into binaries in a squashfs-root folder
```
./Slippi_Online-x86_64-ExiAI.AppImage --appimage-extract
```

Get melee.iso and squashfs-root and put them at the root of the directory (i.e. cs224r-smash-sim/)

## Training and Evaluation
The codebase supports three types of agents:
1. Behavior Cloning (BC)
2. Implicit Q-Learning (IQL)
3. Proximal Policy Optimization (PPO)

### Example of running the emulator
Checkout test_melee_env.py

### Training
#### Behavior Cloning
```
python src/scripts/train_melee.py --data_dir=data --exp_name=bc_train --method=bc
```

#### Implicit Q-Learning
```
python src/scripts/train_melee.py --data_dir=data --exp_name=iql_train --method=iql
```

#### Proximal Policy Optimization
```
python src/scripts/train_melee.py --data_dir=data --exp_name=ppo_train --method=ppo \
    --gamma=0.99 \
    --gae_lambda=0.95 \
    --ppo_epochs=10 \
    --clip_ratio=0.2 \
    --value_coef=0.5 \
    --entropy_coef=0.01 \
    --max_steps=1000
```

### Evaluation
#### Behavior Cloning
```
python src/scripts/eval_melee.py --logdir=data/bc_train --exp_name=bc_eval --method=bc
```

#### Implicit Q-Learning
```
python src/scripts/eval_melee.py --logdir=data/iql_train --exp_name=iql_eval --method=iql
```

#### Proximal Policy Optimization
```
python src/scripts/eval_melee.py --logdir=data/ppo_train --exp_name=ppo_eval --method=ppo \
    --gamma=0.99 \
    --gae_lambda=0.95 \
    --ppo_epochs=10 \
    --clip_ratio=0.2 \
    --value_coef=0.5 \
    --entropy_coef=0.01
```

### Common Parameters
- `--policy_type`: Type of policy network ('mlp', 'transformer', 'gpt', 'gpt_ar')
- `--n_layers`: Number of layers in policy network
- `--size`: Size of hidden layers
- `--learning_rate`: Learning rate
- `--batch_size`: Batch size for training
- `--frame_window`: Number of frames to use per observation
- `--n_episodes`: Number of episodes for evaluation
- `--max_steps`: Maximum steps per episode

### PPO-Specific Parameters
- `--gamma`: Discount factor (default: 0.99)
- `--gae_lambda`: GAE lambda parameter (default: 0.95)
- `--ppo_epochs`: Number of PPO epochs (default: 10)
- `--clip_ratio`: PPO clip ratio (default: 0.2)
- `--value_coef`: Value loss coefficient (default: 0.5)
- `--entropy_coef`: Entropy coefficient (default: 0.01)

### Fine-tuning with PPO
You can fine-tune a pre-trained BC or IQL model using PPO by loading the pre-trained weights:
```
python src/scripts/train_melee.py --data_dir=data --exp_name=ppo_finetune --method=ppo \
    --loaddir=data/bc_train \
    --gamma=0.99 \
    --gae_lambda=0.95 \
    --ppo_epochs=10 \
    --clip_ratio=0.2 \
    --value_coef=0.5 \
    --entropy_coef=0.01 \
    --max_steps=1000
```