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
* Under Settings/Game, press the “Launch Dolphin” button.
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
### Environment setup
```
conda create -n melee python=3.11
conda activate melee
python -m pip install -r requirements.txt
```