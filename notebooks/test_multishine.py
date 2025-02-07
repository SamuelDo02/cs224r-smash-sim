# %%
from pathlib import Path

import torch
from streaming import StreamingDataset
from tensordict import TensorDict

from hal.constants import ACTION_BY_IDX
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.io import load_model_from_artifact_dir
from hal.training.streaming_dataset import HALStreamingDataset

torch.set_printoptions(threshold=torch.inf)
# %%
ACTION_BY_IDX

# %%
artifact_dir = Path("/opt/projects/hal2/runs/2025-02-04_13-53-10/arch@GPTv1-4-4_local_batch_size@32_n_samples@262144/")
model, config = load_model_from_artifact_dir(artifact_dir)

# %%
x = TensorDict.load(artifact_dir / "training_samples/32/inputs")
y = TensorDict.load(artifact_dir / "training_samples/32/targets")

# %%
y_hat = model(x)

# %%
y_hat["buttons"][0]
# %%
y_hat["buttons"][0].argmax(dim=-1)
# %%
y["buttons"][0].argmax(dim=-1)

# %%
# Load closed loop replay and run it through the model
replay_dir = Path("/opt/projects/hal2/data/multishine_eval/test")

data_config = DataConfig(
    data_dir="/opt/projects/hal2/data/multishine_eval",
    seq_len=28800,
)
test_dataset = HALStreamingDataset(
    local=str(replay_dir),
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data_config,
    embedding_config=config.embedding,
)

# %%
test_dataset[0]

# %%
x = test_dataset[0]["inputs"][:256].unsqueeze(0)
y_hat = model(x)

# %%
x["ego_action"][0]
# %%
predicted_buttons = y_hat["buttons"][0].argmax(dim=-1)
predicted_buttons
# %%
y_hat["buttons"][0][:64]
# %%
actual_buttons = test_dataset[0]["targets"]["buttons"][:256].argmax(dim=-1)
actual_buttons
# %%
predicted_main_stick = y_hat["main_stick"][0].argmax(dim=-1)
predicted_main_stick
# %%
actual_main_stick = test_dataset[0]["targets"]["main_stick"][:256].argmax(dim=-1)
actual_main_stick
# %%
test_replay_dir = Path("/opt/projects/hal2/data/multishine_eval_argmax/test")

data_config = DataConfig(
    data_dir="/opt/projects/hal2/data/multishine_eval_argmax",
    seq_len=28800,
)
test_dataset = HALStreamingDataset(
    local=str(test_replay_dir),
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data_config,
    embedding_config=config.embedding,
)
# %%
x = test_dataset[0]["inputs"][:256].unsqueeze(0)
x["ego_action"]
# %%
y_hat = model(x)
# %%
predicted_buttons = y_hat["buttons"][0].argmax(dim=-1)
predicted_buttons
# %%
actual_buttons = test_dataset[0]["targets"]["buttons"][:256].argmax(dim=-1)
actual_buttons
# %%
predicted_main_stick = y_hat["main_stick"][0].argmax(dim=-1)
predicted_main_stick
# %%
actual_main_stick = test_dataset[0]["targets"]["main_stick"][:256].argmax(dim=-1)
actual_main_stick
# %%
model

# %%
mds_dir = Path("/opt/projects/hal2/data/multishine/test")
data_config = DataConfig(
    data_dir="/opt/projects/hal2/data/multishine",
    seq_len=28800,
)
train_dataset = HALStreamingDataset(
    local=str(mds_dir),
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data_config,
    embedding_config=config.embedding,
)
# %%
x_train = train_dataset[0]["inputs"][:256].unsqueeze(0)
y_train = train_dataset[0]["targets"][:256].unsqueeze(0)
# %%
actions = x_train["ego_action"][0].tolist()
buttons = y_train["buttons"][0].argmax(dim=-1).tolist()
for i, (action, button) in enumerate(zip(actions, buttons)):
    print(f"{i:03d}: {action:03d} -> {button:03d}")
# %%
for i, button in enumerate():
    print(i, button)
# %%
x_train = train_dataset[0]["inputs"][:256].unsqueeze(0)
x_train["ego_action"]
# %%
y_hat = model(x_train)
y_hat["buttons"][0].argmax(dim=-1)
# %%
(x_train["ego_action"] == x["ego_action"]).sum()
# %%
for k, tensor in (x_train == x).items():
    print(k, tensor)

# %%
artifact_dir = Path("/opt/projects/hal2/runs/2025-02-05_13-12-15/arch@GPTv1-4-4_local_batch_size@32_n_samples@262144/")
model, config = load_model_from_artifact_dir(artifact_dir)
# %%
x_train = train_dataset[0]["inputs"][:256].unsqueeze(0)
x_train["ego_action"]
# torch.argwhere(x_train["ego_action"] == 42)
# %%
y_hat = model(x_train)
torch.argwhere(y_hat["buttons"][0].argmax(dim=-1) == 1)
# %%
torch.argwhere(y_hat["main_stick"][0].argmax(dim=-1) == 3)
# %%
x_test = test_dataset[0]["inputs"][:256].unsqueeze(0)
# %%
torch.argwhere(torch.all(x_test["gamestate"] == x_train["gamestate"], dim=-1) == False)
# %%
raw_test_dataset = StreamingDataset(local=str(test_replay_dir), remote=None)
# %%
raw_test_dataset[0]
# %%
raw_test_dataset[0]["p1_action"][:256]
# %%
torch.argwhere(x_train["ego_action"] == 14)
# %%
torch.argwhere(torch.tensor(raw_test_dataset[0]["p1_action"][:256] == 14))

# %%
mds_dir = Path("/opt/projects/hal2/data/multishine/test")
data_config = DataConfig(
    data_dir="/opt/projects/hal2/data/multishine",
    seq_len=256,
)
embedding_config = EmbeddingConfig(
    input_preprocessing_fn="inputs_v1",
)
train_dataset = HALStreamingDataset(
    local=str(mds_dir),
    remote=None,
    batch_size=1,
    shuffle=False,
    data_config=data_config,
    embedding_config=embedding_config,
)
# %%
train_dataset.preprocessor.frame_offsets_by_feature
# %%
x_train = train_dataset[0]["inputs"][:256].unsqueeze(0)
y_train = train_dataset[0]["targets"][:256].unsqueeze(0)
# %%
prev_buttons = x_train["controller"][0, :, -6:].argmax(dim=-1)
torch.argwhere(prev_buttons != 5)
# %%
buttons = y_train["buttons"][0].argmax(dim=-1)
torch.argwhere(buttons != 5)
# %%
prev_main_stick = x_train["controller"][0, :, :21].argmax(dim=-1)
torch.argwhere(prev_main_stick != 0)
# %%
main_stick = y_train["main_stick"][0].argmax(dim=-1)
torch.argwhere(main_stick != 0)
# %%
prev_buttons = x_train["gamestate"][0, :-48].tolist()
buttons = y_train["buttons"][0].argmax(dim=-1).tolist()
for i, (prev_button, button) in enumerate(zip(prev_buttons, buttons)):
    print(f"{i:03d}: {prev_button:03d} -> {button:03d}")
