
import datetime
import json
import os
import time
from uuid import uuid4

import gradio as gr
import torch
import yaml
from huggingface_hub import CommitScheduler, hf_hub_download
from omegaconf import OmegaConf

from model.leo_agent import LeoAgentLLM

LOG_DIR = 'logs'
MESH_DIR = 'assets/scene_meshes'
MESH_NAMES = [os.path.splitext(fname)[0] for fname in os.listdir(MESH_DIR)]
ENABLE_BUTTON = gr.update(interactive=True)
DISABLE_BUTTON = gr.update(interactive=False)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ROLE_PROMPT = "You are an AI visual assistant situated in a 3D scene. "\
              "You can perceive (1) an ego-view image (accessible when necessary) and (2) the objects (including yourself) in the scene (always accessible). "\
              "You should properly respond to the USER's instruction according to the given visual information. "
EGOVIEW_PROMPT = "Ego-view image:"
OBJECTS_PROMPT = "Objects (including you) in the scene:"
TASK_PROMPT = "USER: {instruction} ASSISTANT:"
OBJ_FEATS_DIR = 'assets/obj_features'

with open('cfg.yaml') as f:
    cfg = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg)

# build model
agent = LeoAgentLLM(cfg)

# load checkpoint
if cfg.launch_mode == 'hf':
    ckpt_path = hf_hub_download(cfg.hf_ckpt_path[0], cfg.hf_ckpt_path[1])
else:
    ckpt_path = cfg.local_ckpt_path
ckpt = torch.load(ckpt_path, map_location='cpu')
agent.load_state_dict(ckpt, strict=False)
agent.eval()
agent.to(DEVICE)

os.makedirs(LOG_DIR, exist_ok=True)
t = datetime.datetime.now()
log_fname = os.path.join(LOG_DIR, f'{t.year}-{t.month:02d}-{t.day:02d}-{uuid4()}.json')

if cfg.launch_mode == 'hf':
    scheduler = CommitScheduler(
        repo_id=cfg.hf_log_path,
        repo_type='dataset',
        folder_path=LOG_DIR,
        path_in_repo=LOG_DIR,
    )


def change_scene(dropdown_scene: str):
    # reset 3D scene and chatbot history
    return os.path.join(MESH_DIR, f'{dropdown_scene}.glb'), None


def receive_instruction(chatbot: gr.Chatbot, user_chat_input: gr.Textbox):
    # display user input, after submitting user message, before inference
    chatbot.append((user_chat_input, None))
    return (chatbot, gr.update(value=""),) + (DISABLE_BUTTON,) * 5


def generate_response(
        chatbot: gr.Chatbot,
        dropdown_scene: gr.Dropdown,
        dropdown_conversation_mode: gr.Dropdown,
        repetition_penalty: float, length_penalty: float
    ):
    # response starts
    chatbot[-1] = (chatbot[-1][0], "▌")
    yield (chatbot,) + (DISABLE_BUTTON,) * 5

    # create data_dict, batch_size = 1
    data_dict = {
        'prompt_before_obj': [ROLE_PROMPT],
        'prompt_middle_1': [EGOVIEW_PROMPT],
        'prompt_middle_2': [OBJECTS_PROMPT],
        'img_tokens': torch.zeros(1, 1, 4096).float(),
        'img_masks': torch.zeros(1, 1).bool(),
        'anchor_locs': torch.zeros(1, 3).float(),
    }

    # initialize prompt
    prompt = ""
    if 'Multi-round' in dropdown_conversation_mode:
        # multi-round dialogue, with memory
        for (q, a) in chatbot[:-1]:
            prompt += f"USER: {q.strip()} ASSISTANT: {a.strip()}</s>"

    prompt += f"USER: {chatbot[-1][0]} ASSISTANT:"
    data_dict['prompt_after_obj'] = [prompt]

    # anchor orientation
    anchor_orient = torch.zeros(1, 4).float()
    anchor_orient[:, -1] = 1
    data_dict['anchor_orientation'] = anchor_orient

    # load preprocessed scene features
    data_dict.update(torch.load(os.path.join(OBJ_FEATS_DIR, f'{dropdown_scene}.pth'), map_location='cpu'))

    # inference
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(DEVICE)

    output = agent.generate(
        data_dict,
        repetition_penalty=float(repetition_penalty),
        length_penalty=float(length_penalty),
    )
    output = output[0]

    # display response
    for out_len in range(1, len(output)-1):
        chatbot[-1] = (chatbot[-1][0], output[:out_len] + '▌')
        yield (chatbot,) + (DISABLE_BUTTON,) * 5
        time.sleep(0.01)
    
    chatbot[-1] = (chatbot[-1][0], output)
    vote_response(chatbot, 'log', dropdown_scene, dropdown_conversation_mode)
    yield (chatbot,) + (ENABLE_BUTTON,) * 5


def vote_response(
        chatbot: gr.Chatbot, vote_type: str,
        dropdown_scene: gr.Dropdown,
        dropdown_conversation_mode: gr.Dropdown
    ):
    t = datetime.datetime.now()
    this_log = {
        'time': f'{t.hour:02d}:{t.minute:02d}:{t.second:02d}',
        'type': vote_type,
        'scene': dropdown_scene,
        'mode': dropdown_conversation_mode,
        'dialogue': [chatbot[-1]] if 'Single-round' in dropdown_conversation_mode else chatbot,
    }

    if cfg.launch_mode == 'hf':
        with scheduler.lock:   # use scheduler
            if os.path.exists(log_fname):
                with open(log_fname) as f:
                    logs = json.load(f)
                logs.append(this_log)
            else:
                logs = [this_log]
            with open(log_fname, 'w') as f:
                json.dump(logs, f, indent=2)
    else:
        if os.path.exists(log_fname):
            with open(log_fname) as f:
                logs = json.load(f)
            logs.append(this_log)
        else:
            logs = [this_log]
        with open(log_fname, 'w') as f:
            json.dump(logs, f, indent=2)


def upvote_response(
        chatbot: gr.Chatbot,
        dropdown_scene: gr.Dropdown,
        dropdown_conversation_mode: gr.Dropdown
    ):
    vote_response(chatbot, 'upvote', dropdown_scene, dropdown_conversation_mode)
    return ("",) + (DISABLE_BUTTON,) * 3


def downvote_response(
        chatbot: gr.Chatbot,
        dropdown_scene: gr.Dropdown,
        dropdown_conversation_mode: gr.Dropdown
    ):
    vote_response(chatbot, 'downvote', dropdown_scene, dropdown_conversation_mode)
    return ("",) + (DISABLE_BUTTON,) * 3


def flag_response(
        chatbot: gr.Chatbot,
        dropdown_scene: gr.Dropdown,
        dropdown_conversation_mode: gr.Dropdown
    ):
    vote_response(chatbot, 'flag', dropdown_scene, dropdown_conversation_mode)
    return ("",) + (DISABLE_BUTTON,) * 3


def clear_history():
    # reset chatbot history
    return (None, "",) + (DISABLE_BUTTON,) * 4
