import os

import gradio as gr

from utils import *


with gr.Blocks(title='LEO Demo') as demo:
    gr.HTML(value="<h1 align='center'>An Embodied Generalist Agent in 3D World</h1>")
    gr.HTML(value="<div align='center' style='margin-top:-1em; margin-bottom:-1em;'><img src='/file=assets/leo.svg' width='4%'></div>")
    # gr.HTML(value="<img src='/file=assets/teaser.png' alt='Teaser' width='760px' style='display: block; margin: auto;'>")
    gr.HTML(value="<p align='center' style='font-size: 1.2em; color: #485fc7;'><a href='https://arxiv.org/abs/2311.12871' target='_blank'>arXiv</a> | <a href='https://embodied-generalist.github.io/' target='_blank'>Project Page</a> | <a href='https://github.com/embodied-generalist/embodied-generalist' target='_blank'>Code</a></p>")
    gr.HTML(value="<p align='center' style='font-size: 1.15em;'><i>LEO: an embodied generalist agent capable of perceiving, grounding, reasoning, planning, and acting in 3D world.</i></p>")

    with gr.Row():
        with gr.Column(scale=5):
            dropdown_scene = gr.Dropdown(
                choices=MESH_NAMES,
                value=MESH_NAMES[0],
                interactive=True,
                label='Select a 3D scene',
            )
            model_3d = gr.Model3D(
                value=os.path.join(MESH_DIR, f'{MESH_NAMES[0]}.glb'),
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label='3D Scene',
                camera_position=(90, 30, 10),
                height=659,
            )
            gr.HTML(
                """<center><strong>
                👆 SCROLL and DRAG on the 3D Scene
                to zoom in/out and rotate. Press CTRL and DRAG to pan.
                </strong></center>
                """
            )
        with gr.Column(scale=5):
            dropdown_conversation_mode = gr.Dropdown(
                choices=['Single-round mode', 'Multi-round mode'],
                value='Single-round mode',
                interactive=True,
                label='Select conversation mode',
            )
            chatbot = gr.Chatbot(label='Chat with LEO')
            with gr.Row():
                with gr.Column(scale=8):
                    user_chat_input = gr.Textbox(
                        placeholder="Enter text here to chat with LEO",
                        show_label=False,
                        autofocus=True,
                    )
                with gr.Column(scale=2, min_width=0):
                    send_button = gr.Button('Send', variant='primary', scale=2)
            with gr.Row():
                upvote_button = gr.Button(value='👍 Upvote', interactive=False)
                downvote_button = gr.Button(value='👎 Downvote', interactive=False)
                flag_button = gr.Button(value='⚠️ Flag', interactive=False)
                clear_button = gr.Button(value='🗑️ Clear', interactive=False)
            with gr.Row():
                with gr.Accordion(label="Examples for user instruction:", open=True):
                    gr.Examples(
                        examples=[
                            ["How many armchairs are there in this room?"],
                            ["Is there a radio in the room?"],
                            ["Where is the wardrobe located?TODO"],
                            ["What is the shape of the shelf in front of the picture?TODO"],
                            ["Plan for the task: Tidy up and arrange the nursery room.TODO"],
                       ],
                        inputs=user_chat_input,
                    )

    # generation_config
    with gr.Accordion('Parameters', open=False):
        repetition_penalty = gr.Slider(
            minimum=0.0,
            maximum=10.0,
            value=3.0,
            step=1.0,
            interactive=True,
            label='Repetition penalty',
        )
        length_penalty = gr.Slider(
            minimum=0.0,
            maximum=10.0,
            value=1.0,
            step=1.0,
            interactive=True,
            label="Length penalty",
        )
    gr.Markdown("### Terms of Service")
    gr.HTML(
        """By using this service, users are required to agree to the following terms:
           the service is a research preview intended for non-commercial use only
           and may collect user dialogue data for future research."""
    )
    gr.Markdown("### Acknowledgment")
    gr.HTML(
        """Template adapted from <a href="https://llava.hliu.cc/">LLaVA</a> and
           <a href="http://sled-whistler.eecs.umich.edu:7777/">LLM-Grounder</a>."""
    )

    # Event handling
    button_list = [upvote_button, downvote_button, flag_button, clear_button]

    dropdown_scene.change(
        fn=change_scene,
        inputs=[dropdown_scene],
        outputs=[model_3d, chatbot],
        queue=False,
    )

    dropdown_conversation_mode.change(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot, user_chat_input] + button_list,
        queue=False,
    )

    user_chat_input.submit(
        fn=receive_instruction,
        inputs=[chatbot, user_chat_input],
        outputs=[chatbot, user_chat_input, send_button] + button_list,
        queue=False,
    ).then(
        fn=generate_response,
        inputs=[
            chatbot,
            dropdown_scene,
            dropdown_conversation_mode,
            repetition_penalty,
            length_penalty,
        ],
        outputs=[chatbot, send_button] + button_list,
        scroll_to_output=True,
    )

    send_button.click(
        fn=receive_instruction,
        inputs=[chatbot, user_chat_input],
        outputs=[chatbot, user_chat_input, send_button] + button_list,
        queue=False,
    ).then(
        fn=generate_response,
        inputs=[
            chatbot,
            dropdown_scene,
            dropdown_conversation_mode,
            repetition_penalty,
            length_penalty,
        ],
        outputs=[chatbot, send_button] + button_list,
        scroll_to_output=True,
    )

    upvote_button.click(
        upvote_response,
        [chatbot, dropdown_scene, dropdown_conversation_mode],
        [user_chat_input, upvote_button, downvote_button, flag_button],
        queue=False,
    )
    downvote_button.click(
        downvote_response,
        [chatbot, dropdown_scene, dropdown_conversation_mode],
        [user_chat_input, upvote_button, downvote_button, flag_button],
        queue=False,
    )
    flag_button.click(
        flag_response,
        [chatbot, dropdown_scene, dropdown_conversation_mode],
        [user_chat_input, upvote_button, downvote_button, flag_button],
        queue=False,
    )
    clear_button.click(
        fn=clear_history,
        inputs=[],
        outputs=[chatbot, user_chat_input] + button_list,
        queue=False,
    )


demo.queue().launch(share=True, allowed_paths=['assets'])
