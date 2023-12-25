import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import get_peft_model, LoraConfig
from transformers import LlamaForCausalLM, LlamaTokenizer


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LeoAgentLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if hasattr(cfg, 'model'):
            cfg = cfg.model

        # LLM
        if cfg.llm.use_ckpt == 'hf':
            llm_cfg_path = snapshot_download(cfg.llm.hf_cfg_path)
        else:
            llm_cfg_path = cfg.llm.local_cfg_path
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_cfg_path, use_fast=False,
                                                            truncation_side=cfg.llm.truncation_side)
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_model = LlamaForCausalLM.from_pretrained(llm_cfg_path, torch_dtype=torch.float16)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for param in self.llm_model.parameters():
            param.requires_grad = False
        self.llm_model.eval()
        self.llm_model.train = disabled_train

        # LoRA-based LLM fine-tuning
        if cfg.llm.lora.flag:
            lora_config = LoraConfig(
                r=cfg.llm.lora.rank,
                lora_alpha=cfg.llm.lora.alpha,
                target_modules=cfg.llm.lora.target_modules,
                lora_dropout=cfg.llm.lora.dropout,
                bias='none',
                modules_to_save=[],
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config=lora_config)

        self.max_context_len = cfg.llm.max_context_len

    @property
    def device(self):
        return list(self.parameters())[0].device

    def build_right_justified_sequence(self, data_dict):
        """
        Concat six sequences: `prompt_before_obj`, `prompt_middle_1`, `img_tokens`, `prompt_middle_2`, `obj_tokens`, `prompt_after_obj`.
        Return right justified sequence for causal LM: <pad>, <role/situation>, <img>, <objs>, <instruction>.
        """
        bs = len(data_dict['prompt_before_obj'])

        self.llm_tokenizer.padding_side = 'left'
        text_input_tokens_pre = self.llm_tokenizer(
            data_dict['prompt_before_obj'],
            return_tensors='pt',
            padding='longest'
        ).to(self.device)   # [PAD, BOS, tokens], (B, T1)

        text_input_tokens_mid1 = self.llm_tokenizer(
            data_dict['prompt_middle_1'],
            return_tensors='pt',
            padding='longest'
        ).to(self.device)

        img_tokens = data_dict['img_tokens'].to(self.device)
        img_masks = data_dict['img_masks'].to(self.device)
        img_masks = img_masks.reshape(-1, 1).repeat(1, img_tokens.size(1))

        text_input_tokens_mid2 = self.llm_tokenizer(
            data_dict['prompt_middle_2'],
            return_tensors='pt',
            padding='longest'
        ).to(self.device)

        obj_tokens = data_dict['obj_tokens'].to(self.device)
        obj_masks = data_dict['obj_masks'].to(self.device)

        self.llm_tokenizer.padding_side = 'right'   # no need to be 'left', as padding tokens will be shifted
        self.llm_tokenizer.truncation_side = 'left'   # truncate history
        text_input_tokens_post = self.llm_tokenizer(
            data_dict['prompt_after_obj'],
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_context_len,
        ).to(self.device)   # [BOS, tokens, PAD], (B, T3)

        # hardcode, remove bos, make "tokenize subseq and concat" equivalent to "tokenize the whole seq"
        assert text_input_tokens_mid1.attention_mask.all() and text_input_tokens_mid2.attention_mask.all(), \
               "prompt_middle should be the same and thus no padding"

        text_input_tokens_mid1.input_ids = text_input_tokens_mid1.input_ids[:, 1:]
        text_input_tokens_mid1.attention_mask = text_input_tokens_mid1.attention_mask[:, 1:]
        for i in range(bs):
            if not img_masks[i].any():
                # no image input, also mask the text prompt for image tokens
                text_input_tokens_mid1.attention_mask[i].fill_(0)

        text_input_tokens_mid2.input_ids[:, 0] = 869   # 1 (bos) -> 869 (▁.)
        text_input_tokens_post.input_ids[:, 0] = 869   # 1 (bos) -> 869 (▁.)

        inputs_embeds_pre = self.llm_model.get_input_embeddings()(text_input_tokens_pre.input_ids)
        inputs_embeds_mid1 = self.llm_model.get_input_embeddings()(text_input_tokens_mid1.input_ids)
        inputs_embeds_mid2 = self.llm_model.get_input_embeddings()(text_input_tokens_mid2.input_ids)
        inputs_embeds_post = self.llm_model.get_input_embeddings()(text_input_tokens_post.input_ids)

        # since img_tokens, prompt_mid, obj_tokens are fixed length without padding, we concat them first
        inputs_embeds_mid = torch.cat([inputs_embeds_mid1, img_tokens, inputs_embeds_mid2, obj_tokens], dim=1)
        attn_mask_mid = torch.cat([
            text_input_tokens_mid1.attention_mask, img_masks,
            text_input_tokens_mid2.attention_mask, obj_masks
        ], dim=1)

        post_pad_length = torch.logical_not(text_input_tokens_post.attention_mask).sum(-1)

        bs, l1, hidden_dim = inputs_embeds_pre.shape
        _, l2, _ = inputs_embeds_mid.shape
        _, l3, _ = inputs_embeds_post.shape

        inputs_embeds = torch.zeros(
            bs, l1+l2+l3, hidden_dim
        ).type(inputs_embeds_pre.dtype).to(self.device)

        attention_mask = torch.zeros(
            bs, l1+l2+l3
        ).type(obj_masks.dtype).to(self.device)

        # assign by chunks
        for i in range(bs):
            post_pad_len = post_pad_length[i]

            if post_pad_len > 0:
                inputs_embeds[i, :post_pad_len] = inputs_embeds_post[i, -post_pad_len:]
                attention_mask[i, :post_pad_len] = 0
                inputs_embeds[i, post_pad_len+l1+l2:] = inputs_embeds_post[i, :-post_pad_len]
                attention_mask[i, post_pad_len+l1+l2:] = 1
            else:
                # no padding
                inputs_embeds[i, -l3:] = inputs_embeds_post[i]
                attention_mask[i, -l3:] = 1

            inputs_embeds[i, post_pad_len: post_pad_len+l1] = inputs_embeds_pre[i]
            attention_mask[i, post_pad_len: post_pad_len+l1] = text_input_tokens_pre.attention_mask[i]

            inputs_embeds[i, post_pad_len+l1: post_pad_len+l1+l2] = inputs_embeds_mid[i]
            attention_mask[i, post_pad_len+l1: post_pad_len+l1+l2] = attn_mask_mid[i]

        return inputs_embeds, attention_mask

    @torch.no_grad()
    def generate(
        self,
        data_dict,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        repetition_penalty=3.0,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        assert 'img_tokens' in data_dict and 'obj_tokens' in data_dict, "Visual features should have been processed offline."

        inputs_embeds, attention_mask = self.build_right_justified_sequence(data_dict=data_dict)
        bs = inputs_embeds.shape[0]

        # give bos token as condition
        bos_tokens = self.llm_tokenizer(
            [self.llm_tokenizer.bos_token] * bs,
            return_tensors='pt',
        ).to(self.device)
        bos_tokens_ids = bos_tokens.input_ids[:, 0:1]   # (B, 1)
        bos_tokens_attn = bos_tokens.attention_mask[:, 0:1]   # (B, 1)

        # prepare a `bos_token`
        bos_embeds = self.llm_model.get_input_embeddings()(bos_tokens_ids)   # (B, 1, D)
        inputs_embeds = torch.cat([inputs_embeds, bos_embeds], dim=1)   # (B, T1+O+T2+1, D)
        attention_mask = torch.cat([attention_mask, bos_tokens_attn], dim=1)   # (B, T1+O+T2+1)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        outputs[outputs == 0] = 2   # convert output id 0 (unk_token) to 2 (eos_token)

        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text
