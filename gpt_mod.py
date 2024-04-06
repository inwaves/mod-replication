from transformers import GPT2PreTrainedModel, GPT2Attention, GPT2FlashAttention2
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch

GPT2_ATTENTION_CLASSES = {
    "eager": GPT2Attention,
    "flash_attention_2": GPT2FlashAttention2,
}

class GPT2BlockMixtureOfDepths(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        self.capacity = config.capacity
        attention_class = GPT2_ATTENTION_CLASSES[config._attn_implementation]

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = attention_class(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.router_weights = nn.Linear(hidden_size, hidden_size, bias=False)

        if config.add_cross_attention:
            self.crossattention = attention_class(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        """Modified GPT2Block class which implements the mixture-of-depths trick.

        First, learned router weights are multiplied by the incoming hidden states. Then, we select
        the top-k largest from these outputs, where k is the specified model capacity. k is always 
        some percentage of the total sequence length – its intuition is "number of tokens that can
        participate in this block".

        We then select the tokens corresponding to the k largest outputs; only these tokens go through
        attention and MLP. The other, unselected tokens are rejoined to the updated tokens at the end
        of the forward function. The effect is that compute expended by the block is reduced, since
        not all tokens participate."""
        
        hidden_state_copies = hidden_states.copy()
        router_outputs = self.router_weights(hidden_states)
        topk_router_outputs, topk_router_indices = t.topk(router_outputs, self.capacity, 1)
        assert topk_router_indices.shape[1] == self.capacity

        selected_hidden_states = t.gather(hidden_state_copies, 1, topk_router_indices)
        hidden_states = self.ln_1(selected_hidden_states)
        
        attn_outputs = self.attn(
            selected_hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + selected_hidden_states

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection

            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)

        # Hidden states multiplied by router values so gradient updates router weights, plus residual.
        hidden_states = residual + feed_forward_hidden_states * topk_router_values

        # Updated selected states are rejoined with the non-selected states.
        all_hidden_states = t.scatter(hidden_state_copies, 1, topk_router_indices, selected_hidden_states)

        if use_cache:
            outputs = (all_hidden_states,) + outputs
        else:
            outputs = (all_hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

class GPT2PreTrainedModelMixtureOfDepths(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(self.embed_im, eps=config.layer_norm_epsilon)
        
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation
        
        self.post_init()

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     token_type_ids: Optional[torch.LongTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     encoder_hidden_states: Optional[torch.Tensor] = None,
    #     encoder_attention_mask: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

    #     # These are just booleans that control whether we return the attentions and hidden states.
    #     output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    #     output_hidden_states = (
    #         output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    #     )
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     if input_ids is not None and inputs_embeds is not None:
    #         raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    #     elif input_ids is not None:
    #         self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
    #         input_shape = input_ids.size()
    #         input_ids = input_ids.view(-1, input_shape[-1])
    #         batch_size = input_ids.shape[0]
    #     elif inputs_embeds is not None:
    #         input_shape = inputs_embeds.size()[:-1]
    #         batch_size = inputs_embeds.shape[0]
    #     else:
    #         raise ValueError("You have to specify either input_ids or inputs_embeds")

    #     device = input_ids.device if input_ids is not None else inputs_embeds.device

    #     if token_type_ids is not None:
    #         token_type_ids = token_type_ids.view(-1, input_shape[-1])

    #     if past_key_values is None:
    #         past_length = 0
    #         past_key_values = tuple([None] * len(self.blocks))
    #     else:
    #         past_length = past_key_values[0][0].size(-2)
    #     if position_ids is None:
    #         position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    #         position_ids = position_ids.unsqueeze(0)

    #     # Attention mask.
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.view(batch_size, -1)
    #         if self._attn_implementation == "flash_attention_2":
    #             attention_mask = attention_mask if 0 in attention_mask else None
    #         else:
    #             # We create a 3D attention mask from a 2D tensor mask.
    #             # Sizes are [batch_size, 1, 1, to_seq_length]
    #             # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    #             # this attention mask is more simple than the triangular masking of causal attention
    #             # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    #             attention_mask = attention_mask[:, None, None, :]

    #             # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    #             # masked positions, this operation will create a tensor which is 0.0 for
    #             # positions we want to attend and the dtype's smallest value for masked positions.
    #             # Since we are adding it to the raw scores before the softmax, this is
    #             # effectively the same as removing these entirely.
    #             attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    #             attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    #     # If a 2D or 3D attention mask is provided for the cross-attention
    #     # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    #     if self.config.add_cross_attention and encoder_hidden_states is not None:
    #         encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
    #         encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    #         if encoder_attention_mask is None:
    #             encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
    #         if self._attn_implementation != "flash_attention_2":
    #             encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    #     else:
    #         encoder_attention_mask = None

    #     # Prepare head mask if needed
    #     # 1.0 in head_mask indicate we keep the head
    #     # attention_probs has shape bsz x n_heads x N x N
    #     # head_mask has shape n_layer x batch x n_heads x N x N
    #     head_mask = self.get_head_mask(head_mask, self.config.n_layer)

    #     # Generate the embeddings if we don't have them cached already.
    #     if inputs_embeds is None:
    #         inputs_embeds = self.wte(input_ids)
    #     position_embeds = self.wpe(position_ids)
    #     hidden_states = inputs_embeds + position_embeds

    #     if token_type_ids is not None:
    #         token_type_embeds = self.wte(token_type_ids)
    #         hidden_states = hidden_states + token_type_embeds

    #     hidden_states = self.drop(hidden_states)

    #     output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

    #     if self.gradient_checkpointing and self.training:
    #         if use_cache:
    #             logger.warning_once(
    #                 "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #             )
    #             use_cache = False

    #     presents = () if use_cache else None

    #     # Prepare these collections so we can report attention/hidden states if asked to do so.
    #     all_self_attentions = () if output_attentions else None
    #     all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    #     all_hidden_states = () if output_hidden_states else None
    #     for i, (block, layer_past) in enumerate(zip(self.blocks, past_key_values)):
    #         # Model parallel
    #         if self.model_parallel:
    #             torch.cuda.set_device(hidden_states.device)
    #             # Ensure layer_past is on same device as hidden_states (might not be correct)
    #             if layer_past is not None:
    #                 layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
    #             # Ensure that attention_mask is always on the same device as hidden_states
    #             if attention_mask is not None:
    #                 attention_mask = attention_mask.to(hidden_states.device)
    #             if isinstance(head_mask, torch.Tensor):
    #                 head_mask = head_mask.to(hidden_states.device)
    #         if output_hidden_states:
    #             all_hidden_states = all_hidden_states + (hidden_states,)

    #         if self.gradient_checkpointing and self.training:
    #             outputs = self._gradient_checkpointing_func(
    #                 block.__call__,
    #                 hidden_states,
    #                 None,
    #                 attention_mask,
    #                 head_mask[i],
    #                 encoder_hidden_states,
    #                 encoder_attention_mask,
    #                 use_cache,
    #                 output_attentions,
    #             )
    #         else:
    #             outputs = block(
    #                 hidden_states,
    #                 layer_past=layer_past,
    #                 attention_mask=attention_mask,
    #                 head_mask=head_mask[i],
    #                 encoder_hidden_states=encoder_hidden_states,
    #                 encoder_attention_mask=encoder_attention_mask,
    #                 use_cache=use_cache,
    #                 output_attentions=output_attentions,
    #             )

    #         hidden_states = outputs[0]
    #         if use_cache is True:
    #             presents = presents + (outputs[1],)

    #         if output_attentions:
    #             all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
    #             if self.config.add_cross_attention:
    #                 all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

    #         # Model Parallel: If it's the last layer for that device, put things on the next device
    #         if self.model_parallel:
    #             for k, v in self.device_map.items():
    #                 if i == v[-1] and "cuda:" + str(k) != self.last_device:
    #                     hidden_states = hidden_states.to("cuda:" + str(k + 1))

    #     hidden_states = self.ln_f(hidden_states)

    #     hidden_states = hidden_states.view(output_shape)
    #     # Add last hidden state
    #     if output_hidden_states:
    #         all_hidden_states = all_hidden_states + (hidden_states,)

    #     if not return_dict:
    #         return tuple(
    #             v
    #             for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
    #             if v is not None
    #         )

    #     return BaseModelOutputWithPastAndCrossAttentions(
    #         last_hidden_state=hidden_states,
    #         past_key_values=presents,
    #         hidden_states=all_hidden_states,
    #         attentions=all_self_attentions,
    #         cross_attentions=all_cross_attentions,
    #     )