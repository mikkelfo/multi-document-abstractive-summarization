import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Tuple, Union

import types

def prophetnet_fixes(model):
    # Encoder
    for layer in model.prophetnet.encoder.layers:
        layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)
    model.prophetnet.encoder.forward = types.MethodType(encoder_forward, model.prophetnet.encoder)

    # Decoder
    for layer in model.prophetnet.decoder.layers:
        layer.self_attn.forward = types.MethodType(ngram_attn_forward, layer.self_attn)
        layer.self_attn.get_predict_relative_pos_embeddings = types.MethodType(custom_get_predict_relative_pos_embeddings, layer.self_attn)
        layer.self_attn.get_main_relative_pos_embeddings = types.MethodType(custom_get_main_relative_pos_embeddings, layer.self_attn)
    model.prophetnet.decoder.forward = types.MethodType(decoder_forward, model.prophetnet.decoder)
    model.prophetnet.decoder.prepare_predict_attention_mask = types.MethodType(custom_prepare_predict_attention_mask, model.prophetnet.decoder)
    model.prophetnet.decoder.prepare_attention_mask = types.MethodType(custom_prepare_attention_mask, model.prophetnet.decoder)

def attn_forward(
    self,
    hidden_states,
    key_value_states: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    layer_head_mask: Optional[Tensor] = None,
    past_key_value: Optional[Tuple[Tensor]] = None,
    output_attentions: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:

    batch_size, tgt_len, hidden_size = hidden_states.size()

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    assert list(hidden_states.size()) == [
        batch_size,
        tgt_len,
        hidden_size,
    ], f"Size of hidden states should be {batch_size, tgt_len, hidden_size}, but is {hidden_states.size()}"

    # previous time steps are cached - no need to recompute key and value if they are static
    query_states = self.query_proj(hidden_states) / (self.head_dim**0.5)

    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_states = past_key_value[0]
        value_states = past_key_value[1]
    elif is_cross_attention:
        # cross_attentions
        key_states = self._shape(self.key_proj(key_value_states), -1, batch_size)
        value_states = self._shape(self.value_proj(key_value_states), -1, batch_size)
    else:
        # self_attention
        key_states = self._shape(self.key_proj(hidden_states), -1, batch_size)
        value_states = self._shape(self.value_proj(hidden_states), -1, batch_size)

    if is_cross_attention:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_states, value_states)

    # project states into the correct shape
    proj_shape = (batch_size, self.num_attn_heads, -1, self.head_dim)   # CHANGED (batch_size * self.num) to (batch_size, self.num)
    query_states = self._shape(query_states, tgt_len, batch_size).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(2)    # CHANGED .size(1) to .size(2)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))   # CHANGED (bmm to matmul) and transpose(1,2) to tranpose(2,3)
    assert attn_weights.size() == (
        batch_size, 
        self.num_attn_heads,
        tgt_len,
        src_len,
    ), f"`attn_weights` should be of size {batch_size, self.num_attn_heads, tgt_len, src_len}, but is of size {attn_weights.shape}"

    # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
    if attention_mask is not None and attention_mask.dim() == 0:
        attention_mask = None
    assert attention_mask is None or attention_mask.size() == (
        batch_size,
        self.num_attn_heads,
        1,
        src_len,
    ), f"`attention_mask` should be `None` or of shape attention_mask.size() == {batch_size, self.num_attn_heads, 1, src_len}, but is {attention_mask.shape}"

    if attention_mask is not None:  # don't attend to padding symbols
        attn_weights = attn_weights + attention_mask

    if output_attentions:
        # this operation is a bit awkward, but it's required to
        # make sure that attn_weights keeps its gradient.
        # In order to do so, attn_weights have to be reshaped
        # twice and have to be reused in the following
        attn_weights_reshaped = attn_weights.view(batch_size, self.num_attn_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(batch_size, self.num_attn_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        assert layer_head_mask.size() == (
            self.num_attn_heads,
        ), f"Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is {layer_head_mask.size()}"
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
            batch_size, self.num_attn_heads, tgt_len, src_len
        )
        attn_weights = attn_weights.view(batch_size, self.num_attn_heads, tgt_len, src_len)

        # apply head_mask also on attn_weights_reshaped which is used for n-gram attention inside the model
        attn_weights_reshaped = layer_head_mask.view(1, -1, 1, 1) * attn_weights_reshaped

    attn_probs = nn.functional.dropout(
        attn_weights,
        p=self.attention_dropout,
        training=self.training,
    )

    attn_output = torch.matmul(attn_probs, value_states)    # CHANGED (bmm to matmul)
    assert attn_output.size() == (
        batch_size, 
        self.num_attn_heads,
        tgt_len,
        self.head_dim,
    ), f"`attn_output` should be of shape {batch_size, self.num_attn_heads, tgt_len, self.head_dim}, but is of shape {attn_output.size()}"

    attn_output = (
        attn_output.view(batch_size, self.num_attn_heads, tgt_len, self.head_dim)
        .transpose(1, 2)
        .reshape(batch_size, tgt_len, hidden_size)
    )

    attn_output = self.out_proj(attn_output)

    attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
    return attn_output, attn_weights_reshaped, past_key_value

from transformers.modeling_outputs import BaseModelOutput
def encoder_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutput]:
    r"""
    Returns:
    Example:
    ```python
    >>> from transformers import ProphetNetTokenizer, ProphetNetEncoder
    >>> import torch
    >>> tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
    >>> model = ProphetNetEncoder.from_pretrained("patrickvonplaten/prophetnet-large-uncased-standalone")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds has to be passed.")
    elif input_ids is not None and inputs_embeds is not None:
        raise ValueError("Make sure to only pass input_ids or inputs_embeds.")
    elif input_ids is not None and inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    # prepare attention mask
    if attention_mask is not None:
        extended_attention_mask = (
            1.0 - attention_mask[:, None, None, :].repeat(1, self.config.num_encoder_attention_heads, 1, 1)
        ) * -10000.0
        extended_attention_mask = extended_attention_mask.to(inputs_embeds.dtype)
    else:
        extended_attention_mask = None

    position_embeddings, position_ids = self.position_embeddings(inputs_embeds.shape[:2], inputs_embeds.device)

    hidden_states = inputs_embeds + position_embeddings
    hidden_states = self.embeddings_layer_norm(hidden_states)
    hidden_states = nn.functional.dropout(hidden_states, p=self.config.dropout, training=self.training)

    encoder_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None

    # check if head_mask has a correct number of layers specified if desired
    if head_mask is not None:
        assert head_mask.size()[0] == (
            len(self.layers)
        ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    for idx, encoder_layer in enumerate(self.layers):
        if output_hidden_states:
            encoder_hidden_states = encoder_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(encoder_layer),
                hidden_states,
                extended_attention_mask,
                (head_mask[idx] if head_mask is not None else None),
            )
        else:
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                output_attentions=output_attentions,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    if output_hidden_states:
        encoder_hidden_states = encoder_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, encoder_hidden_states, all_attentions] if v is not None)
    return BaseModelOutput(
        last_hidden_state=hidden_states, hidden_states=encoder_hidden_states, attentions=all_attentions
    )

from transformers.models.prophetnet.modeling_prophetnet import ProphetNetDecoderModelOutput, ngram_attention_bias, softmax
def decoder_forward(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, ProphetNetDecoderModelOutput]:
    r"""
    encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
        the model is configured as a decoder.
    encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
        the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
    cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
        Mask to nullify selected heads of the cross-attention modules. Mask values selected in `[0, 1]`:
        - 1 indicates the head is **not masked**,
        - 0 indicates the head is **masked**.
    past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.
        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    Returns:
    Example:
    ```python
    >>> from transformers import ProphetNetTokenizer, ProphetNetDecoder
    >>> import torch
    >>> tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased")
    >>> model = ProphetNetDecoder.from_pretrained("microsoft/prophetnet-large-uncased", add_cross_attention=False)
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```"""
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either `decoder_input_ids` or `decoder_inputs_embeds` has to be passed.")
    elif input_ids is not None and inputs_embeds is not None:
        raise ValueError("Make sure to only pass `decoder_input_ids` or `decoder_inputs_embeds`.")
    elif input_ids is not None and inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids)

    batch_size, sequence_length = inputs_embeds.shape[:2]

    main_stream_pos_embed, position_ids = self.position_embeddings(
        (batch_size, sequence_length),
        device=inputs_embeds.device,
        past_key_values=past_key_values,
    )

    if past_key_values is not None:
        main_relative_position_buckets, predict_relative_position_buckets = None, None
    else:
        (
            main_relative_position_buckets,
            predict_relative_position_buckets,
        ) = self.compute_buffered_relative_buckets(position_ids)
    predicting_stream_pos_embed = self.position_embeddings._forward(position_ids + 1)

    # add position embeddings
    hidden_states = inputs_embeds + main_stream_pos_embed

    ngram_embeddings = self.ngram_embeddings.weight

    # prepare attention mask
    if past_key_values is not None:
        assert (
            hidden_states.size(1) == 1
        ), "At the moment `use_cache` is only supported for `decoder_input_ids` of length 1"

        ngram_hidden_states = [
            (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed).repeat(batch_size, 1, 1)
            for ngram in range(self.ngram)
        ]
        extended_attention_mask = None
        extended_predict_attention_mask = None
    else:
        ngram_hidden_states = [
            (ngram_embeddings[ngram - 1] + predicting_stream_pos_embed) for ngram in range(self.ngram)
        ]
        extended_attention_mask = self.prepare_attention_mask(hidden_states, attention_mask)
        extended_predict_attention_mask = self.prepare_predict_attention_mask(hidden_states, attention_mask)

    # prepare encoder attention mask
    if encoder_attention_mask is not None:
        extended_encoder_attention_mask = (
            1.0 - encoder_attention_mask[:, None, None, :].repeat(1, self.config.num_decoder_attention_heads, 1, 1)    # Changed
        ) * -10000.0
        extended_encoder_attention_mask = extended_encoder_attention_mask.to(inputs_embeds.dtype)
    else:
        extended_encoder_attention_mask = None

    hidden_states = torch.cat([hidden_states] + ngram_hidden_states, 1)

    if self.embeddings_layer_norm:
        hidden_states = self.embeddings_layer_norm(hidden_states)

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # init attentions, hidden_states and cache with empty tuples
    all_main_stream_hidden_states = () if output_hidden_states else None
    all_ngram_stream_hidden_states = () if output_hidden_states and self.config.ngram > 0 else None

    all_main_stream_attns = () if output_attentions else None
    all_ngram_stream_attns = () if output_attentions else None
    all_cross_attns = () if output_attentions and self.config.add_cross_attention else None
    present_key_values = () if use_cache else None

    # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
        if attn_mask is not None:
            assert attn_mask.size()[0] == (
                len(self.layers)
            ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            # grad cannot be kept because tensor is sliced
            all_main_stream_hidden_states += (hidden_states[:, :sequence_length],)
            if self.config.ngram > 0:
                all_ngram_stream_hidden_states += (hidden_states[:, sequence_length:],)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                extended_attention_mask,
                encoder_hidden_states,
                extended_encoder_attention_mask,
                (head_mask[idx] if head_mask is not None else None),
                (cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                extended_predict_attention_mask,
                main_relative_position_buckets,
                predict_relative_position_buckets,
                position_ids,
                None,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attn_mask=extended_encoder_attention_mask,
                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                cross_attn_layer_head_mask=(
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                ),
                extended_predict_attention_mask=extended_predict_attention_mask,
                main_relative_position_buckets=main_relative_position_buckets,
                predict_relative_position_buckets=predict_relative_position_buckets,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        
        hidden_states = layer_outputs[0]

        if use_cache:
            present_key_values += (layer_outputs[4 if output_attentions else 1],)

        if output_attentions:
            all_main_stream_attns += (layer_outputs[1],)
            all_ngram_stream_attns += (layer_outputs[2],)

            if self.config.add_cross_attention:
                all_cross_attns += (layer_outputs[3],)

    if output_hidden_states:
        all_main_stream_hidden_states += (hidden_states[:, :sequence_length],)
        if self.config.ngram > 0:
            all_ngram_stream_hidden_states += (hidden_states[:, sequence_length:],)

    # split last_hidden_state for return
    last_hidden_state = hidden_states[:, :sequence_length]
    last_hidden_state_ngram = hidden_states[:, sequence_length:] if self.config.ngram > 0 else None

    if not return_dict:
        return tuple(
            v
            for v in [
                last_hidden_state,
                last_hidden_state_ngram,
                present_key_values,
                all_main_stream_hidden_states,
                all_ngram_stream_hidden_states,
                all_main_stream_attns,
                all_ngram_stream_attns,
                all_cross_attns,
            ]
            if v is not None
        )
    return ProphetNetDecoderModelOutput(
        last_hidden_state=last_hidden_state,
        last_hidden_state_ngram=last_hidden_state_ngram,
        past_key_values=present_key_values,
        hidden_states=all_main_stream_hidden_states,
        hidden_states_ngram=all_ngram_stream_hidden_states,
        attentions=all_main_stream_attns,
        ngram_attentions=all_ngram_stream_attns,
        cross_attentions=all_cross_attns,
    )

def ngram_attn_forward(
    self,
    hidden_states,
    past_key_value: Optional[Tuple[Tensor]] = None,
    attention_mask=None,
    layer_head_mask=None,
    extended_predict_attention_mask=None,
    main_relative_position_buckets=None,
    predict_relative_position_buckets=None,
    position_ids=None,
):
    batch_size, ngram_sequence_length, hidden_size = hidden_states.size()

    assert list(hidden_states.size()) == [
        batch_size,
        ngram_sequence_length,
        hidden_size,
    ], f"`hidden_states` should be of shape {batch_size, ngram_sequence_length, hidden_size}, but is of shape {hidden_states.shape}"

    # project
    query_states = self.query_proj(hidden_states)
    key_states = self.key_proj(hidden_states)
    value_states = self.value_proj(hidden_states)

    # normalize
    query_states = query_states / (self.head_dim**0.5)

    # reshape
    query_states = self._shape(query_states, ngram_sequence_length, batch_size)
    key_states = self._shape(key_states, -1, batch_size)
    value_states = self._shape(value_states, -1, batch_size)

    proj_shape = (batch_size, self.num_attn_heads, -1, self.head_dim)   # Changed (B*Head to B,head)

    query_states = query_states.view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    # chunk into main stream and predict stream
    hidden_states_list = hidden_states.chunk(1 + self.ngram, dim=1)

    query_states_list = query_states.chunk(1 + self.ngram, dim=2)   # Changed 1 to 2
    key_states_list = key_states.chunk(1 + self.ngram, dim=2)       # Changed 1 to 2
    value_states_list = value_states.chunk(1 + self.ngram, dim=2)   # Changed 1 to 2

    main_hidden_states, hidden_states_predict_list = hidden_states_list[0], hidden_states_list[1:]
    main_query_states, predict_query_states_list = query_states_list[0], query_states_list[1:]
    main_key_states, predict_key_states_list = key_states_list[0], key_states_list[1:]
    main_value_states, predict_value_states_list = value_states_list[0], value_states_list[1:]

    # saved states are stored with shape (batch_size, num_attn_heads, seq_len, head_dim)
    if past_key_value is not None:
        prev_main_key_states = past_key_value[0].view(batch_size, self.num_attn_heads, -1, self.head_dim)
        main_key_states = torch.cat((prev_main_key_states, main_key_states), dim=1)
        prev_main_value_states = past_key_value[1].view(batch_size, self.num_attn_heads, -1, self.head_dim)
        main_value_states = torch.cat((prev_main_value_states, main_value_states), dim=1)

    # Update cache
    past_key_value = (
        main_key_states,#.view(batch_size, self.num_attn_heads, -1, self.head_dim),     # Redundant
        main_value_states#.view(batch_size, self.num_attn_heads, -1, self.head_dim),    # Redundant
    )

    # get seq_length of main stream only
    sequence_length = ngram_sequence_length // (1 + self.ngram)

    # MAIN-STREAM
    # main attn weights
    main_attn_weights = torch.matmul(main_query_states, main_key_states.transpose(2, 3))   # Changed to matmul

    # retrieve relative position embeddings for each layer -> see paper for more details
    
    main_relative_pos_embeddings = self.get_main_relative_pos_embeddings(
        main_hidden_states, main_attn_weights, position_ids, main_relative_position_buckets
    )
    
    main_attn_weights = main_attn_weights + main_relative_pos_embeddings

    if attention_mask is not None:
        main_attn_weights = main_attn_weights + attention_mask

    main_attn_probs = softmax(
        main_attn_weights,
        dim=-1,
        onnx_trace=self.onnx_trace,
    ).type_as(main_attn_weights)

    if layer_head_mask is not None:
        assert layer_head_mask.size() == (
            self.num_attn_heads,
        ), f"Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is {layer_head_mask.size()}"
        main_attn_probs = layer_head_mask.view(1, -1, 1, 1) * main_attn_probs.view(
            batch_size, self.num_attn_heads, -1, sequence_length
        )
        main_attn_probs = main_attn_probs.view(batch_size, self.num_attn_heads, -1, sequence_length)

    main_attn_probs = nn.functional.dropout(main_attn_probs, p=self.attention_dropout, training=self.training)
    # project to attn_output
    main_attn_output = torch.matmul(main_attn_probs, main_value_states)     # Changed to matmul

    # reshape so that num_heads dim is merged into last `head_dim` axis
    main_attn_output = (
        main_attn_output#.view(batch_size, self.num_attn_heads, sequence_length, self.head_dim)     # Redundant
        .transpose(1, 2)
        .reshape(batch_size, 1, sequence_length, hidden_size)
    )
    main_attn_output = self.out_proj(main_attn_output)

    # PREDICT-STREAM
    # [ngram, B*head, T, c]
    predict_query_states = torch.cat(predict_query_states_list, 0).view(
        self.ngram, batch_size, self.num_attn_heads, sequence_length, self.head_dim                      # Changed
    )
    # [ngram, B*head, 2*T, c]
    predict_key_states = torch.cat(
        [torch.cat([main_key_states, key], 2).unsqueeze(0) for key in predict_key_states_list], 0       # Changed dim 1 to 2 for cat
    )

    # [ngram, T, B, C]
    predict_hidden_states = torch.cat(hidden_states_predict_list, 0).view(
        self.ngram, batch_size, sequence_length, hidden_size            # Changed ordering
    ).transpose(1, 2)       # Added

    # [ngram, B*head, 2*T, c]
    predict_value_states = torch.cat(
        [torch.cat([main_value_states, v_p], 2).unsqueeze(0) for v_p in predict_value_states_list], 0   # Changed dim 1 to 2 for cat
    )
    # [ngram, B*head, T, 2*T]
    predict_attn_weights = torch.einsum("nbhtc,nbhsc->nbhts", (predict_query_states, predict_key_states))   # Added h for num_heads

    # [ngram, B*head, T, S]
    # retrieve relative position embeddings for each layer -> see paper for more details

    predict_relative_pos_embeddings = self.get_predict_relative_pos_embeddings(
        predict_hidden_states, predict_attn_weights, position_ids, predict_relative_position_buckets
    )

    # [ngram, B*head, T, 2*T]
    predict_attn_weights = predict_attn_weights + predict_relative_pos_embeddings

    if extended_predict_attention_mask is not None:
        predict_attn_weights = predict_attn_weights + extended_predict_attention_mask.to(
            predict_attn_weights.dtype
        )

    predict_attn_probs = softmax(
        predict_attn_weights,
        dim=-1,
        onnx_trace=self.onnx_trace,
    ).type_as(predict_attn_weights)

    if layer_head_mask is not None:
        assert layer_head_mask.size() == (
            self.num_attn_heads,
        ), f"Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is {layer_head_mask.size()}"
        predict_attn_probs = layer_head_mask.view(1, 1, -1, 1, 1) * predict_attn_probs.view(
            self.ngram, batch_size, self.num_attn_heads, sequence_length, 2 * sequence_length
        )
        predict_attn_probs = predict_attn_probs.view(
            self.ngram, batch_size, self.num_attn_heads, sequence_length, 2 * sequence_length
        )

    predict_attn_probs = nn.functional.dropout(
        predict_attn_probs, p=self.attention_dropout, training=self.training
    )
    # project to attention output
    # [ngram, B*head, T, c]
    predict_attn_output = torch.einsum("nbhts,nbhsc->nbhtc", (predict_attn_probs, predict_value_states))    # Added h

    # reshape so that num_heads dim is merged into last `head_dim` axis
    # [ngram, B, T, C]
    predict_attn_output = (
        predict_attn_output#.view(self.ngram, batch_size, self.num_attn_heads, sequence_length, self.head_dim)  # Redundant
        .permute(1, 0, 3, 2, 4)
        .reshape(batch_size, self.ngram, sequence_length, hidden_size)
    )
    predict_attn_output = self.out_proj(predict_attn_output)

    # concat to single attn output
    # [B, 1+ngram*T, C]
    attn_output = torch.cat([main_attn_output, predict_attn_output], 1).view(batch_size, -1, hidden_size)
    # reshape into better form for `config.output_attentions`
    main_attn_probs = main_attn_probs.view(batch_size, self.num_attn_heads, sequence_length, -1)
    predict_attn_probs = predict_attn_probs.view(
        self.ngram, batch_size, self.num_attn_heads, sequence_length, -1
    ).transpose(0, 1)

    attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)

    return attn_output, main_attn_probs, predict_attn_probs, past_key_value


def custom_get_main_relative_pos_embeddings(
    self, hidden_states, attn_weights, position_ids, main_relative_position_buckets
):
    # input hidden_states [B,T,C], input attn_weights [T*head,T,S], input position_ids [B,T] or [1,1]

    if main_relative_position_buckets is None:
        batch_size, sequence_length = hidden_states.shape[:2]
        relative_positions = (
            torch.arange(1, attn_weights.shape[-1] + 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, sequence_length, 1)
            .to(position_ids.device)
        )
        relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(
            batch_size, sequence_length, 1
        )  # [B, T, s]
        main_relative_position_buckets = compute_relative_buckets(
            self.num_buckets, self.relative_max_distance, relative_positions, False
        )
    
    rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)  # [B,T,Buckets*head]
    
    rel_pos_embeddings = rel_pos_embeddings.view(
        rel_pos_embeddings.shape[:2] + (self.num_buckets, self.num_attn_heads)
    ).permute(
        0, 3, 1, 2
    )  # [B,T,Buckets,head]
    rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:3] + (-1,))  # [B*head,T,Buckets]   # Changed [:2] to [:3]

    main_relative_position_buckets = (
        main_relative_position_buckets.repeat(1, self.num_attn_heads, 1)
        .view(-1, main_relative_position_buckets.shape[-1])
        .long()
    )  # [B*head*T, T]
    rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))  # [B*head*T,Buckets]

    main_relative_pos_embeddings = torch.gather(
        rel_pos_embeddings, dim=1, index=main_relative_position_buckets
    ).view(attn_weights.shape[:3] + (-1,))      # Changed [:2] to [:3]

    return main_relative_pos_embeddings

def custom_get_predict_relative_pos_embeddings(
    self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets
):
    # input hidden_states [ngram, T,B,C], input attn_weights [ngram, B*head,T,S], input position_ids [B,T] or [1,1], input predict_relative_position_buckets [B,T, 2*T] or None
    sequence_length, batch_size = hidden_states.shape[1:3]

    if predict_relative_position_buckets is None:
        key_sequence_length = attn_weights.shape[-1]
        assert (
            position_ids[0][0] == key_sequence_length - 1
        ), "`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)"
        relative_positions = (
            torch.arange(0, key_sequence_length)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, sequence_length, 1)
            .to(position_ids.device)
        )

        relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
        predict_relative_position_buckets = compute_relative_buckets(
            self.num_buckets, self.relative_max_distance, relative_positions, False
        )
    
    hidden_states = hidden_states.transpose(1, 2)  # [ngram, B, T, C]
    rel_pos_embeddings = self.relative_pos_embeddings(hidden_states).view(
        hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads)
    )  # [ngram, B, T, bucket, head]

    rel_pos_embeddings = rel_pos_embeddings.permute(0, 1, 4, 2, 3).reshape(
        self.ngram * batch_size * self.num_attn_heads, sequence_length, -1
    )  # [ngram*B*head, T, bucket]

    predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0).repeat(
        self.ngram, 1, self.num_attn_heads, 1
    )  # [ngram, B, head*T, S]

    rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.size(-1))
    predict_relative_position_buckets = predict_relative_position_buckets.view(
        -1, predict_relative_position_buckets.size(-1)
    ).long()  # [ngram*B*head*T, S]

    predict_relative_pos_embeddings = torch.gather(
        rel_pos_embeddings, dim=1, index=predict_relative_position_buckets
    ).view(
        self.ngram, batch_size, self.num_attn_heads, sequence_length, -1
    )  # [ngram, B, head, T, S]

    return predict_relative_pos_embeddings


def custom_prepare_predict_attention_mask(self, hidden_states, attention_mask):
    batch_size, seq_length = hidden_states.shape[:2]

    # get causal mask
    predict_causal_mask = ngram_attention_bias(
        self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype
    )
    predict_causal_mask = torch.cat(
        [
            predict_causal_mask[:, :seq_length, :seq_length],
            predict_causal_mask[
                :, :seq_length, self.max_target_positions : self.max_target_positions + seq_length
            ],
        ],
        dim=-1,
    )
    
    extended_predict_causal_mask = predict_causal_mask[:, None, :, :].expand(
        predict_causal_mask.shape[:1] + (batch_size,) + predict_causal_mask.shape[1:]
    )

    # add usual attention mask
    if attention_mask is not None:
        extended_attention_mask = (1.0 - attention_mask[None, :, None, :]) * -10000.0
        extended_attention_mask = extended_attention_mask.expand((self.ngram, batch_size, seq_length, seq_length))
        # predicted stream attention_mask should always be 0
        extended_attention_mask = torch.cat(
            [extended_attention_mask, torch.zeros_like(extended_attention_mask)], dim=-1
        )
        extended_predict_attention_mask = extended_predict_causal_mask + extended_attention_mask
    else:
        extended_predict_attention_mask = extended_predict_causal_mask[:, :, None, : :]     # Changed
    return extended_predict_attention_mask.repeat(1, 1, self.config.num_decoder_attention_heads, 1, 1).to(      # Changed
        hidden_states.dtype
    )

def custom_prepare_attention_mask(self, hidden_states, attention_mask):
    batch_size, seq_length = hidden_states.shape[:2]

    # get causal mask
    causal_mask = torch.full(
        (seq_length, seq_length), -float("inf"), dtype=hidden_states.dtype, device=hidden_states.device
    )
    causal_mask = torch.triu(causal_mask, 1)
    extended_causal_mask = causal_mask[:seq_length, :seq_length][None, :, :].expand(
        (batch_size,) + causal_mask.shape
    )

    # add usual attention mask
    if attention_mask is not None:
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0   # Changed
        extended_attention_mask = extended_causal_mask + extended_attention_mask
    else:
        extended_attention_mask = extended_causal_mask[:, None, :, :]                   # Changed
    return extended_attention_mask.repeat(1, self.config.num_decoder_attention_heads, 1, 1).to(hidden_states.dtype) # Changed


if __name__ == '__main__':
    from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer

    model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased')
    tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')

    input_string = ['Hello, my dog is cute', 'I have a cat that is not cute']
    labels = ['My dog is cute', 'My cat is not cute']

    inputs = tokenizer(input_string, return_tensors="pt", padding=True, truncation=True)
    targets = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)

    # Inverse the ordering of the input and labels using [::-1]
    inputs_inv = tokenizer(input_string[::-1], return_tensors="pt", padding=True, truncation=True)
    targets_inv = tokenizer(labels[::-1], return_tensors="pt", padding=True, truncation=True)

    import types
    from prophetnet_fixes import attn_forward, encoder_forward
    for layer in model.prophetnet.encoder.layers:
        layer.self_attn.forward = types.MethodType(attn_forward, layer.self_attn)

    model.prophetnet.encoder.forward = types.MethodType(encoder_forward, model.prophetnet.encoder)

    import types
    from prophetnet_fixes import decoder_forward, ngram_attn_forward, custom_get_main_relative_pos_embeddings, custom_get_predict_relative_pos_embeddings, custom_prepare_predict_attention_mask, custom_prepare_attention_mask

    for layer in model.prophetnet.decoder.layers:
        layer.self_attn.forward = types.MethodType(ngram_attn_forward, layer.self_attn)
        layer.self_attn.get_predict_relative_pos_embeddings = types.MethodType(custom_get_predict_relative_pos_embeddings, layer.self_attn)
        layer.self_attn.get_main_relative_pos_embeddings = types.MethodType(custom_get_main_relative_pos_embeddings, layer.self_attn)

    model.prophetnet.decoder.forward = types.MethodType(decoder_forward, model.prophetnet.decoder)
    model.prophetnet.decoder.prepare_predict_attention_mask = types.MethodType(custom_prepare_predict_attention_mask, model.prophetnet.decoder)
    model.prophetnet.decoder.prepare_attention_mask = types.MethodType(custom_prepare_attention_mask, model.prophetnet.decoder)

    output = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
    output_inv = model(input_ids=inputs_inv.input_ids, attention_mask=inputs_inv.attention_mask, labels=targets_inv.input_ids)
    print(output.loss.item(), output_inv.loss.item())