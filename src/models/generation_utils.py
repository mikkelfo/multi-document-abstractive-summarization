import torch
import types

def prep_inputs_for_generation(self, bos_token_id, encoder_outputs):
    length = encoder_outputs.last_hidden_state.shape[1]
    return torch.ones(1, length, dtype=torch.long, device=self.device) * -100

def expanded_inputs(input_ids: torch.LongTensor, expand_size: int = 1, is_encoder_decoder: bool = False, attention_mask = None, encoder_outputs = None, **model_kwargs):
    expanded_return_idx = (torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device))
    input_ids = input_ids.index_select(0, expanded_return_idx)
    expanded_encoder_idx = (torch.arange(encoder_outputs.last_hidden_state.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(encoder_outputs.last_hidden_state.device))
    encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(0, expanded_encoder_idx.to(encoder_outputs.last_hidden_state.device))
    model_kwargs["encoder_outputs"] = encoder_outputs

    if attention_mask is not None:
        model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_encoder_idx)

    return input_ids, model_kwargs

def serial_forward(self,hidden_states,attention_mask=None,encoder_hidden_states=None,encoder_attn_mask=None,layer_head_mask=None,cross_attn_layer_head_mask=None,extended_predict_attention_mask=None,main_relative_position_buckets=None,predict_relative_position_buckets=None,position_ids=None,past_key_value=None,use_cache: bool = True,output_attentions: bool = False,):
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    ngram_attention_output, self_attn_weights, self_attn_weights_ngram, present_key_value = self.self_attn(hidden_states=hidden_states,past_key_value=self_attn_past_key_value,attention_mask=attention_mask,layer_head_mask=layer_head_mask,extended_predict_attention_mask=extended_predict_attention_mask,main_relative_position_buckets=main_relative_position_buckets,predict_relative_position_buckets=predict_relative_position_buckets,position_ids=position_ids,)
    hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

    cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
    cross_attn_weights = None
    num_beams = hidden_states.shape[0]
    batch_size = encoder_hidden_states.shape[0] // num_beams
    if encoder_hidden_states is not None:
        # SERIAL INPUT
        for i in range(batch_size):
            if len(encoder_attn_mask.shape) == 4:   # With ProphetNet fix
                attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states[i*num_beams:(i+1)*num_beams],
                    attention_mask=encoder_attn_mask[(i*num_beams):(i+1)*num_beams],
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
            elif len(encoder_attn_mask.shape) == 3: # Without ProphetNet fix
                attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states[i*num_beams*16:(i+1)*num_beams*16],
                    attention_mask=encoder_attn_mask[(i*num_beams*16):(i+1)*num_beams*16],
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)
            present_key_value = present_key_value + cross_attn_present_key_value

    feed_forward_output = self.feed_forward(hidden_states)
    hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def setup_serial_generation(model):
    model._prepare_input_ids_for_generation = types.MethodType(prep_inputs_for_generation, model)
    model._expand_inputs_for_generation = expanded_inputs
    for layer in model.prophetnet.decoder.layers:
        layer.forward = types.MethodType(serial_forward, layer)
    return model

def get_forward(model):
    return [layer.forward for layer in model.prophetnet.decoder.layers]

def revert_forwards(model, forwards):
    for i in range(len(forwards)):
        layer = model.prophetnet.decoder.layers[i]
        layer.forward = forwards[i]
    return model