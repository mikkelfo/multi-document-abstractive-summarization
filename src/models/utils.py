import torch
import os
import gzip, json
import types


def custom_forward_mds(model, input_ids, attention_mask, labels, args):
    enc_output = model.prophetnet.encoder(input_ids=input_ids, attention_mask=attention_mask)

    # No attention_mask if method is 'mean'
    if args.method == 'mean':
        enc_output.last_hidden_state = enc_output.last_hidden_state.mean(1).unsqueeze(0)
        output = model(encoder_outputs=enc_output, labels=labels, use_cache=False)
    # SDS and Serial uses attention_mask
    else:
        output = model(encoder_outputs=enc_output, attention_mask=attention_mask, labels=labels, use_cache=False)
    return output


def process_chunk(split, chunk_idx, args):
    summary = torch.load(f'data/processed/{args.dir}/summary/{split}/chunk_{chunk_idx}.pt', map_location=torch.device('cuda'))
    text = torch.load(f'data/processed/{args.dir}/text/{split}/chunk_{chunk_idx}.pt', map_location=torch.device('cuda'))

    if args.mds:
        N = len(summary)  # ~64
        for i in range(N):
            input_ids = text[i].input_ids[:, :args.token_length]
            attention_mask = text[i].attention_mask[:, :args.token_length]
            labels = summary[i].input_ids[:, :args.token_length]
            labels = labels.masked_fill_(labels == 0, -100)
            if args.method == 'sds':
                labels = labels.expand(len(input_ids), -1)
            elif args.serial_strat == 'shuffle':
                shuffle = torch.randperm(input_ids.shape[0])
                input_ids = input_ids[shuffle].clone().detach()
                attention_mask = attention_mask[shuffle].clone().detach()
            elif args.serial_strat == 'prio':
                input_ids = input_ids.flip(0).clone().detach()
                attention_mask = attention_mask.flip(0).clone().detach()
            yield input_ids, attention_mask, labels

    else:
        input_ids, attention_mask, = text['input_ids'][:, :args.token_length], text['attention_mask'][:, :args.token_length]
        labels = summary['input_ids'][:, :args.token_length]
        labels = labels.masked_fill_(labels == 0, -100)

        N = len(input_ids)  # 512
        for i in range(0, N, args.batch_size):
            batch = input_ids[i:(i+args.batch_size)], attention_mask[i:(i+args.batch_size)], labels[i:(i+args.batch_size)]
            yield batch


def read_jsonl_gz(path):
    with gzip.open(path) as f:
        for l in f:
            yield json.loads(l)


def target_summaries(split):
    # VM's doesnt like this import
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    data = list(read_jsonl_gz(f'data/raw/wcep/{split}.jsonl.gz'))

    targets = []
    for cluster in data:
        articles = cluster['articles']
        text = [a['text'] for a in articles][:16]
        X = vectorizer.fit_transform(text)
        targets.append(cosine_similarity(X).sum(1).argmax())

    return targets


def concat_chunks(dir):
    input_ids = torch.tensor([])
    attention_mask = torch.tensor([])
    for i in range(len(os.listdir(dir))):
        chunk = torch.load(f'{dir}/chunk_{i}.pt')
        input_ids = torch.cat((input_ids, chunk.input_ids))
        attention_mask = torch.cat((attention_mask, chunk.attention_mask))

    return list(zip(input_ids, attention_mask))


def implement_serial_input(model):
    for layer in model.prophetnet.decoder.layers:
        layer.forward = types.MethodType(serial_forward, layer)
    return model


# Copy pasted from https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/prophetnet/modeling_prophetnet.py#L1176
# Added for loop during cross_attn
def serial_forward(
    self,
    hidden_states,
    attention_mask=None,
    encoder_hidden_states=None,
    encoder_attn_mask=None,
    layer_head_mask=None,
    cross_attn_layer_head_mask=None,
    extended_predict_attention_mask=None,
    main_relative_position_buckets=None,
    predict_relative_position_buckets=None,
    position_ids=None,
    past_key_value=None,
    use_cache: bool = True,
    output_attentions: bool = False,
):
    # 1st residual block
    # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    ngram_attention_output, self_attn_weights, self_attn_weights_ngram, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=self_attn_past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        extended_predict_attention_mask=extended_predict_attention_mask,
        main_relative_position_buckets=main_relative_position_buckets,
        predict_relative_position_buckets=predict_relative_position_buckets,
        position_ids=position_ids,
    )
    hidden_states = self.self_attn_layer_norm(hidden_states + ngram_attention_output)

    # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
    cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
    cross_attn_weights = None
    if encoder_hidden_states is not None:
        # SERIAL INPUT
        for i in range(len(encoder_hidden_states)):
            # 2nd residual block
            attention_output, cross_attn_weights, cross_attn_present_key_value = self.cross_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states[i:i+1],
                attention_mask=encoder_attn_mask[i:i+1],
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = self.cross_attn_layer_norm(attention_output + hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

    # 3rd residual block
    feed_forward_output = self.feed_forward(hidden_states)
    hidden_states = self.feed_forward_layer_norm(feed_forward_output + hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights, self_attn_weights_ngram, cross_attn_weights)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

