import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import os

import cv2
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    GRU,
    Input,
    Lambda,
    MaxPool2D,
)
from tensorflow.keras.models import Model

letters = [' ', ')', '+', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', 'i', 'k', 'l', '|', '×', 'ǂ',
           'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х',
           'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'і', 'ѣ', '–', '…', '⊕', '⊗']


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, trg_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.generator = generator

    def forward(self, src, trg, src_mask, trg_mask, src_lengths, trg_lengths):
        encoder_hidden, encoder_final = self.encode(src, src_mask, src_lengths)
        return self.decode(encoder_hidden, encoder_final, src_mask, trg, trg_mask)

    def encode(self, src, src_mask, src_lengths):
        return self.encoder(self.src_embed(src), src_mask, src_lengths)

    def decode(
        self,
        encoder_hidden,
        encoder_final,
        src_mask,
        trg,
        trg_mask,
        decoder_hidden=None,
    ):
        return self.decoder(
            self.trg_embed(trg),
            encoder_hidden,
            encoder_final,
            src_mask,
            trg_mask,
            hidden=decoder_hidden,
        )


class Generator(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, x, mask, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        fwd_final = final[0 : final.size(0) : 2]
        bwd_final = final[1 : final.size(0) : 2]
        final = torch.cat([fwd_final, bwd_final], dim=2)
        return output, final


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(
        self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5, bridge=True
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.rnn = nn.GRU(
            emb_size + 2 * hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.bridge = (
            nn.Linear(2 * hidden_size, hidden_size, bias=True) if bridge else None
        )
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(
            hidden_size + 2 * hidden_size + emb_size, hidden_size, bias=False
        )

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        query = hidden[-1].unsqueeze(1)
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key, value=encoder_hidden, mask=src_mask
        )
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        return output, hidden, pre_output

    def forward(
        self,
        trg_embed,
        encoder_hidden,
        encoder_final,
        src_mask,
        trg_mask,
        hidden=None,
        max_len=None,
    ):
        if max_len is None:
            max_len = trg_mask.size(-1)
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        proj_key = self.attention.key_layer(encoder_hidden)
        decoder_states = []
        pre_output_vectors = []
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden
            )
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)
        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors

    def init_hidden(self, encoder_final):
        if encoder_final is None:
            return None
        return torch.tanh(self.bridge(encoder_final))


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        self.alphas = None

    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None
        query = self.query_layer(query)
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        scores.data.masked_fill_(mask == 0, -float("inf"))
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas
        context = torch.bmm(alphas, value)
        return context, alphas


def load_encoder_decoder_model(config, device):
    model_params = config["model"]
    model_path = model_params["model_path"]
    emb_size = model_params["emb_size"]
    hidden_size = model_params["hidden_size"]
    num_layers = model_params["num_layers"]
    dropout = model_params["dropout"]

    state_dict = torch.load(model_path)
    source_dim = state_dict["src_embed.weight"].shape[0]
    target_dim = state_dict["trg_embed.weight"].shape[0]

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(
            emb_size,
            hidden_size,
            BahdanauAttention(hidden_size),
            num_layers=num_layers,
            dropout=dropout,
        ),
        nn.Embedding(source_dim, emb_size),
        nn.Embedding(target_dim, emb_size),
        Generator(hidden_size, target_dim),
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


class CharTokenizer(object):
    def __init__(self, config):
        self.config = config
        self.src_stoi = self.config["vocab"]["src_stoi"]
        self.trg_stoi = self.config["vocab"]["trg_stoi"]
        self.src_itos = {v: k for k, v in self.src_stoi.items()}
        self.trg_itos = {v: k for k, v in self.trg_stoi.items()}
        self.eos_token = self.config["tok"]["eos_token"]
        self.unk_token = self.config["tok"]["unk_token"]
        self.pad_token = self.config["tok"]["pad_token"]
        self.sos_token = self.config["tok"]["sos_token"]

    def encode(self, sequence):
        enc = [
            self.src_stoi[char]
            if char in self.src_stoi
            else self.stoi[self.unk_token_id]
            for char in list(sequence)
        ] + [self.src_stoi[self.eos_token]]
        return torch.tensor(enc).unsqueeze(0)

    def create_mask(self, enc):
        return (enc != self.src_stoi[self.pad_token]).unsqueeze(-2)

    def get_length(self, enc):
        return torch.tensor(enc.shape[-1], dtype=torch.int64).unsqueeze(0)


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def dummy_copy(sequence, output):
    diff = len(sequence) - len(output)
    if diff > 0:
        return output + sequence[-diff:]
    return output


def greedy_decode(sequence, model, tokenizer, device, max_len=128):
    src = tokenizer.encode(sequence).to(device)
    src_mask = tokenizer.create_mask(src).to(device)
    src_length = tokenizer.get_length(src).to(device)
    sos_index = tokenizer.trg_stoi[tokenizer.sos_token]
    eos_index = tokenizer.trg_stoi[tokenizer.eos_token]

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(src, src_mask, src_length)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(src)
        trg_mask = torch.ones_like(prev_y)

    output = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
                encoder_hidden, encoder_final, src_mask, prev_y, trg_mask, hidden
            )
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(src).fill_(next_word)

    output = np.array(output)
    if eos_index is not None:
        first_eos = np.where(output == eos_index)[0]
        if len(first_eos) > 0:
            output = output[: first_eos[0]]

    output = "".join([tokenizer.trg_itos[token_id] for token_id in output.tolist()])
    return output


def process_image(img):
    w, h = img.shape

    new_w = 128
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    if w < 128:
        add_zeros = np.full((128 - w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 1024:
        add_zeros = np.full((w, 1024 - h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 1024 or w > 128:
        dim = (1024, 128)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = img / 255

    return img


def create_model():
    inputs = Input(shape=(128, 1024, 1))

    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(4, 2), strides=2)(conv_1)

    conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(4, 2), strides=2)(conv_2)

    conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)

    conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)

    pool_4 = MaxPool2D(pool_size=(4, 1), padding='same')(conv_4)

    conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_4)

    batch_norm_5 = BatchNormalization()(conv_5)

    conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(4, 1), padding='same')(batch_norm_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

    blstm_1 = Bidirectional(GRU(256, return_sequences=True, dropout=0.2))(squeezed)
    blstm_2 = Bidirectional(GRU(256, return_sequences=True, dropout=0.2))(blstm_1)

    outputs = Dense(len(letters) + 1, activation='softmax')(blstm_2)
    act_model = Model(inputs=inputs, outputs=outputs)

    return act_model


def get_prediction(act_model, encoder_decoder_model, tokenizer, device, test_images):
    prediction = act_model.predict(test_images)

    decoded = K.ctc_decode(
        prediction,
        input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
        greedy=True,
    )[0][0]

    out = K.get_value(decoded)

    prediction = []
    for i, x in enumerate(out):
        pred = ""
        for p in x:
            if int(p) != -1:
                pred += letters[int(p)]
        encoder_decoder_pred = greedy_decode(
            pred, encoder_decoder_model, tokenizer, device
        )
        prediction.append(encoder_decoder_pred)
    return prediction


def write_prediction(names_test, prediction, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for num, (name, line) in enumerate(zip(names_test, prediction)):
        with open(os.path.join(output_dir, name.replace('.jpg', '.txt')), 'w') as file:
            file.write(line)


def load_test_images(test_image_dir):
    test_images = []
    names_test = []
    for name in os.listdir(test_image_dir):
        img = cv2.imread(test_image_dir + '/' + name, cv2.IMREAD_GRAYSCALE)
        img = process_image(img)
        test_images.append(img)
        names_test.append(name)
    test_images = np.asarray(test_images)
    return names_test, test_images


def main():
    test_image_dir = "/data"
    filepath = "checkpoint/model.hdf5"
    pred_path = "/output"
    encoder_decoder_filepath = "encoder_decoder_model/encoder_decoder_model.pt"
    encoder_decoder_model_config_path = "encoder_decoder_model/config.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Creating model...", end=" ")
    act_model = create_model()
    print("Success")

    print(f"Loading weights from {filepath}...", end=" ")
    act_model.load_weights(filepath)
    print("Success")

    print(f"Loading encoder-decoder model config...", end=" ")
    encoder_decoder_model_config = load_config(encoder_decoder_model_config_path)
    print("Success")

    print(f"Creating tokenizer...", end=" ")
    tokenizer = CharTokenizer(encoder_decoder_model_config)
    print("Success")

    print(f"Loading encoder-decoder model from {encoder_decoder_filepath}...", end=" ")
    encoder_decoder_model = load_encoder_decoder_model(
        encoder_decoder_model_config, device
    )
    print("Success")

    print(f"Loading test images from {test_image_dir}...", end=" ")
    names_test, test_images = load_test_images(test_image_dir)
    print("Success")

    print("Running inference...")
    prediction = get_prediction(
        act_model, encoder_decoder_model, tokenizer, device, test_images
    )

    print("Writing predictions...")
    write_prediction(names_test, prediction, pred_path)


if __name__ == "__main__":
    main()

