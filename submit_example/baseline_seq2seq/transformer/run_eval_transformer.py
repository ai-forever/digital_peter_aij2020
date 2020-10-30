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

import json
from keras_transformer import get_model
from keras_transformer import decode
import tensorflow.python.util.deprecation as deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

letters = [' ', ')', '+', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']', 'i', 'k', 'l', '|', '×', 'ǂ',
           'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х',
           'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я', 'і', 'ѣ', '–', '…', '⊕', '⊗']


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


def load_json(fname_path):
    with open(fname_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_transformer_model(config):
    model_params = config["model"]
    source_token_dict = config["vocab"]["source_token_dict"]
    target_token_dict = config["vocab"]["target_token_dict"]
    embed_dim = model_params["embed_dim"]
    hidden_dim = model_params["hidden_dim"]
    head_num = model_params["head_num"]
    encoder_num = model_params["encoder_num"]
    decoder_num = model_params["decoder_num"]
    dropout_rate = model_params["dropout_rate"]
    use_same_embed = bool(model_params["use_same_embed"])

    model = get_model(
        token_num=max(len(source_token_dict), len(target_token_dict)),
        embed_dim=embed_dim,
        encoder_num=encoder_num,
        decoder_num=decoder_num,
        head_num=head_num,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        use_same_embed=use_same_embed,
    )
    return model


def transformer_model_predict(prediction, transformer_model, config):
    source_token_dict = config["vocab"]["source_token_dict"]
    target_token_dict = config["vocab"]["target_token_dict"]
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    sos_token = config["tok"]["sos_token"]
    eos_token = config["tok"]["eos_token"]
    pad_token = config["tok"]["pad_token"]

    encode_tokens = [[sos_token] + list(pred) + [eos_token] for pred in prediction]
    source_max_len = max(map(len, encode_tokens))
    encode_tokens = [
        tokens + [pad_token] * (source_max_len - len(tokens))
        for tokens in encode_tokens
    ]
    encode_input = [
        list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens
    ]

    decoded = decode(
        transformer_model,
        encode_input,
        start_token=target_token_dict[sos_token],
        end_token=target_token_dict[eos_token],
        pad_token=target_token_dict[pad_token],
        temperature=1.0,
    )

    res = []

    for pred in decoded:
        pred = [
            char
            for char in pred
            if not target_token_dict_inv[char] in (sos_token, eos_token, pad_token)
        ]
        pred = "".join(map(lambda x: target_token_dict_inv[x], pred))
        res.append(pred)

    return res


def get_prediction(act_model, transformer_model, config, test_images):
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
        prediction.append(pred)

    transformer_prediction = transformer_model_predict(
        prediction, transformer_model, config
    )
    return transformer_prediction


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
    transformer_filepath = "transformer_model/transformer_model_base.h5"
    transformer_model_config_path = "transformer_model/config.json"

    print("Creating model...", end=" ")
    act_model = create_model()
    print("Success")

    print(f"Loading weights from {filepath}...", end=" ")
    act_model.load_weights(filepath)
    print("Success")

    print(f"Loading transformer model config...", end=" ")
    transformer_model_config = load_json(transformer_model_config_path)
    print("Success")

    print(f"Creating transformer model...", end=" ")
    transformer_model = load_transformer_model(transformer_model_config)
    print("Success")

    print(f"Loading transformer model weights from {transformer_filepath}...", end=" ")
    transformer_model.load_weights(transformer_filepath)
    print("Success")

    print(f"Loading test images from {test_image_dir}...", end=" ")
    names_test, test_images = load_test_images(test_image_dir)
    print("Success")

    print("Running inference...")
    prediction = get_prediction(
        act_model, transformer_model, transformer_model_config, test_images
    )

    print("Writing predictions...")
    write_prediction(names_test, prediction, pred_path)


if __name__ == "__main__":
    main()
