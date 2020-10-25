

import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from adaptive_instance_norm import AdaIN


class StyleTransferNet(object):

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()

    def transform(self, content, style):
        content = tf.reverse(content, axis=[-1])
        style   = tf.reverse(style,   axis=[-1])

        content = self.encoder.preprocess(content)
        style   = self.encoder.preprocess(style)

        enc_c, enc_c_layers = self.encoder.encode(content)
        enc_s, enc_s_layers = self.encoder.encode(style)

        self.encoded_content_layers = enc_c_layers
        self.encoded_style_layers   = enc_s_layers


        target_features = AdaIN(enc_c, enc_s)
        self.target_features = target_features

        generated_img = self.decoder.decode(target_features)


        generated_img = self.encoder.deprocess(generated_img)


        generated_img = tf.reverse(generated_img, axis=[-1])

        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)

        return generated_img

