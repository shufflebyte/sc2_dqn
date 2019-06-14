import ray
import ray.rllib.agents.a3c.a3c as a3c
import ray.rllib.agents.dqn.dqn as dqn
import ray.rllib.agents.impala.impala as impala
import ray.rllib.agents.pg.pg as pg
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.utils.annotations import override
from ray.rllib.models.misc import normc_initializer, get_activation_fn, flatten
import tensorflow.contrib.slim as slim


from tensor2tensor.models import image_transformer_2d
from tensor2tensor.utils import registry

from tensor2tensor.layers import modalities
from tensor2tensor.data_generators import image_utils
from tensor2tensor.utils import t2t_model
import copy
from tensor2tensor.layers import common_image_attention as cia
from tensor2tensor.utils.hparam import HParams

import tensorflow as tf

IMAGE = 21


@registry.register_problem
class ImageCeleba(image_utils.ImageProblem):
  """CelebA dataset, aligned and cropped images."""
  IMG_DATA = ("img_align_celeba.zip",
              "https://drive.google.com/uc?export=download&"
              "id=0B7EVK8r0v71pZjFTYXZWM3FlRnM")
  LANDMARKS_DATA = ("celeba_landmarks_align",
                    "https://drive.google.com/uc?export=download&"
                    "id=0B7EVK8r0v71pd0FJY3Blby1HUTQ")
  ATTR_DATA = ("celeba_attr", "https://drive.google.com/uc?export=download&"
               "id=0B7EVK8r0v71pblRyaVFSWGxPY0U")

  LANDMARK_HEADINGS = ("lefteye_x lefteye_y righteye_x righteye_y "
                       "nose_x nose_y leftmouth_x leftmouth_y rightmouth_x "
                       "rightmouth_y").split()
  ATTR_HEADINGS = (
      "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
      "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
      "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
      "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
      "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
      "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
      "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
  ).split()

@registry.register_problem
class Img2imgCeleba(ImageCeleba):
  """8px to 32px problem."""

  def dataset_filename(self):
    return "image_celeba"

  def preprocess_example(self, example, unused_mode, unused_hparams):
    image = example["inputs"]
    # Remove boundaries in CelebA images. Remove 40 pixels each side
    # vertically and 20 pixels each side horizontally.
    image = tf.image.crop_to_bounding_box(image, 40, 20, 218 - 80, 178 - 40)
    image_8 = image_utils.resize_by_area(image, 8)
    image_32 = image_utils.resize_by_area(image, 32)

    example["inputs"] = image_8
    example["targets"] = image_32
    return example


class ImgEncTransformer(t2t_model.T2TModel):
  """jbr"""

  def body(self, features):
    hparams = copy.copy(self._hparams)
    #targets = features["targets"]
    inputs = features["inputs"]
#    if not (tf.get_variable_scope().reuse or
#            hparams.mode == tf.estimator.ModeKeys.PREDICT):
#      tf.summary.image("inputs", inputs, max_outputs=1)
#      tf.summary.image("targets", targets, max_outputs=1)

    encoder_input = cia.prepare_encoder(inputs, hparams)
    encoder_output = cia.transformer_encoder_layers(
        encoder_input,
        hparams.num_encoder_layers,
        hparams,
        attention_type=hparams.enc_attention_type,
        name="encoder")
    from tensor2tensor.layers import common_layers
    targets_shape = common_layers.shape_list(inputs)
    targets = tf.reshape(encoder_output,
                         [-1, hparams.img_len, hparams.img_len, hparams.hidden_size])
                         #[targets_shape[0], -1, hparams.img_len, 1])
    #decoder_input, rows, cols = cia.prepare_decoder(
    #    targets, hparams)
    #print(decoder_input, rows, cols)
    #return encoder_output
    return targets

  def loss(self, logits, features):
      return 0





def _get_filter_config(shape):
    shape = list(shape)
    filters_84x84 = [
        [16, [8, 8], 4],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]
    filters_42x42 = [
        [16, [4, 4], 2],
        [32, [4, 4], 2],
        [256, [11, 11], 1],
    ]
    if len(shape) == 3 and shape[:2] == [84, 84]:
        return filters_84x84
    elif len(shape) == 3 and shape[:2] == [42, 42]:
        return filters_42x42
    else:
        raise ValueError(
            "No default configuration for obs shape {}".format(shape) +
            ", you must specify `conv_filters` manually as a model option. "
            "Default configurations are only available for inputs of shape "
            "[42, 42, K] and [84, 84, K]. You may alternatively want "
            "to use a custom model or preprocessor.")


class MyModelClass(Model):
    """Generic vision network."""

    @override(Model)
    def _build_layers_v2(self, input_dict, num_outputs, options):
        inputs = input_dict["obs"]
        filters = options.get("conv_filters")
        if not filters:
            filters = _get_filter_config(inputs.shape.as_list()[1:])

        activation = get_activation_fn(options.get("conv_activation"))

        inputs = slim.conv2d(
            inputs,
            16,
            (3,3),
            1,
            activation_fn=activation,
            scope="conv_trans_in")

        tf.layers.max_pooling2d(
            inputs,
            (2,2),
            strides=1,
            padding='same',
            # data_format='channels_last',
            name="pooling"
        )

        """ Begin Transformer"""
        hparams = image_transformer_2d.img2img_transformer2d_tiny()
        hparams.data_dir = ""
        hparams.img_len = IMAGE
        hparams.num_channels = 16
        hparams.hidden_size = 8

        p_hparams = Img2imgCeleba().get_hparams(hparams)

        p_hparams.modality = {
                       "inputs": modalities.ModalityType.IMAGE,
                       "targets": modalities.ModalityType.IMAGE,
                   }
        p_hparams.vocab_size = {
                         "inputs": IMAGE,
                         "targets": IMAGE,
                     }
        features = {
          "inputs": inputs,
          #"targets": target,
          #"target_space_id": tf.constant(1, dtype=tf.int32),
        }
        #model = image_transformer_2d.Img2imgTransformer(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)
        model = ImgEncTransformer(hparams, tf.estimator.ModeKeys.TRAIN, p_hparams)

        trans_logits, trans_losses = model(features)
        print("trans_logits", trans_logits)
        print("inputs", inputs)
        """ End Transformer"""



        #inputs = trans_logits

        ## TAKE CARE! Normalization?!
        inputs = tf.contrib.layers.batch_norm(
            trans_logits,
            data_format='NHWC',  # Matching the "cnn" tensor which has shape (?, 480, 640, 128).
            center=True,
            scale=True,
            #is_training=training,
            scope='cnn-batch_norm')

        with tf.name_scope("vision_net"):
            for i, (out_size, kernel, stride) in enumerate(filters[:-1], 1):
                inputs = slim.conv2d(
                    inputs,
                    out_size,
                    kernel,
                    stride,
                    activation_fn=activation,
                    scope="conv{}".format(i))
                print(i, inputs)
            out_size, kernel, stride = filters[-1]
            fc1 = slim.conv2d(
                inputs,
                out_size,
                kernel,
                stride,
                activation_fn=activation,
                padding="VALID",
                scope="fc1")
            fc2 = slim.conv2d(
                fc1,
                num_outputs, [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope="fc2")
            print(fc1, fc2)
            print(flatten(fc1), flatten(fc2))
            # exit(123)
            return flatten(fc2), flatten(fc1)

ModelCatalog.register_custom_model("my_model", MyModelClass)

model = {
    "use_lstm": True,
    "conv_activation": "elu",
    "custom_model": "my_model",
    "dim": 42,
    "grayscale": True,
    "zero_mean": False,
    # Reduced channel depth and kernel size from default
    "conv_filters": [
        [32, [3, 3], 2],
        [32, [3, 3], 2],
        [32, [3, 3], 2],
        [32, [3, 3], 2],
    ]
}


ray.init(
#    local_mode=True
)
config = { # for A3C
    "num_workers": 16,
    "num_gpus": 2,
    "sample_async": False,
    "sample_batch_size": 20,
    # "use_pytorch": False,
    # "vf_loss_coeff": 0.5,
    # "entropy_coeff": -0.01,
    "gamma": 0.99,
    # "grad_clip": 40.0,
    # "lambda": 1.0,
    "lr": 0.0001,
    "observation_filter": "NoFilter",
    "preprocessor_pref": "rllib",
    "model": model,
    "log_level": "DEBUG"
}
agent = impala.ImpalaAgent(config=config, env="PongDeterministic-v4")
#agent = pg.PGAgent(config=config, env="PongDeterministic-v4")
#agent = a3c.A3CAgent(config=config, env="PongDeterministic-v4")
#agent = dqn.DQNAgent(config=config, env="PongDeterministic-v4")
policy_graph = agent.local_evaluator.policy_map["default"].sess.graph
writer = tf.summary.FileWriter(agent._result_logger.logdir, policy_graph)
writer.close()




while True:
    result = agent.train()
    print(result)
    print("training_iteration", result["training_iteration"])
    print("timesteps this iter", result["timesteps_this_iter"])
    print("timesteps_total", result["timesteps_total"])
    print("time_this_iter_s", result["time_this_iter_s"])
    print("time_total_s", result["time_total_s"])
    print("episode_reward_mean", result["episode_reward_mean"])
    print()