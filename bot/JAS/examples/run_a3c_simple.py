import ray
import ray.rllib.agents.a3c.a3c as a3c
from ray.rllib.models import ModelCatalog, Model
from ray.rllib.utils.annotations import override
from ray.rllib.models.misc import normc_initializer, get_activation_fn, flatten
import tensorflow as tf
import tensorflow.contrib.slim as slim

import copy
from tensor2tensor.models import image_transformer_2d
from tensor2tensor.layers import common_image_attention as cia
import tensorflow as tf

class MyModelClass(Model):
    
    @override(Model)
    def _build_layers_v2(self, input_dict, num_outputs, options):
#        print(input_dict)
#        exit(222)
        hparams = copy.copy(options["custom_options"]["hparams"])
        #targets = tf.placeholder(
        #    tf.float32, [None, 11, 11, 1])
        targets = input_dict["prev_actions"]
        inputs = input_dict["obs"]
        # if not (tf.get_variable_scope().reuse or
        #         hparams.mode == tf.estimator.ModeKeys.PREDICT):
        #     tf.summary.image("inputs", inputs, max_outputs=1)
        #     tf.summary.image("targets", targets, max_outputs=1)
        with tf.name_scope('enc_prep'):
            encoder_input = cia.prepare_encoder(inputs, hparams)
        with tf.name_scope('enc_layers'):
            encoder_output = cia.transformer_encoder_layers(
                encoder_input,
                hparams.num_encoder_layers,
                hparams,
                attention_type=hparams.enc_attention_type,
                name="encoder")
        with tf.name_scope('dec_prep'):
            decoder_input, rows, cols = cia.prepare_decoder(
                targets, hparams)
        with tf.name_scope('dec_layers'):
            decoder_output = cia.transformer_decoder_layers(
                decoder_input,
                encoder_output,
                hparams.num_decoder_layers,
                hparams,
                attention_type=hparams.dec_attention_type,
                name="decoder")
        #with tf.name_scope('dec_out'):
        #    output = cia.create_output(decoder_output, rows, cols, targets, hparams)
        #print(output, encoder_output)

        out_size, kernel, stride = [32, [3, 3], 2]
        activation = get_activation_fn(options.get("conv_activation"))
        fc1 = slim.conv2d(
            decoder_output,
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
        #print(fc1, fc2)
        #print(flatten(fc1), flatten(fc2))
        #exit(123)
        return flatten(fc2), flatten(fc1)

        #return flatten(output), flatten(last_layer)

        

ModelCatalog.register_custom_model("my_model", MyModelClass)


hparams = image_transformer_2d.imagetransformer2d_base()
hparams.img_len = 84
hparams.num_channels = 1
hparams.add_hparam("mode", tf.estimator.ModeKeys.TRAIN)

my_mdl = False
if my_mdl:
    model = {
                "use_lstm": True,
                #"dim": 42,
                "custom_model": "my_model",
                "custom_options": {"hparams": hparams},  # extra options to pass to your model
                "grayscale": True,
            }
else:
    model = {
               "use_lstm": True,
               "conv_activation": "elu",
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


ray.init()
config = {
            "num_workers": 1,
            "sample_async": False,
            "sample_batch_size": 20,
            "use_pytorch": False,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": -0.01,
            "gamma": 0.99,
            "grad_clip": 40.0,
            "lambda": 1.0,
            "lr": 0.0001,
            "observation_filter": "NoFilter",
            "preprocessor_pref": "rllib",
            "model":model,
          }
agent = a3c.A3CAgent(config=config, env="PongDeterministic-v3")
policy_graph = agent.local_evaluator.policy_map["default"].sess.graph
writer = tf.summary.FileWriter(agent._result_logger.logdir, policy_graph)
writer.close()


    #transformer.TransformerEncoder(hparams, mode=tf.estimator.ModeKeys.TRAIN)
#x = <your inputs, which should be of shape [batch_size, timesteps, 1, hparams.hidden_dim]>
#y = encoder({"inputs": x})


while True:
    result = agent.train()
    print("training_iteration", result["training_iteration"])
    print("timesteps this iter", result["timesteps_this_iter"])
    print("timesteps_total", result["timesteps_total"])
    print("time_this_iter_s", result["time_this_iter_s"])
    print("time_total_s", result["time_total_s"])
    print("episode_reward_mean", result["episode_reward_mean"])
    print()