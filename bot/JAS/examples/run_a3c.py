import ray
import ray.rllib.agents.a3c.a3c as a3c
from ray.rllib.models import ModelCatalog, Model
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models.model import Model
from ray.rllib.models.misc import normc_initializer, get_activation_fn
from ray.rllib.utils.annotations import override
from ray.tune.tune import _make_scheduler, run_experiments

IMAGE_SIZE = 84

def make_image(tensor):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    from PIL import Image
    print(tensor.shape)
    _, height, width, channel = tensor.shape
    image = Image.fromarray(tensor)
    import io
    output = io.BytesIO()
    image.save(output, format='PNG')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=height,
                         width=width,
                         colorspace=channel,
                         encoded_image_string=image_string)


class MyModelClass(Model):

    @override(Model)
    def _build_layers_v2(self, input_dict, num_outputs, options):

        inputs = input_dict["obs"]
        #make_image(inputs)
        #tf.summary.image('images', tf.reshape(inputs, [-1, IMAGE_SIZE, IMAGE_SIZE, 1]), 50)
        hiddens = options.get("fcnet_hiddens")
        activation = get_activation_fn(options.get("fcnet_activation"))

        # Conv Layers
        convs = [32, 32, 32, 32]
        kerns = [3, 3, 3, 3]
        strides = [2, 2, 2, 2]
        pads = 'valid'
        #fc = 256
        activ = tf.nn.elu

        with tf.name_scope('conv1'):
            conv1 = tf.layers.conv2d(
                inputs=inputs,
                filters=convs[0],
                kernel_size=kerns[0],
                strides=strides[0],
                padding=pads,
                activation=activ,
                name='conv1')
        with tf.name_scope('conv2'):
            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=convs[1],
                kernel_size=kerns[1],
                strides=strides[1],
                padding=pads,
                activation=activ,
                name='conv2')
        with tf.name_scope('conv3'):
            conv3 = tf.layers.conv2d(
                inputs=conv2,
                filters=convs[2],
                kernel_size=kerns[2],
                strides=strides[2],
                padding=pads,
                activation=activ,
                name='conv3')
        with tf.name_scope('conv4'):
            conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=convs[3],
                kernel_size=kerns[3],
                strides=strides[3],
                padding=pads,
                activation=activ,
                name='conv4')
        flat = tf.layers.flatten(conv4)

        with tf.name_scope("fc_net"):
            i = 1
            last_layer = flat
            for size in hiddens:
                label = "fc{}".format(i)
                last_layer = slim.fully_connected(
                    last_layer,
                    size,
                    weights_initializer=normc_initializer(1.0),
                    activation_fn=activation,
                    scope=label)
                i += 1
            label = "fc_out"
            output = slim.fully_connected(
                last_layer,
                num_outputs,
                weights_initializer=normc_initializer(0.01),
                activation_fn=None,
                scope=label)
            return output, last_layer


ModelCatalog.register_custom_model("my_model", MyModelClass)

#ray.init(num_gpus=2)
ray.init()
print("hello!")

def my_train_fn(config, reporter):
    agent = a3c.A3CAgent(config=config)
    policy_graph = agent.local_evaluator.policy_map["default"].sess.graph
    writer = tf.summary.FileWriter(agent._result_logger.logdir, policy_graph)
    writer.close()
    for _ in range(10):
        result = agent.train()
        reporter(**result)

    agent.stop()








# experiments = {
#     "a3c_pong": {  # i.e. log to ~/ray_results/default
#         "run": my_train_fn,
#         "resources_per_trial": {
#             "cpu": 10,
#         },
#     }
# }

experiments = {
    "pong-a3c": {
        "env": "PongDeterministic-v3",
        "run": my_train_fn,
        "config": {
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
            "model":{
                "custom_model": "my_model",
                "custom_options": {},  # extra options to pass to your model
            }
#            "model":{
#                "use_lstm": True,
#                "conv_activation": "elu",
#                "dim": 42,
#                "grayscale": True,
#                "zero_mean": False,
#                # Reduced channel depth and kernel size from default
#                "conv_filters": [
#                    [32, [3, 3], 2],
#                    [32, [3, 3], 2],
#                    [32, [3, 3], 2],
#                    [32, [3, 3], 2],
#                ]
#            }
        }
    }
}


run_experiments(
    experiments,
#    raise_on_failed_trial=True,
#    resume=True
)


# while True:
#     result = agent.train()
#     # print(result)
#     print("training_iteration", result['training_iteration'])
#     print("timesteps this iter", result['timesteps_this_iter'])
#     print("timesteps_total", result['timesteps_total'])
#     print("time_this_iter_s", result['time_this_iter_s'])
#     print("time_total_s", result['time_total_s'])
#     print("episode_reward_mean", result['episode_reward_mean'])
#     print()