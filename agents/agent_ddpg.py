import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, initializers
from keras import backend as K

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a sigmoid to keep the action
    between 0 and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out, self.model = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out, self.target_model = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        inputs = layers.Input(shape=(self.s_dim,))
        net = layers.Dense(units=400, use_bias=False)(inputs)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=300, use_bias=False)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
        out = layers.Dense(units=self.a_dim, kernel_initializer=w_init, activation='sigmoid')(net)
        # Scale output to 0 to action_bound
        scaled_out = layers.Lambda(lambda x: (x * self.action_bound))(out)

        # Create Keras model
        model = models.Model(inputs=inputs, outputs=scaled_out)

        return inputs, out, scaled_out, model

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.target_model.predict_on_batch(inputs)

    def update_target_network(self):
        soft_update(self.model, self.target_model, self.tau)


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out, self.model = self.create_critic_network()

        # self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out, self.target_model = self.create_critic_network()

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = layers.Input(shape=(self.s_dim,))
        action = layers.Input(shape=(self.a_dim,))
        net = layers.Dense(units=400, use_bias=False)(inputs)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = layers.Dense(units=300, use_bias=False)(net)
        t2 = layers.Dense(units=300)(action)

        net = layers.Add()([t1, t2])
        net = layers.Activation('relu')(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = initializers.RandomUniform(minval=-0.003, maxval=0.003)
        out = layers.Dense(units=1, kernel_initializer=w_init)(net)

        # Create Keras model
        model = models.Model(inputs=[inputs, action], outputs=out)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        return inputs, action, out, model

    def train(self, inputs, action, predicted_q_value):
        return self.model.train_on_batch(x=[inputs, action], y=predicted_q_value)

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.target_model.predict_on_batch([inputs, action])

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        soft_update(self.model, self.target_model, self.tau)


def soft_update(local_model, target_model, tau):
    """Soft update model parameters."""
    local_weights = np.array(local_model.get_weights())
    target_weights = np.array(target_model.get_weights())

    assert len(local_weights) == len(target_weights), 'Local and target model parameters must have the same size'

    new_weights = tau * local_weights + (1 - tau) * target_weights
    target_model.set_weights(new_weights)
