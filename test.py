import tensorflow as tf

from edward.models import Empirical
import tensorflow as tf
import edward as ed
from edward.models import Normal
import numpy as np

# z = Normal(loc=0.0, scale=1.0)
# x = Normal(loc=tf.ones(10) * z, scale=1.0)
# sess = tf.Session()
# print(x.eval(sess))
#
# qz = Empirical(tf.Variable(tf.zeros(500)))
# proposal_z = Normal(loc=z, scale=0.5)
# data = {x: np.array([0.0] * 10, dtype=np.float32)}
# inference = ed.MetropolisHastings({z: qz}, {z: proposal_z}, data)
#
# tf.nn.sigmoid_cross_entropy_with_logits()

