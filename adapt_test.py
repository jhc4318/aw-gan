import SoftAdapt as sa
import tensorflow as tf

loss = [[1.1, 1.2], [0.5, 0.6]]
loss_tensor = tf.constant(loss)
sess = tf.Session()
# sess.run(loss_tensor)
adapt = sa.Adapt(sess.run(loss_tensor))