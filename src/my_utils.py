# import tensorflow as tf

# # @tf.function
# # def c_init_tf(l, n_id, n_scid, slot_number, num_symbols_per_slot):
# #     """TensorFlow version of c_init from 3GPP 38.211"""
# #     lambda_bar = tf.constant(0, dtype=tf.float32)

# #     term1 = tf.cast(2**17, tf.int32) * (
# #         num_symbols_per_slot * slot_number + tf.cast(l, tf.int32) + 1
# #     ) * (2 * n_id + 1)
# #     term2 = tf.cast(2**17 * tf.math.floor(lambda_bar / 2), tf.int32)
# #     term3 = 2 * n_id + n_scid
# #     c_init = (term1 + term2 + term3) % (2**31)
# #     return tf.cast(c_init, tf.int32)

# @tf.function
# def generate_dmrs_grid_tf(num_symbols_per_slot, slot_number,
#                           n_id, n_scid,
#                           dmrs_port_set, num_subcarriers, num_symbols,
#                           first_subcarrier, l_bar, l_prime, n,
#                           config_type, deltas, w_f, w_t,
#                           l_ref, beta):
#     num_ports = tf.shape(dmrs_port_set)[0]
#     a_tilde = tf.TensorArray(tf.complex64, size=num_ports)
#     for j in tf.range(num_ports):
#         a_tilde = a_tilde.write(j, tf.zeros([num_subcarriers, num_symbols], dtype=tf.complex64))

#     for l_bar_val in l_bar:
#         for l_prime_val in l_prime:
#             l = l_bar_val + l_prime_val
#             c_init = 2**17 * (num_symbols_per_slot * slot_number + l + 1) * (2 * n_id + 1) + 2 * n_id + n_scid
#             c_init = tf.math.mod(c_init, 2**31)

#             if config_type == 1:
#                 _skip = first_subcarrier
#                 _len = num_subcarriers
#             else:
#                 _skip = 2 * first_subcarrier // 3
#                 _len = 2 * num_subcarriers // 3

#             c = generate_prng_seq_tf(_skip + _len, c_init)[_skip:]
#             r = (1 / tf.sqrt(2.0)) * (tf.cast(1 - 2 * c[::2], tf.complex64) +
#                                       1j * tf.cast(1 - 2 * c[1::2], tf.complex64))

#             for j_ind in tf.range(num_ports):
#                 for n_val in n:
#                     for k_prime in [0, 1]:
#                         if config_type == 1:
#                             k = 4 * n_val + 2 * k_prime + deltas[j_ind]
#                         else:
#                             k = 6 * n_val + k_prime + deltas[j_ind]

#                         value = r[2 * n_val + k_prime] * w_f[k_prime][j_ind] * w_t[l_prime_val][j_ind]
#                         updated = tf.tensor_scatter_nd_update(
#                             a_tilde.read(j_ind),
#                             [[k, l_ref + l]],
#                             [value]
#                         )
#                         a_tilde = a_tilde.write(j_ind, updated)

#     result = tf.stack([beta * a_tilde.read(i) for i in tf.range(num_ports)])
#     return result

