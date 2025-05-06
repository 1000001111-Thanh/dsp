import tensorflow as tf

def generate_prng_seq_tf(length, c_init):
    r"""TensorFlow implementation of pseudo-random sequence generator as defined in Sec. 5.2.1
    in [3GPP38211]_ based on a length-31 Gold sequence.

    Parameters
    ----------
    length: `int`
        Desired output sequence length

    c_init: `int`
        Initialization sequence of the PRNG. Must be in the range of 0 to
        :math:`2^{32}-1`.

    Output
    ------
    : `tf.Tensor` of shape [length] and dtype tf.float32
        Containing the scrambling sequence (0s and 1s)
    """
    # Validate inputs
    length = tf.cast(length, tf.int32)
    c_init = tf.cast(c_init, tf.int32)
    
    # tf.debugging.assert_non_negative(length, message="length must be a positive integer")
    # tf.debugging.assert_less(c_init, 2**32, message="c_init must be in [0, 2^32-1]")
    # tf.debugging.assert_greater_equal(c_init, 0, message="c_init must be in [0, 2^32-1]")

    # Internal parameters
    n_seq = 31  # length of gold sequence
    n_c = 1600  # defined in 5.2.1 in 38.211
    total_length = length + n_c + n_seq

    # Initialize sequences
    x1 = tf.TensorArray(tf.float32, size=total_length, clear_after_read=False)
    x2 = tf.TensorArray(tf.float32, size=total_length, clear_after_read=False)

    # Initialize x1 and x2
    # x1[0] = 1, rest are 0
    x1 = x1.write(0, 1.0)
    
    # Convert c_init to binary and pad to 31 bits
    c_init_bin = tf.bitwise.right_shift(
        tf.bitwise.bitwise_and(
            tf.reverse(tf.range(n_seq, dtype=tf.int32), [1] * n_seq),
            c_init), 
        tf.range(n_seq, dtype=tf.int32))
    c_init_bin = tf.cast(c_init_bin, tf.float32)
    
    # Initialize x2 with c_init_bin
    for i in tf.range(n_seq):
        x2 = x2.write(i, c_init_bin[i])

    # Run the generator
    def body(idx, x1, x2):
        x1_val = tf.math.floormod(x1.read(idx + 3) + x1.read(idx), 2)
        x1 = x1.write(idx + 31, x1_val)
        
        x2_val = tf.math.floormod(
            x2.read(idx + 3) + x2.read(idx + 2) + x2.read(idx + 1) + x2.read(idx), 2)
        x2 = x2.write(idx + 31, x2_val)
        return idx + 1, x1, x2

    _, x1, x2 = tf.while_loop(
        lambda idx, *_: idx < length + n_c,
        body,
        loop_vars=(0, x1, x2),
        maximum_iterations=length + n_c)

    # Generate output sequence
    c = tf.TensorArray(tf.float32, size=length)
    
    def output_body(idx, c_arr):
        val = tf.math.floormod(x1.read(idx + n_c) + x2.read(idx + n_c), 2)
        c_arr = c_arr.write(idx, val)
        return idx + 1, c_arr

    _, c = tf.while_loop(
        lambda idx, *_: idx < length,
        output_body,
        loop_vars=(0, c),
        maximum_iterations=length)

    return c.stack()