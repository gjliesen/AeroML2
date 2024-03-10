import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from aero_ml.data_engine import DataEngine
from aero_ml.network_engine import NetworkEngine
from aero_ml.test_engine import TestEngine


class S2VDataEngine(DataEngine):
    def write_example(self, tfrecord: typing.IO):
        """Write example to tfrecord file

        S2V RNN uses tensors in the input and output so the numpy arrays are
        serialized to bytes list

        Args:
            tfrecord (typing.IO): tfrecord file to write to
        """
        states_input = tf.convert_to_tensor(self.cur_input, dtype=tf.float32)
        states_output = tf.convert_to_tensor(self.cur_output, dtype=tf.float32)

        serialized_input = tf.io.serialize_tensor(states_input)
        serialized_output = tf.io.serialize_tensor(states_output)

        data = {
            "input_state": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[serialized_input.numpy()])
            ),
            "output_state": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[serialized_output.numpy()])
            ),
        }

        example = tf.train.Example(features=tf.train.Features(feature=data))
        tfrecord.write(example.SerializeToString())

    def extract_data(self):
        """
        Extracts state vector from the simulation run and reshapes it to be used as
        input and output for the network.
        """
        # Create lists of runs to vertical stack later
        states = self.sim.state_vec

        # reshapes the input shape to be 2D rather than 1D
        self.cur_input = states[0].reshape(-1, 13)
        # removes the first state from the output
        self.cur_output = states[1:]

        if self.shuffle:
            self.shuffle_data()

    def map_fn(
        self, serialized_example: tf.train.Example
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """_summary_

        Args:
            serialized_example (tf.train.Example): _description_

        Returns:
            tuple[tf.Tensor, tf.Tensor]: _description_
        """
        feature = {
            "input_state": tf.io.FixedLenFeature([], tf.string),
            "output_state": tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(serialized_example, feature)

        input_state = tf.io.parse_tensor(example["input_state"], out_type=tf.float32)
        output_state = tf.io.parse_tensor(example["output_state"], out_type=tf.float32)

        return (input_state, output_state)

    def normalize(
        self, example: tf.Tensor, is_input: bool, invert: bool = False
    ) -> tf.Tensor:
        """Normalizes the given example based on known max values.

        Args:
            example (tf.Tensor): tensor to be normalized
            is_input (bool): whether the tensor is an input or output
            invert (bool, optional): bool to indicate if the tensor
            should be normalized or denormalize. Defaults to False.

        Returns:
            tf.Tensor: normalized or denormalized tensor
        """
        if is_input:
            norm_factors = self.input_norm_factors
        else:
            norm_factors = self.output_norm_factors

        dim = len(norm_factors)
        max_values_tensor = tf.constant(
            norm_factors, shape=(1, 1, dim), dtype=tf.float32
        )
        if invert:
            normalized_example = tf.multiply(example, max_values_tensor)
        else:
            normalized_example = tf.divide(example, max_values_tensor)

        normalized_example = tf.squeeze(normalized_example, axis=0)

        return normalized_example


class S2VNetworkEngine(NetworkEngine):
    pass


class S2VTestEngine(TestEngine):
    pass
