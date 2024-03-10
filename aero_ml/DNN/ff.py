import typing
import numpy as np
import pandas as pd
import tensorflow as tf
from aero_ml.data_engine import DataEngine
from aero_ml.network_engine import NetworkEngine
from aero_ml.test_engine import TestEngine


class FFDataEngine(DataEngine):
    def write_example(self, tfrecord: typing.IO):
        """Write example to tfrecord file

        FF uses lists as inputs and outputs so the inputs and outputs are just
        Args:
            tfrecord (typing.IO): tfrecord file to write to
        """
        for input_state, output_state in zip(self.cur_input, self.cur_output):
            data = {
                "input_state": tf.train.Feature(
                    float_list=tf.train.FloatList(value=input_state)
                ),
                "output_state": tf.train.Feature(
                    float_list=tf.train.FloatList(value=output_state)
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
        self.cur_input = self.concatInputs(states[0]).squeeze()
        self.cur_output = states
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
            "input_state": tf.io.FixedLenFeature([self.input_dim], tf.float32),
            "output_state": tf.io.FixedLenFeature([self.output_dim], tf.float32),
        }
        example = tf.io.parse_single_example(serialized_example, feature)
        return example["input_state"], example["output_state"]

    def normalize(
        self, example: tf.train.Example, is_input: bool, invert: bool = False
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

        normalized_example = tf.squeeze(normalized_example)

        return normalized_example


class FFNetworkEngine(NetworkEngine):
    pass


class FFTestEngine(TestEngine):
    pass
