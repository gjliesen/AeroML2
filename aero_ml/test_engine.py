import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
from plotly.subplots import make_subplots
from simulation.aircraft_sim import quat_to_euler


class TestEngine:
    date_str: str
    input_features_euler: list
    output_features_euler: list
    input_features: list
    output_features: list

    def __init__(
        self,
        data_eng,
        att_mode: str = "Euler",
    ):
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")
        self.data_eng = data_eng
        # dynamically etting the configuration attributes
        self.attitude_mode = att_mode
        self.columns_input = []
        self.columns_output = []
        self._set_columns()

        self.plot_columns = []
        self.line_dicts = {}
        self._set_figure_params()

    def _set_columns(self):
        """Set the input and output columns based on the attitude mode."""
        if self.attitude_mode == "Euler":
            self.columns_input = self.input_features_euler
            self.columns_output = self.output_features_euler
        else:
            self.columns_input = self.input_features
            self.columns_output = self.output_features

    def _set_figure_params(self):
        col_types = ["_Truth", "_Prediction"]
        colors = px.colors.qualitative.Dark24
        for i, col in enumerate(self.columns_output):
            subplot = []
            for col_type in col_types:
                full_name = f"{col}{col_type}"
                subplot.append(full_name)
                if col_type == "_Truth":
                    self.line_dicts[full_name] = dict(color=colors[i])
                else:
                    self.line_dicts[full_name] = dict(color=colors[i], dash="dash")
            self.plot_columns.append(subplot)

    def test_model(self, model_to_test, data_dir: str):
        """Test the model on the test data and plot the results.

        Args:
            model_to_test (_type_): tensorflow model to be tested
            data_dir (str): directory containing the test data
        """
        # Defining the directories
        test_dir = f"{data_dir}/test"
        plot_dir = f"test_data_plots/{data_dir}"
        os.makedirs(plot_dir, exist_ok=True)

        # Placing all test data records into a list
        files = os.listdir(test_dir)

        # Looping through test_data and plotting
        for fname in files:
            # Define test record file name, joining the path
            tfrecord_path = f"{test_dir}/{fname}"
            # Creating a comparison dataframe to plot
            comp_df = self._process_data(tfrecord_path, model_to_test)
            # Plotting comparison dataframe
            self.test_plots(comp_df, fname, plot_dir)

    def _process_data(self, fname: str, model_to_test) -> pd.DataFrame:
        """Process the test data and return a dataframe for plotting.

        Args:
            fname (str): file name of the test data
            model_to_test (_type_): tensorflow model to be tested

        Returns:
            pd.DataFrame: dataframe of the test and predicted data used for plotting
        """
        # Looping through and building dataset from tfrecords
        dataset, _ = self.data_eng.load_dataset(fname=fname)
        # Extracting the normalized input and output array from the datset
        norm_input_arr, norm_output_arr = self._extract_data(dataset)

        # Running the inputs through the model
        norm_model_output = model_to_test(norm_input_arr)

        norm_arrays = [norm_input_arr, norm_output_arr, norm_model_output]
        denorm_arrays = self._denormalize_data(norm_arrays)
        if self.attitude_mode == "Euler":
            denorm_arrays = self._convert_attitude(denorm_arrays)

        return self._create_dataframe(denorm_arrays)

    def _extract_data(self, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Extract the input and output data from the dataset.

        Args:
            dataset (tf.data.Dataset): dataset containing the test data

        Returns:
            tuple[np.ndarray, np.ndarray]: tuple of the normalized input and output data
            arrays
        """
        input_data_list = []
        output_data_list = []
        for inp, outp in dataset:  # ignore: not iterable
            input_data_list.append(inp.numpy())
            output_data_list.append(outp.numpy())
            # Combining all the records into two numpy arrays
        # Test Input
        norm_input_arr = np.vstack(tuple(input_data_list))
        # Test Output
        norm_output_arr = np.vstack(tuple(output_data_list))

        return (norm_input_arr, norm_output_arr)

    def _denormalize_data(self, norm_arrays: list[np.ndarray]) -> list[np.ndarray]:
        """Denormalize the input and output arrays.

        Args:
            norm_arrays (list[np.ndarray]): list of the normalized input and output
            arrays

        Returns:
            list[np.ndarray]: list of the denormalized input and output arrays
        """
        is_inputs = [True, False, False]
        denorm_arrays = []
        for arr, is_input in zip(norm_arrays, is_inputs):
            denorm_arrays.append(
                (
                    self.data_eng.normalize(arr, is_input=is_input, invert=True)
                    .numpy()
                    .squeeze()
                )
            )
        return denorm_arrays

    def _convert_attitude(self, denorm_arrays: list[np.ndarray]) -> list[np.ndarray]:
        """wrapper function for _convert_quaternions that loops through the arrays that
        need to be converted.

        Args:
            denorm_arrays (list[np.ndarray]): list of the denormalized input and output
            arrays

        Returns:
            list[np.ndarray]: list of the denormalized input and output arrays with
            euler angles rather than quaternions
        """
        # Convert attitude to euler angles if selected
        quat_indices = [4, 3, 3]
        euler_arrays = []
        for arr, quat_idx in zip(denorm_arrays, quat_indices):
            euler_arrays.append(self._convert_quaternions(arr, quat_idx))
        return euler_arrays

    def _convert_quaternions(self, arr: np.ndarray, q_start: int) -> np.ndarray:
        """Applies the quaternion to euler conversion to the given array and returns an
        array of euler angles.

        Args:
            arr (np.ndarray): array containing the quaternions
            q_start (int): starting index of the quaternions

        Returns:
            np.ndarray: array containing the euler angles
        """
        # Ending index of quaternions
        q_end = q_start + 4
        # Euler angle conversion
        euler_angles = np.apply_along_axis(quat_to_euler, 1, arr[0:, q_start:q_end])
        # It is only position if an output array is passed
        time_and_pos = arr[0:, 0:q_start]
        vel_and_body_rates = arr[0:, q_end:]
        # Return new array
        return np.hstack((time_and_pos, euler_angles, vel_and_body_rates))

    def _create_dataframe(self, denorm_arrays: list[np.ndarray]) -> pd.DataFrame:
        """Create a dataframe for plotting from the input, output, and predicted arrays.

        Args:
            denorm_arrays (list[np.ndarray]): list of the denormalized input and output

        Returns:
            pd.DataFrame: dataframe of the input, output, and predicted arrays
        """
        [true_input_arr, true_output_arr, pred_output_arr] = denorm_arrays
        # Defining truth and input dataframes
        input_df = pd.DataFrame(true_input_arr, columns=self.columns_input)

        true_output_df = pd.DataFrame(true_output_arr, columns=self.columns_output)
        # Adding time to output array for simplicity
        true_output_df["Time"] = input_df["Time"]

        # Creating dataframe for output array
        pred_output_df = pd.DataFrame(pred_output_arr, columns=self.columns_output)

        # Adding time for simplicty
        pred_output_df["Time"] = input_df["Time"]

        # Joining the two dataframes for plotting and comparison
        comp_df = true_output_df.join(
            pred_output_df, lsuffix="_Truth", rsuffix="_Prediction"
        )
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
        comp_df = comp_df.drop(columns=["Time_Prediction"])

        # Return dataframe for plotting
        return comp_df

    def test_plots(
        self, comp_df: pd.DataFrame, title: str, plot_dir: str, height: int = 6000
    ):
        """Plot the test data and the model predictions.

        Args:
            comp_df (pd.DataFrame): dataframe of the test and predicted data
            title (str): title of the plot
            plot_dir (str): directory to save the plot
            height (int, optional): height of the plot in pixels. Defaults to 6000.
        """
        # Initializing subplot
        fig = make_subplots(
            rows=len(self.plot_columns),
            cols=1,
            shared_xaxes=False,
            subplot_titles=tuple(self.columns_output),
        )

        # Looping through the column sets to plot
        for row, col_list in enumerate(self.plot_columns):
            for col in col_list:
                fig.add_trace(
                    go.Scatter(
                        x=comp_df["Time_Truth"],
                        y=comp_df[col],
                        name=col,
                        line=self.line_dicts[col],
                        legendgroup=str(row + 1),
                    ),
                    row=row + 1,
                    col=1,
                )
        # Displaying the figure
        fig.update_layout(hovermode="x unified", title=title, height=height)
        fig.write_html(f"{plot_dir}/{title}.html")
