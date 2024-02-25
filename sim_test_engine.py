import os
import random
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from simulation.aircraft_sim import quat_to_euler
from sim_data_engine import SimDataEngine


class SimTestEngine:
    def __init__(
        self,
        attitude_mode="Euler",
        input_normalization_limits=[
            10,
            90,
            90,
            12000,
            1,
            1,
            1,
            1,
            60,
            10,
            20,
            12,
            12,
            12,
        ],
        output_normalization_limits=[90, 90, 12000, 1, 1, 1, 1, 60, 10, 20, 0.5, 1, 1],
    ):

        pio.renderers.default = "browser"
        colors = px.colors.qualitative.Dark24
        self.attitude_mode = attitude_mode
        self.input_normalization_limits = input_normalization_limits
        self.output_normalization_limits = output_normalization_limits

        if self.attitude_mode == "Euler":
            self.columns_input = [
                "t",
                "Lat",
                "Lon",
                "Alt",
                "Phi",
                "Theta",
                "Psi",
                "Vx",
                "Vy",
                "Vz",
                "P",
                "Q",
                "R",
            ]
            self.columns_output = [
                "Lat",
                "Lon",
                "Alt",
                "Phi",
                "Theta",
                "Psi",
                "Vx",
                "Vy",
                "Vz",
                "P",
                "Q",
                "R",
            ]
        else:
            self.columns_input = [
                "t",
                "Lat",
                "Lon",
                "Alt",
                "q0",
                "q1",
                "q2",
                "q3",
                "Vx",
                "Vy",
                "Vz",
                "P",
                "Q",
                "R",
            ]
            self.columns_output = [
                "Lat",
                "Lon",
                "Alt",
                "q0",
                "q1",
                "q2",
                "q3",
                "Vx",
                "Vy",
                "Vz",
                "P",
                "Q",
                "R",
            ]

        col_types = ["_Truth", "_Prediction"]

        self.line_dicts = {}
        self.plot_columns = []
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

    def _convert_quaternions(self, arr, q_start):
        # Ending index of quaternions
        q_end = q_start + 4
        # Euler angle conversion
        euler_angles = np.apply_along_axis(quat_to_euler, 1, arr[0:, q_start:q_end])
        # It is only position if an output array is passed
        time_and_pos = arr[0:, 0:q_start]
        vel_and_body_rates = arr[0:, q_end:]
        # Return new array
        return np.hstack((time_and_pos, euler_angles, vel_and_body_rates))

    def test_model(self, model_to_test, test_dir="test_data", cnt=None):
        files = os.listdir(test_dir)
        if cnt is not None:
            files = random.sample(files, k=cnt)
        for fname in files:
            comp_df = self.process_data(f"{test_dir}/{fname}", model_to_test)
            self.test_plots(comp_df, fname)

    def process_data(self, fname, model_to_test, dir="test_data"):
        # Looping through and building dataset from tfrecords
        dataset = SimDataEngine.load_dataset("Test", dir=dir, fname=fname)
        input_data_list = []
        output_data_list = []
        for item in dataset:
            inp, model_output_arr = item
            input_data_list.append(inp.numpy())
            output_data_list.append(model_output_arr.numpy())
        # Combining all the records into two numpy arrays
        # Test Input
        norm_input_arr = np.vstack(tuple(input_data_list))
        true_input_arr = SimDataEngine.normalize_data(
            norm_input_arr, self.input_normalization_limits, True
        )

        # Test Output
        norm_output_arr = np.vstack(tuple(output_data_list))
        true_output_arr = SimDataEngine.normalize_data(
            norm_output_arr, self.output_normalization_limits, True
        )

        # Running the inputs through the model
        norm_model_output = model_to_test(norm_input_arr)

        # Denormalize out array
        model_output_arr = SimDataEngine.normalize_data(
            norm_model_output, self.output_normalization_limits, invert=True
        )

        # Convert attitude to euler angles if selected
        if self.attitude_mode == "Euler":
            true_input_arr = self._convert_quaternions(true_input_arr.numpy(), 4)
            true_output_arr = self._convert_quaternions(true_output_arr.numpy(), 3)
            model_output_arr = self._convert_quaternions(model_output_arr.numpy(), 3)
        else:
            model_output_arr = model_output_arr.numpy().squeeze()

        # Defining truth and input dataframes
        input_df = pd.DataFrame(true_input_arr, columns=self.columns_input)
        true_output_df = pd.DataFrame(true_output_arr, columns=self.columns_output)
        # Adding time to output array for simplicity
        true_output_df["time"] = input_df["t"]

        # Creating dataframe for output array
        predicted_output_df = pd.DataFrame(
            model_output_arr, columns=self.columns_output
        )
        # Adding time for simplicty
        predicted_output_df["time"] = input_df["t"]

        # Joining the two dataframes for plotting and comparison
        comp_df = true_output_df.join(
            predicted_output_df, lsuffix="_Truth", rsuffix="_Prediction"
        )
        comp_df = comp_df.reindex(sorted(comp_df.columns), axis=1)
        comp_df = comp_df.drop(columns=["time_Prediction"])

        # Return dataframe for plotting
        return comp_df

    def test_plots(self, comp_df, title="test", height=6000):
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
                        x=comp_df["time_Truth"],
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
        fig.show()
