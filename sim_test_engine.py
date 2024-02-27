import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from simulation.aircraft_sim import quat_to_euler
from sim_data_engine import SimDataEngine


class SimTestEngine:
    def __init__(self, config, att_mode="Euler", project_name="Test"):
        colors = px.colors.qualitative.Dark24
        self.project_name = project_name
        self.attitude_mode = att_mode
        self.input_norm_factors = config["DEFAULTS"]["INPUT_NORM_FACTORS"]
        self.output_norm_factors = config["DEFAULTS"]["OUTPUT_NORM_FACTORS"]

        if self.attitude_mode == "Euler":
            self.columns_input = config["DEFAULTS"]["INPUT_COLUMNS_EULER"]
            self.columns_output = config["DEFAULTS"]["OUTPUT_COLUMNS_EULER"]
        else:
            self.columns_input = config["DEFAULTS"]["INPUT_COLUMNS"]
            self.columns_output = config["DEFAULTS"]["OUTPUT_COLUMNS"]

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
        self.date_str = datetime.now().strftime("%m%d%Y_%H%M%S")

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
            self.test_plots(comp_df, fname, test_dir)

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
            norm_input_arr, self.input_norm_factors, True
        )

        # Test Output
        norm_output_arr = np.vstack(tuple(output_data_list))
        true_output_arr = SimDataEngine.normalize_data(
            norm_output_arr, self.output_norm_factors, True
        )

        # Running the inputs through the model
        norm_model_output = model_to_test(norm_input_arr)

        # Denormalize out array
        model_output_arr = SimDataEngine.normalize_data(
            norm_model_output, self.output_norm_factors, invert=True
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

    def test_plots(
        self, comp_df, title="test", plot_dir=f"test_data_plots", height=6000
    ):
        plot_dir = f"{self.date_str}_{plot_dir}"
        os.makedirs(plot_dir, exist_ok=True)
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
        fig.write_html(f"{plot_dir}/{title}.html")
