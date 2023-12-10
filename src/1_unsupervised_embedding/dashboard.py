'''A module for plotting model statistics and embedding visualizations.'''''

from collections import namedtuple
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
import plotly.subplots

PlotLineParameters = namedtuple('PlotLineParameters', ['y', 'group_name', 'color', 'has_dash'])

class TransformerDashboard(): # pylint: disable=too-many-instance-attributes, too-many-public-methods
    """A class for plotting model statistics and diagnostics."""

    def __init__(self, titles, loss_step_size : int, training_step_size : int, validation_step_size : int): # pylint: disable=line-too-long
        '''Initializes the dashboard.
        
        Args:
            titles (list): The titles of the subplots.
            loss_step_size (int): The step size of the loss plot.
            training_step_size (int): The step size of the training accuracy plot.
            validation_step_size (int): The step size of the validation accuracy plot.
        '''

        self.losses = {}
        self.train_accuracies = {}
        self.val_a_accuracies = {}
        self.val_b_accuracies = {}
        self.embeddings = None

        self.embedding_names = None

        self.embedding_mask_a = None
        self.embedding_mask_b = None
        self.embedding_mask_c = None
        self.embedding_mask_d = None

        self.embedding_names_a = None
        self.embedding_names_b = None
        self.embedding_names_c = None
        self.embedding_names_d = None

        self.loss_y_range = None
        self.train_y_range = None
        self.validation_a_y_range = None
        self.validation_b_y_range = None

        self.titles = titles
        self.loss_step_size = loss_step_size
        self.training_step_size = training_step_size
        self.validation_step_size = validation_step_size

    def register_loss(self, name : str, color : str, has_dash : bool):                     self.losses[name] = PlotLineParameters([], name, color, has_dash) # pylint: disable=missing-function-docstring, multiple-statements, line-too-long
    def register_train_accuracies(self, name : str, color : str, has_dash : bool):         self.train_accuracies[name]	= PlotLineParameters([], name, color, has_dash) # pylint: disable=missing-function-docstring, multiple-statements, line-too-long
    def register_validation_a_accuracies(self, name : str, color : str, has_dash : bool):  self.val_a_accuracies[name]	= PlotLineParameters([], name, color, has_dash) # pylint: disable=missing-function-docstring, multiple-statements, line-too-long
    def register_validation_b_accuracies(self, name : str, color : str, has_dash : bool):  self.val_b_accuracies[name]	= PlotLineParameters([], name, color, has_dash) # pylint: disable=missing-function-docstring, multiple-statements, line-too-long

    def register_embedding_color_mask_a(self, color_mask): self.embedding_mask_a = color_mask # pylint: disable=missing-function-docstring, multiple-statements
    def register_embedding_color_mask_b(self, color_mask): self.embedding_mask_b = color_mask # pylint: disable=missing-function-docstring, multiple-statements
    def register_embedding_color_mask_c(self, color_mask): self.embedding_mask_c = color_mask # pylint: disable=missing-function-docstring, multiple-statements
    def register_embedding_color_mask_d(self, color_mask): self.embedding_mask_d = color_mask # pylint: disable=missing-function-docstring, multiple-statements

    def register_embedding_names_a(self, embedding_names): self.embedding_names_a = embedding_names # pylint: disable=missing-function-docstring, multiple-statements
    def register_embedding_names_b(self, embedding_names): self.embedding_names_b = embedding_names # pylint: disable=missing-function-docstring, multiple-statements
    def register_embedding_names_c(self, embedding_names): self.embedding_names_c = embedding_names # pylint: disable=missing-function-docstring, multiple-statements
    def register_embedding_names_d(self, embedding_names): self.embedding_names_d = embedding_names # pylint: disable=missing-function-docstring, multiple-statements

    def update_loss(self, value, name):                    self.losses[name].y.append(value) # pylint: disable=missing-function-docstring, multiple-statements
    def update_train_accuracy(self, value, name):          self.train_accuracies[name].y.append(value) # pylint: disable=missing-function-docstring, multiple-statements, line-too-long
    def update_validation_a_accuracy(self, value, name):   self.val_a_accuracies[name].y.append(value) # pylint: disable=missing-function-docstring, multiple-statements, line-too-long
    def update_validation_b_accuracy(self, value, name):   self.val_b_accuracies[name].y.append(value) # pylint: disable=missing-function-docstring, multiple-statements, line-too-long
    def update_embeddings(self, value):                    self.embeddings = value # pylint: disable=missing-function-docstring, multiple-statements

    def set_loss_y_range(self, value):                      self.loss_y_range = value # pylint: disable=missing-function-docstring, multiple-statements
    def set_train_y_range(self, value):                     self.train_y_range = value # pylint: disable=missing-function-docstring, multiple-statements
    def set_validation_a_y_range(self, value):              self.validation_a_y_range = value # pylint: disable=missing-function-docstring, multiple-statements
    def set_validation_b_y_range(self, value):              self.validation_b_y_range = value # pylint: disable=missing-function-docstring, multiple-statements

    def plot(self, filename : str):
        '''Plots the model statistics and diagnostics and saves them to a HTML file.
        
        Args:
            filename (str): The name of the HTML file to save the plots to.
        '''

        fig = self._init_figure()

        self._plot_loss(fig)
        self._plot_train_accuracy(fig)
        self._plot_validation_a_accuracy(fig)
        self._plot_validation_b_accuracy(fig)

        self._plot_embeddings(fig)

        fig.update_layout(showlegend=False, font=dict(family='Roboto, sans-serif', color='#3f3f3f'), plot_bgcolor='white', bargap=0.1) # pylint: disable=line-too-long
        fig.write_html(filename, auto_open=False)

    def _init_figure(self) -> go.Figure:
        '''Initializes the multiplot figure.
        
        Returns:
            plotly.graph_objects.Figure: The initialized figure.
        '''

        fig = plotly.subplots.make_subplots(
            rows=2, cols=4, 
            subplot_titles=(
                self.titles
            )
        )

        return fig

    def _plot_loss(self, fig : go.Figure):

        for plot_line_parameters in self.losses.values():
            self._apply_line_plot(
                plot_line_parameters.y,
                self.loss_step_size,
                plot_line_parameters.has_dash,
                plot_line_parameters.group_name,
                plot_line_parameters.color,
                fig, row=1, col=1
            )

        fig.update_xaxes(row=1, col=1, gridcolor='#ebebeb', title_text='<b>Steps</b>', showline=True, linewidth=1.5, linecolor='#303030')
        fig.update_yaxes(row=1, col=1, gridcolor='#ebebeb', title_text='<b>Loss</b>', range=[0, self.loss_y_range] )

    def _plot_train_accuracy(self, fig : go.Figure):

        for plot_line_parameters in self.train_accuracies.values():
            self._apply_line_plot(
                np.hstack(plot_line_parameters.y),
                self.training_step_size,
                plot_line_parameters.has_dash,
                plot_line_parameters.group_name,
                plot_line_parameters.color,
                fig, row=1, col=2
            )

        fig.update_xaxes(row=1, col=2, gridcolor='#ebebeb', title_text='<b>Steps</b>', showline=True, linewidth=1.5, linecolor='#303030') # pylint: disable=line-too-long
        fig.update_yaxes(row=1, col=2, gridcolor='#ebebeb', title_text='<b>Accuracy</b>', range=[0, self.train_y_range] ) # pylint: disable=line-too-long

    def _plot_validation_a_accuracy(self, fig : go.Figure):
        '''Plots the validation A accuracy into the multiplot figure (row=1, col=3).

        Args:
            fig (plotly.graph_objects.Figure): The figure to add the validation A accuracy to.
        '''

        for plot_line_parameters in self.val_a_accuracies.values():
            self._apply_line_plot(
                np.hstack(plot_line_parameters.y),
                self.validation_step_size,
                plot_line_parameters.has_dash,
                plot_line_parameters.group_name,
                plot_line_parameters.color,
                fig,
                row=1,
                col=3
            )

        fig.update_xaxes(row=1, col=3, gridcolor='#ebebeb', title_text='<b>Epoch</b>', showline=True, linewidth=1.5, linecolor='#303030') # pylint: disable=line-too-long
        fig.update_yaxes(row=1, col=3, gridcolor='#ebebeb', title_text='<b></b>', range=[0, self.validation_a_y_range] ) # pylint: disable=line-too-long

    def _plot_validation_b_accuracy(self, fig : go.Figure):
        '''Plots the validation B accuracy into the multiplot figure (row=1, col=4).

        Args:
            fig (plotly.graph_objects.Figure): The figure to add the validation B accuracy to.
        '''

        for plot_line_parameters in self.val_b_accuracies.values():
            self._apply_line_plot(
                np.hstack(plot_line_parameters.y),
                self.validation_step_size,
                plot_line_parameters.has_dash,
                plot_line_parameters.group_name,
                plot_line_parameters.color,
                fig,
                row=1,
                col=4
            )

        fig.update_xaxes(row=1, col=4, gridcolor='#ebebeb', title_text='<b>Epoch</b>', showline=True, linewidth=1.5, linecolor='#303030') # pylint: disable=line-too-long
        fig.update_yaxes(row=1, col=4, gridcolor='#ebebeb', title_text='<b></b>', range=[0, self.validation_b_y_range]) # pylint: disable=line-too-long

    def _plot_embeddings(self, fig : go.Figure):
        '''Plots the embeddings into the multiplot figure.
        
        Args:
            fig (plotly.graph_objects.Figure): The figure to add the embeddings to.
        '''

        for (row, col, mask, text) in zip(
            [2, 2, 2, 2],
            [1, 2, 3, 4],
            [self.embedding_mask_a, self.embedding_mask_b, self.embedding_mask_c, self.embedding_mask_d], # pylint: disable=line-too-long
            [self.embedding_names_a, self.embedding_names_b, self.embedding_names_c, self.embedding_names_d] # pylint: disable=line-too-long
        ):

            colors = np.unique(mask)

            for color in colors:
                segment_mask 	= mask == color
                x 				= self.embeddings[segment_mask, 0]
                y 				= self.embeddings[segment_mask, 1]
                labels 			= text[segment_mask]

                self._apply_scatter_plot(x, y, labels, color, color, fig, row=row, col=col)

            fig.update_xaxes(row=row, col=col, gridcolor='#ebebeb', title_text='<b>SNE 1</b>')
            fig.update_yaxes(row=row, col=col, gridcolor='#ebebeb', title_text='<b>SNE 2</b>')

    @staticmethod
    def _apply_line_plot(y, step_size : int, has_dash : bool, group_name : str, color : str, fig : go.Figure, row : int, col : int): # pylint: disable=too-many-arguments, line-too-long
        '''Creates a line plot and adds it to a figure.

        Args:
            y (np.ndarray): The y-coordinates of the line plot.
            step_size (int): The step size of the line plot.
            has_dash (bool): Whether the line plot should be dashed.
            group_name (str): The name of the group.
            color (str): The color of the group.
            fig (plotly.graph_objects.Figure): The figure to add the line plot to.
            row (int): The row of the figure to add the line plot to.
            col (int): The column of the figure to add the line plot to.
        '''

        x = np.arange(len(y)) * step_size

        fig.add_trace(
            go.Scattergl(
                x=x, y=y,
                name=group_name,
                line={'color' : color, 'width' : 4, 'dash' : 'dot' if has_dash is True else None},
                mode='lines',
            ),
            row=row, col=col
        )

    @staticmethod
    def _apply_scatter_plot(x : np.ndarray, y : np.ndarray, names : Sequence, group_name : str, color : str, fig : go.Figure, row : int, col : int): # pylint: disable=too-many-arguments, line-too-long
        '''Creates a scatter plot and adds it to a figure.
        
        Args:
            x (np.ndarray): The x-coordinates of the scatter plot.
            y (np.ndarray): The y-coordinates of the scatter plot.
            names (np.ndarray): The names of the scatter plot.
            group_name (str): The name of the group.
            color (str): The color of the group.
            fig (plotly.graph_objects.Figure): The figure to add the scatter plot to.
            row (int): The row of the figure to add the scatter plot to.
            col (int): The column of the figure to add the scatter plot to.
        '''

        fig.add_trace(
            go.Scattergl(
                x=x, y=y,
                text=names,
                name=group_name,
                marker={'opacity' : 0.25, 'color' : color},
                mode='markers',
            ), row=row, col=col
        )
