"""A module for plotting model statistics and diagnostics."""
from collections import namedtuple

import numpy as np

from plotly.subplots import make_subplots
import plotly.graph_objects as go

PlotLineParameters = namedtuple('PlotLineParameters', ['y', 'group_name', 'color', 'has_dash'])

class TransformerDashboard():
	"""A class for plotting model statistics and diagnostics."""

	def __init__(self, titles, loss_step_size, training_step_size, validation_step_size):

		self.losses = {};
		self.train_accuracies = {};
		self.val_a_accuracies = {};
		self.val_b_accuracies = {};
		self.embeddings = None;

		self.embedding_names = None;
		
		self.embedding_a_mask = None;
		self.embedding_b_mask = None;
		self.embedding_c_mask = None;
		self.embedding_d_mask = None;

		self.loss_y_range = None;
		self.train_y_range = None;
		self.validation_a_y_range = None;
		self.validation_b_y_range = None;

		self.titles = titles;
		self.loss_step_size = loss_step_size;
		self.training_step_size = training_step_size;
		self.validation_step_size = validation_step_size;

	def register_loss(self, name, color, has_dash): 					
		self.losses[name] = PlotLineParameters([], name, color, has_dash);

	def register_train_accuracies(self, name, color, has_dash):			
		self.train_accuracies[name]	= PlotLineParameters([], name, color, has_dash);

	def register_validation_a_accuracies(self, name, color, has_dash):	
		self.val_a_accuracies[name]	= PlotLineParameters([], name, color, has_dash);

	def register_validation_b_accuracies(self, name, color, has_dash):	
		self.val_b_accuracies[name]	= PlotLineParameters([], name, color, has_dash);

	def register_embedding_color_mask_a(self, color_mask):
		self.embedding_mask_a = color_mask;
	
	def register_embedding_color_mask_b(self, color_mask):
		self.embedding_mask_b = color_mask;
	
	def register_embedding_color_mask_c(self, color_mask):
		self.embedding_mask_c = color_mask;
	
	def register_embedding_color_mask_d(self, color_mask):
		self.embedding_mask_d = color_mask;

	def register_embedding_names_a(self, embedding_names):
		self.embedding_names_a = embedding_names;

	def register_embedding_names_b(self, embedding_names):
		self.embedding_names_b = embedding_names;

	def register_embedding_names_c(self, embedding_names):
		self.embedding_names_c = embedding_names;

	def register_embedding_names_d(self, embedding_names):
		self.embedding_names_d = embedding_names;


  def update_loss(self, value, name): 								self.losses[name].y.append(value);
  def update_train_accuracy(self, value, name):						self.train_accuracies[name].y.append(value);
  def update_validation_a_accuracy(self, value, name):				self.val_a_accuracies[name].y.append(value);
  def update_validation_b_accuracy(self, value, name):				self.val_b_accuracies[name].y.append(value);
  def update_embeddings(self, value): 								self.embeddings = value;

  def set_loss_y_range(self, value):									self.loss_y_range = value;
  def set_train_y_range(self, value):									self.train_y_range = value;
  def set_validation_a_y_range(self, value):							self.validation_a_y_range = value;
  def set_validation_b_y_range(self, value):							self.validation_b_y_range = value;

  def plot(self, filename):

	  fig = self.init_figure();
	  
	  self.plot_loss(fig);
	  self.plot_train_accuracy(fig);
	  self.plot_validation_a_accuracy(fig);
	  self.plot_validation_b_accuracy(fig);

	  self.plot_embeddings(fig);

	  fig.update_layout(showlegend=False, font=dict(family='Roboto, sans-serif', color='#3f3f3f'), plot_bgcolor='white', bargap=0.1);
	  fig.write_html(filename, auto_open=False);

	  return;

  def init_figure(self):
	  fig = make_subplots(
		  rows=2, cols=4, 
		  subplot_titles=(
			  self.titles
		  )
	  );

	  return fig;

  def plot_loss(self, fig):

	  for plot_line_parameters in self.losses.values():
		  self.apply_line_plot(plot_line_parameters.y, self.loss_step_size, plot_line_parameters.has_dash, plot_line_parameters.group_name, plot_line_parameters.color, fig, row=1, col=1);

	  fig.update_xaxes(row=1, col=1, gridcolor='#ebebeb', title_text='<b>Steps</b>', showline=True, linewidth=1.5, linecolor='#303030');
	  fig.update_yaxes(row=1, col=1, gridcolor='#ebebeb', title_text='<b>Loss</b>', range=[0, self.loss_y_range] );

	  return;

  def plot_train_accuracy(self, fig):

	  for plot_line_parameters in self.train_accuracies.values():

		  self.apply_line_plot(np.hstack(plot_line_parameters.y), self.training_step_size, plot_line_parameters.has_dash, plot_line_parameters.group_name, plot_line_parameters.color, fig, row=1, col=2);

	  fig.update_xaxes(row=1, col=2, gridcolor='#ebebeb', title_text='<b>Steps</b>', showline=True, linewidth=1.5, linecolor='#303030');
	  fig.update_yaxes(row=1, col=2, gridcolor='#ebebeb', title_text='<b>Accuracy</b>', range=[0, self.train_y_range] );

	  return;

  def plot_validation_a_accuracy(self, fig):

	  for plot_line_parameters in self.val_a_accuracies.values():
		  self.apply_line_plot(np.hstack(plot_line_parameters.y), self.validation_step_size, plot_line_parameters.has_dash, plot_line_parameters.group_name, plot_line_parameters.color, fig, row=1, col=3);

	  fig.update_xaxes(row=1, col=3, gridcolor='#ebebeb', title_text='<b>Epoch</b>', showline=True, linewidth=1.5, linecolor='#303030');
	  fig.update_yaxes(row=1, col=3, gridcolor='#ebebeb', title_text='<b></b>', range=[0, self.validation_a_y_range] );

	  return;

  def plot_validation_b_accuracy(self, fig):

	  for plot_line_parameters in self.val_b_accuracies.values():
		  self.apply_line_plot(np.hstack(plot_line_parameters.y), self.validation_step_size, plot_line_parameters.has_dash, plot_line_parameters.group_name, plot_line_parameters.color, fig, row=1, col=4);

	  fig.update_xaxes(row=1, col=4, gridcolor='#ebebeb', title_text='<b>Epoch</b>', showline=True, linewidth=1.5, linecolor='#303030');
	  fig.update_yaxes(row=1, col=4, gridcolor='#ebebeb', title_text='<b></b>', range=[0, self.validation_b_y_range] );

	  return;

  def plot_embeddings(self, fig):

	  for (row, col, mask, text) in zip(
		  [2, 2, 2, 2], 
		  [1, 2, 3, 4], 
		  [self.embedding_mask_a, self.embedding_mask_b, self.embedding_mask_c, self.embedding_mask_d],
		  [self.embedding_names_a, self.embedding_names_b, self.embedding_names_c, self.embedding_names_d]
	  ):

		  colors = np.unique(mask);

		  for color in colors:
			  segment_mask 	= mask == color;
			  x 				= self.embeddings[segment_mask, 0];
			  y 				= self.embeddings[segment_mask, 1];
			  labels 			= text[segment_mask];

			  self.apply_scatter_plot(x, y, labels, color, color, fig, row=row, col=col);

		  fig.update_xaxes(row=row, col=col, gridcolor='#ebebeb', title_text='<b>SNE 1</b>', );
		  fig.update_yaxes(row=row, col=col, gridcolor='#ebebeb', title_text='<b>SNE 2</b>', );

	  return;

  def apply_line_plot(self, y, step_size, has_dash, group_name, color, fig, row, col):

	  x = np.arange(len(y)) * step_size;
	  y = y;

	  fig.add_trace(
		  go.Scattergl(
			  x=x, y=y, 
			  name=group_name, 
			  line=dict(color=color, width=4, dash='dot' if has_dash == True else None),
			  mode='lines', 
		  ), 
		  row=row, col=col
	  );

	  return;

  def apply_scatter_plot(self, x, y, names, group_name, color, fig, row, col):

	  fig.add_trace(
		  go.Scattergl(
			  x=x, y=y, 
			  text=names,
			  name=group_name, 
			  marker=dict(opacity=0.25, color=color),
			  mode='markers', 
		  ), 
		  row=row, col=col
	  );

	  return;
