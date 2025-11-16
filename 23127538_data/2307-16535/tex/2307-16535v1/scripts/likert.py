import plot_likert
import pandas as pd
import numpy as np

# data = pd.read_csv('workshop1likertdata.csv', dtype = str)
data = pd.read_csv('workshop2likertdata.csv', dtype = str)
print(data)

# plot_likert.plot_likert(data, plot_likert.scales.raw7)

axes = plot_likert.plot_likert(data, plot_likert.scales.raw7, colors=plot_likert.colors.likert7)
handles, labels = axes.get_legend_handles_labels()
labels = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree', 'Strongly Agree']
by_label = dict(zip(labels, handles))
print(by_label)
axes.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
# axes.get_figure().savefig('ws1likertplot.png', dpi=600, bbox_inches='tight')
axes.get_figure().savefig('ws2likertplot.png', dpi=600, bbox_inches='tight')