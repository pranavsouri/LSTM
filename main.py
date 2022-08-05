import pandas as pd
import statistics as stats
import numpy as np

pattern_range = 10 # minimum number of data points that defines a trend
max_trend_points = 150 # maximum mnumber of data points in a trend
current_index = 0
x_axis = 'TEST_PERIOD_FORMATTED'
y_axis = 'ATT_TP_DL_MV'
trend_type = "TREND_TYPE"
test_count = "ATT_TEST_COUNT_MV"
file_name = 'amCharts Los Angeles.csv'

data = pd.read_csv(file_name)
df = pd.DataFrame()
df[x_axis] = pd.to_datetime(data[x_axis])
df[y_axis] = data[y_axis]
df[test_count] = data[test_count]
df = df.dropna().reset_index(drop=True)
df[trend_type] = np.nan
df.head()

current_trend = [] # list of datapoints part of current trend
current_tests = [] # list of current test cases in trend

# for displaying on graph
anomalies = pd.DataFrame(index=range(df.shape[0]),columns=[x_axis, y_axis]) # high importance anomalies
anomalies[x_axis] = df[x_axis]
anomalies_low = pd.DataFrame(index=range(df.shape[0]),columns=[x_axis, y_axis]) # low importance anomalies
anomalies_low[x_axis] = df[x_axis]
shifts = pd.DataFrame(index=range(df.shape[0]),columns=[x_axis, y_axis])
shifts[x_axis] = df[x_axis]
trends = pd.DataFrame(index=range(df.shape[0]),columns=[x_axis, y_axis])
trends[x_axis] = df[x_axis]

# builds initial trend from pattern_range number of datapoints
for current_index in range(pattern_range):
  current_trend.append(df[y_axis][current_index])
  current_tests.append(df[test_count][current_index])

for current_index in range(pattern_range - 1, len(df[y_axis])):
    datapoint = df[y_axis][current_index]
    old_mean = stats.mean(current_trend)
    strdev = stats.stdev(current_trend, old_mean)

    test = df[test_count][current_index]
    old_mean_tests = stats.mean(current_tests)
    strdev_tests = stats.stdev(current_tests, old_mean_tests)

    # for trends data points on graph
    if (current_index + 1) % (pattern_range / 2) == 0:
        trends.iat[current_index, 1] = old_mean

    # keeps the current trend list with recent points determined by max_trend_points value
    if len(current_trend) >= max_trend_points:
        current_trend.pop(0)
        current_tests.pop(0)

    # check if new datapoint is anomaly / trend (3 stdev out)
    if datapoint >= strdev * 3 + old_mean or datapoint <= old_mean - strdev * 3:
        temp_trend = []
        temp_trend_tests = []
        for i in range(pattern_range):
            if (current_index + i < len(df[y_axis])):
                temp_trend.append(df[y_axis][current_index + i])
                temp_trend_tests.append(df[test_count][current_index + i])
        new_mean = stats.mean(temp_trend)
        new_mean_tests = stats.mean(temp_trend_tests)

        if len(temp_trend) == pattern_range and (
                new_mean >= strdev * 3 + old_mean or new_mean <= old_mean - strdev * 3):  # trend shift
            print("index: " + str(current_index) + " shift: " + str(df[x_axis][current_index]))
            if old_mean < new_mean:
                df[trend_type][current_index] = "shift up"
            elif old_mean > new_mean:
                df[trend_type][current_index] = "shift down"

            shifts.iat[current_index, 1] = datapoint
            current_trend.clear()
            current_trend.extend(temp_trend)

            current_tests.clear()
            current_tests.extend(temp_trend_tests)

            current_index += pattern_range
        else:  # anomaly
            print("index: " + str(current_index) + " anomaly: " + str(df[x_axis][current_index]))
            if test >= strdev_tests * 2 + old_mean_tests or test <= old_mean_tests - strdev_tests * 2:  # high importance anomaly
                anomalies.iat[current_index, 1] = datapoint
                df[trend_type][current_index] = "high anomaly"
            else:
                anomalies_low.iat[current_index, 1] = datapoint
                df[trend_type][current_index] = "low anomaly"
    else:
        current_trend.append(datapoint)
        current_tests.append(test)

def returnName(mean, first_mean, leniency): # returns trend name based on given means
  if mean + mean * leniency >= first_mean and mean - mean * leniency <= first_mean:
    return "flat"
  elif mean > first_mean:
    return "up"
  elif mean < first_mean:
    return "down"

# trend identification
first_mean = df[y_axis][0]
i = pattern_range/2 - 1
leniency = 0.001 # leniency for considering means to be equal, 0.1%
name = ""
while i + pattern_range/2 <= len(trends[y_axis]):
  mean = trends[y_axis][i]
  name = returnName(mean, first_mean, leniency)
  j = 0
  if not pd.isnull(mean):
    j = i - (pattern_range/2 - 1)
  else:
    i += pattern_range/2
    mean = trends[y_axis][i]
    name = returnName(mean, first_mean, leniency)
  while j <= i:
    val = df[trend_type][j]
    if pd.isnull(val):
      df[trend_type][j] = name
    j += 1

  first_mean = mean
  i += pattern_range/2

# trend identification for last values
if j < len(trends[y_axis]):
  mean = df[y_axis][len(df[y_axis]) - 1]
  name = returnName(mean, first_mean, leniency)
  while j < len(trends[y_axis]):
    val = df[trend_type][j]
    if pd.isnull(val):
      df[trend_type][j] = name
    j += 1

scale = 0
for i in range(len(df[y_axis])):
  print(str(df[x_axis][i + scale]) + " type: " + str(df[trend_type][i + scale]))

from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import math

fig = go.Figure()

if y_axis == 'ATT_TP_UL_MV':
  fig.add_trace(go.Scatter(x=df.TEST_PERIOD_FORMATTED, y=df.ATT_TP_UL_MV,
                          mode='lines', name=y_axis))
  fig.add_trace(go.Scatter(x=anomalies.TEST_PERIOD_FORMATTED, y=anomalies.ATT_TP_UL_MV,
                          mode='markers', name='High Anomaly', marker={'color' : 'red'}))
  fig.add_trace(go.Scatter(x=anomalies_low.TEST_PERIOD_FORMATTED, y=anomalies_low.ATT_TP_UL_MV,
                          mode='markers', name='Low Anomaly', marker={'color' : 'orange'}))
  fig.add_trace(go.Scatter(x=shifts.TEST_PERIOD_FORMATTED, y=shifts.ATT_TP_UL_MV,
                          mode='markers', name='Shift', marker={'color' : 'springgreen'}))
  fig.add_trace(go.Scatter(x=trends.TEST_PERIOD_FORMATTED, y=trends.ATT_TP_UL_MV,
                          mode='markers', name='Trend', marker={'color' : 'mediumpurple'}))
else:
  fig.add_trace(go.Scatter(x=df.TEST_PERIOD_FORMATTED, y=df.ATT_TP_DL_MV,
                          mode='lines', name=y_axis))
  fig.add_trace(go.Scatter(x=anomalies.TEST_PERIOD_FORMATTED, y=anomalies.ATT_TP_DL_MV,
                          mode='markers', name='High Anomaly', marker={'color' : 'red'}))
  fig.add_trace(go.Scatter(x=anomalies_low.TEST_PERIOD_FORMATTED, y=anomalies_low.ATT_TP_DL_MV,
                          mode='markers', name='Low Anomaly', marker={'color' : 'orange'}))
  fig.add_trace(go.Scatter(x=shifts.TEST_PERIOD_FORMATTED, y=shifts.ATT_TP_DL_MV,
                          mode='markers', name='Shift', marker={'color' : 'springgreen'}))
  fig.add_trace(go.Scatter(x=trends.TEST_PERIOD_FORMATTED, y=trends.ATT_TP_DL_MV,
                          mode='markers', name='Trend', marker={'color' : 'mediumpurple'}))

# adds vertical bars and arrows on the graph for shifts
for i in range(len(shifts[x_axis])):
  if not math.isnan(shifts[y_axis][i]):
      if df[trend_type][i] == "shift up":
        fig.add_annotation(x=shifts[x_axis][i], y=df[y_axis][i], ax = shifts[x_axis][i], ay=50,
                          xref='x', yref='y', axref='x', ayref='pixel', text='', showarrow = True,
                          arrowhead=4, arrowsize=2, arrowwidth=2, arrowcolor='dark gray')
      else:
        fig.add_annotation(x=shifts[x_axis][i], y=df[y_axis][i], ax=shifts[x_axis][i], ay=-50,
                          xref='x', yref='y', axref='x', ayref='pixel', text='', showarrow = True,
                          arrowhead=4, arrowsize=2, arrowwidth=2, arrowcolor='dark gray')

place = file_name[9:len(file_name) - 4]
fig.update_layout(showlegend=True, title={
        'text': place,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})

fig.show()
fig.write_image(file=str(place + " " + y_axis + ".png"), width=1280, height=720)