import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot, download_plotlyjs
import pandas as pd

init_notebook_mode(connected=True)

df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\09-Geographical-Plotting"
                 r"\2014_World_Power_Consumption")

print(df.info())

# We have to look out with the locations mode, if we use the country names, we have to specify that we are using that!

# Ex 1;

data = dict(type='choropleth',
            locations=df['Country'],
            locationmode='country names',
            colorscale='Viridis',
            reversescale=True,  # It reverse the color scale.
            text=df['Country'],
            z=df['Power Consumption KWH'],
            colorbar={'title': 'KWH'})

layout = dict(title='2014 Global GDP',
              geo=dict(showframe=True,  # Shows the world line around the plot
                       projection={'type': "winkel tripel"}))

choromap = go.Figure(data=[data], layout=layout)
plot(choromap, filename='Exercise_1_KWH.html')

# https://plotly.com/python/reference/#choropleth


# Ex 2;
df1 = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\09-Geographical-Plotting"
                  r"\2012_Election_Data")

print(df1)
print(df1.info())

data1 = dict(type='choropleth',
             locations=df1['State Abv'],
             locationmode='USA-states',  # provide locations as two-letter state abbreviations:
             colorscale='Earth',
             reversescale=True,
             text=df1['State'],
             z=df1['Voting-Age Population (VAP)'],
             colorbar={'title': 'KWH'})

layout1 = dict(title='Elections 2012',
               geo=dict(showframe=True,
                        scope='usa',
                        showlakes=True,
                        lakecolor='rgb(85, 173, 240)'
                        )
               )

choromap = go.Figure(data=[data1], layout=layout1)
plot(choromap, filename='Exercise_2_VPA.html')
