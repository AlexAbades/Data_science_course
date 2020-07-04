import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import pandas as pd

init_notebook_mode(connected=True)

data = dict(type='choropleth',
            locations=['AZ', 'CA', 'NY'],
            locationmode='USA-states',
            colorscale='Jet',
            text=['text 1', 'text 2', 'text 3'],
            z=[1.0, 2.0, 3.0],
            colorbar={'title': 'Colorbat Title Goes Here'})
print(data)

layout = dict(geo={'scope': 'usa'})
choromap = go.Figure(data=[data], layout=layout)
plot(choromap, filename='USA_map.html')

df = pd.read_csv(
    r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\09-Geographical-Plotting\2011_US_AGRI_Exports")
print(df)

data1 = dict(type='choropleth',
             colorscale='Jet',
             locations=df['code'],
             locationmode='USA-states',
             z=df['total exports'],
             text=df['text'],
             marker=dict(line=dict(color='rgb(255,255,255)', width=2)),  # It's to show the separation between the
             # states itself
             colorbar={'title': 'Millions USD'}
             )
layout1 = dict(title='2011 US Agriculture Exports by State',
               geo=dict(scope='usa', showlakes=True, lakecolor='rgb(85,173,240)'))  # Show lakes (lagos)

choromap2 = go.Figure(data=[data1], layout=layout1)
plot(choromap2)