import pandas as pd
import numpy as np
import chart_studio as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, iplot, plot, init_notebook_mode

init_notebook_mode(connected=True)
df = pd.read_csv(r"D:\Documentos\Trabajo\Udemy\Refactored_Py_DS_ML_Bootcamp-master\09-Geographical-Plotting"
                 r"\2014_World_GDP")
print(df)

data = dict(type='choropleth',
            locations=df['CODE'],
            z=df['GDP (BILLIONS)'],
            text=df['COUNTRY'],
            colorscale='Earth',
            colorbar={'title':'GDP in Billions USD'}
            )
layout=dict(title='2014 Global GDP',
            geo=dict(showframe=False,
                     projection={'type':"winkel tripel"}))

choromap = go.Figure(data=[data], layout=layout)

plot(choromap, filename='Global GDP.html')