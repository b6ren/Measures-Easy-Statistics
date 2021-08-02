from tkinter import messagebox
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xlwings as xw
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import os
import scipy.stats as sc
import statistics
from tkinter import *

class XBARR:
    def submit(self, Column, r_value):
            incidents = pd.DataFrame(pd.read_excel(
                '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx',
                sheet_name='Sheet3',
                header=0,
            )) 
            Column = Column.split(",")
            numberOfColumns = len(Column)
            dic = {i:i for i in range(numberOfColumns)}
            i = 0
            for singleColumn in Column:
                number = ord(singleColumn)-65
                # 数据的 Pandas Series
                targetDf = incidents.iloc[:,int(number)]
                dic[i] = targetDf.values
                i = i+1
            x = pd.DataFrame(dic)
            # MR chart
            #colors_2 = ['RoyalBlue' if x == False else 'crimson' for x in mask_mr]
            sds = x.iloc[:,0].std(ddof=0)
            R1 = (x.max(axis=1) - x.min(axis=1)).sum()/x.iloc[:,0].size
            x1_bar=sds.mean()
            xUCL = x1_bar + 1.023 * R1
            xLCL = x1_bar - 1.023 * R1
            fig = make_subplots(rows=1, cols=1, subplot_titles=("xbar-r控制图"), specs=[[{'secondary_y': True}]])
            fig.add_trace(go.Scatter(x=np.arange(1,x.iloc[:,0].size + 2), y=sds, 
                mode='lines+markers', 
                line_color='RoyalBlue',
                #marker_color=colors_2, 
                line=dict(width=1), 
                marker=dict(size=5), 
                name='x'), 
                row=1, 
                col=1,
                secondary_y=False)
            fig.update_layout(hovermode='x', 
                title='xbar-r控制图', 
                showlegend=False, 
                width=1500, height=1000)
            fig.update_xaxes(title='样本', 
                tick0=0, dtick=10, 
                ticks='outside', tickwidth=1, tickcolor='black', 
                range=[0, x.iloc[:,0].size], 
                zeroline=False, 
                showgrid=False,
                row=1, 
                col=1,)
            fig.update_yaxes(title='MR', 
                ticks='outside', tickwidth=1, tickcolor='black',
                range=[xLCL, xUCL * 2], 
                nticks=5, 
                showgrid=False, 
                secondary_y=False,
                row=1, 
                col=1,)
            fig.add_shape(type='line', 
                line_color='crimson', 
                line_width=1, 
                x0=0, x1=x.iloc[:,0].size, xref='x1', y0=xUCL, y1=xUCL, yref='y2', 
                secondary_y=True,
                row=1, 
                col=1,)
            fig.add_shape(type='line', 
                line_color='LightSeaGreen', 
                line_width=1, 
                x0=0, x1=x.iloc[:,0].size, xref='x1', y0=x1_bar, y1=x1_bar, yref='y2', 
                secondary_y=True,
                row=1, 
                col=1,)
            fig.add_shape(type='line', 
                line_color='crimson', 
                line_width=1, 
                x0=0, x1=x.iloc[:,0].size, xref='x1', y0=xLCL, y1=xLCL, yref='y2', 
                secondary_y=True,
                row=1, 
                col=1,)
            fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                range=[xLCL, xUCL * 2], 
                ticktext=['LCL=' + str(np.round(xLCL, 3)), 
                    'MR-bar=' + str(np.round(x1_bar, 3)), 
                    'UCL=' + str(np.round(xUCL, 3))], 
                tickvals=[xLCL, x1_bar, xUCL], 
                showgrid=False, 
                secondary_y=True,
                row=1, 
                col=1,)

            app=xw.App(visible=False, add_book=True)
            bk=app.books.open('/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx')
            sht=bk.sheets.add()
            sht.name='sheet2'
            sht.pictures.add(fig, name='I-MR plot', update=True, left=sht.range('A1').left, top=sht.range('A1').top, width=750, height=500)
            #bk.save()
            #bk.close()