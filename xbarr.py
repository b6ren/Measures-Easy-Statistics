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
            Column = Column.split(",")
            incidents = pd.DataFrame(pd.read_excel(
                '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx',
                sheet_name='Sheet3',
                header=0,
                )) 
            numberOfColumns = len(Column)
            dic = {i:i for i in range(numberOfColumns)}
            i = 0
            for singleColumn in Column:
                number = ord(singleColumn)-65
                # 数据的 Pandas Series
                targetDf = incidents.iloc[:,int(number)]
                dic[i] = targetDf.values
                i = i+1

            #建表
            constants={"d":[1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.97, 3.078], "A":[1.88, 1.023, 0.729, 0.577, 0.483, 0.719, 0.373, 0.337, 0.308]}
            constants=pd.DataFrame(constants, index=[2,3,4,5,6,7,8,9,10])

            #画图部分
            df = pd.DataFrame(dic)

            #range相关计算
            ranges = df.max(axis=1) - df.min(axis=1)
            R1 = (df.max(axis=1) - df.min(axis=1)).sum()/df.iloc[:,0].size
            range_bar=ranges.mean()
            xUCL_r = range_bar + constants["A"][numberOfColumns] * R1
            xLCL_r = range_bar - constants["A"][numberOfColumns] * R1

            x1_bar = df.mean().mean()
            results = df[dic.keys()].mean(axis=1)
            results.index = range(0,df.iloc[:,0].size)
            # 插入一个空值 mR 比 x 少1个自由度
            MR = [np.nan]
            # 计算移动极差
            i = 1
            for n in range(len(results) - 1):
                MR.append(abs(results[n + 1] - results[n]))
            i += 1
            # 移动极差的 Pandas Series
            MR = pd.Series(MR)
            # 数据的 Pandas Dataframe
            data = pd.concat([results, MR], axis=1)
            data.columns = ['results', 'MR']

            # 计算 MR-bar
            R1 = (df.max(axis=1) - df.min(axis=1)).sum()/df.iloc[:,0].size
            #df['sample_no'] = ['#'+str(i) for i in range(1,6)]
            
            #sd = results.values.std()
            #print(type(results))
            #print(results)
            #results.plot(xticks=results.index)
            #plt.show()

            xUCL = x1_bar + constants["A"][numberOfColumns] * R1
            xLCL = x1_bar - constants["A"][numberOfColumns] * R1
            #xUCL = x1_bar + 3 * mr_s
            #xLCL = x1_bar - 3 * mr_s
            print(constants["A"][numberOfColumns])
            print(x1_bar)
            print(R1)
            print(xUCL)
            print(xLCL)
            fig = make_subplots(rows=2, cols=1, subplot_titles=("xbar-r控制图"), specs=[[{'secondary_y': True}],[{'secondary_y': True}]])
            # 带条件的颜色列表
            #colors_1 = ['RoyalBlue' if x == False else 'crimson' for x in mask_cl]
            # 折线图主体
            fig.add_trace(go.Scatter(x=np.arange(0, df.iloc[:,0].size), y=results,
                        mode='lines+markers',
                        line_color='RoyalBlue',
                        #marker_color=colors_1,
                        line=dict(width=1),
                        marker=dict(size=5),
                        name='x'),
                        row=1, 
                        col=1,
                        #unique y false
                        secondary_y=False)
            # 设置布局
            lowPoint = np.minimum(min(results),xLCL)
            highPoint = np.maximum(max(results),xUCL)
            lowPoint_r = np.minimum(min(results),xLCL_r)
            highPoint_r = np.maximum(max(results),xUCL_r)
            fig.update_layout(hovermode='x',
                            title='xbar控制图',
                            showlegend=False,
                            width=1800, height=1200 * (highPoint+highPoint_r-lowPoint-lowPoint_r) / df.iloc[:,0].size)
            # 设置 x 轴
            fig.update_xaxes(title='样本',
                            tick0=0, dtick=10,
                            ticks='outside', tickwidth=1, tickcolor='black',
                            range=[0, df.iloc[:,0].size],
                            zeroline=False,
                            showgrid=False,
                            row=1, 
                            col=1,)
            # 设置主 y 轴
            fig.update_yaxes(title='xbar',
                            ticks='outside', tickwidth=1, tickcolor='black',
                            range=[lowPoint_r, highPoint_r],
                            nticks=5,
                            showgrid=False,
                            #unique y false
                            secondary_y=False,
                            row=1, 
                            col=1,)
            # UCL 辅助线
            fig.add_shape(type='line',
                        line_color='crimson',
                        line_width=1,
                        x0=0, x1=df.iloc[:,0].size, xref='x1', y0=xUCL, y1=xUCL, yref='y2',
                        secondary_y=True,
                        row=1, 
                        col=1,)
            # 均值辅助线
            fig.add_shape(type='line',
                        line_color='LightSeaGreen',
                        line_width=1,
                        x0=0, x1=df.iloc[:,0].size, xref='x1', y0=x1_bar, y1=x1_bar, yref='y2',
                        secondary_y=True,
                        row=1, 
                        col=1,)
            # LCL 辅助线
            fig.add_shape(type='line',
                        line_color='crimson',
                        line_width=1,
                        x0=0, x1=df.iloc[:,0].size, xref='x1', y0=xLCL, y1=xLCL, yref='y2',
                        secondary_y=True,
                        row=1, 
                        col=1,)
            # 设置副 y 轴 为了方便标记界限值
            fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                            range=[lowPoint_r, highPoint_r],
                            ticktext=['LCL=' + str(np.round(xLCL, 3)),
                                    'x-bar=' + str(np.round(x1_bar, 3)),
                                    'UCL=' + str(np.round(xUCL, 3))],
                            tickvals=[xLCL, x1_bar, xUCL],
                            showgrid=False,
                            secondary_y=True,
                            row=1, 
                            col=1,)

            fig.add_trace(go.Scatter(x=np.arange(1,df.iloc[:,0].size + 2), y=ranges, 
                mode='lines+markers', 
                line_color='RoyalBlue',
                #marker_color=colors_2, 
                line=dict(width=1), 
                marker=dict(size=5), 
                name='x'), 
                row=2, 
                col=1,
                secondary_y=False)
            fig.update_xaxes(title='样本', 
                tick0=0, dtick=10, 
                ticks='outside', tickwidth=1, tickcolor='black', 
                range=[0, df.iloc[:,0].size], 
                zeroline=False, 
                showgrid=False,
                row=2, 
                col=1,)
            fig.update_yaxes(title='R', 
                ticks='outside', tickwidth=1, tickcolor='black',
                range=[xLCL_r, xUCL_r * 2], 
                nticks=5, 
                showgrid=False, 
                secondary_y=False,
                row=2, 
                col=1,)
            fig.add_shape(type='line', 
                line_color='crimson', 
                line_width=1, 
                x0=0, x1=df.iloc[:,0].size, xref='x1', y0=xUCL_r, y1=xUCL_r, yref='y2', 
                secondary_y=True,
                row=2, 
                col=1,)
            fig.add_shape(type='line', 
                line_color='LightSeaGreen', 
                line_width=1, 
                x0=0, x1=df.iloc[:,0].size, xref='x1', y0=range_bar, y1=range_bar, yref='y2', 
                secondary_y=True,
                row=2, 
                col=1,)
            fig.add_shape(type='line', 
                line_color='crimson', 
                line_width=1, 
                x0=0, x1=df.iloc[:,0].size, xref='x1', y0=xLCL_r, y1=xLCL_r, yref='y2', 
                secondary_y=True,
                row=2, 
                col=1,)
            fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                range=[xLCL_r, xUCL_r * 2], 
                ticktext=['LCL=' + str(np.round(xLCL_r, 3)), 
                    'MR-bar=' + str(np.round(range_bar, 3)), 
                    'UCL=' + str(np.round(xUCL_r, 3))], 
                tickvals=[xLCL_r, range_bar, xUCL_r], 
                showgrid=False, 
                secondary_y=True,
                row=2, 
                col=1,)

            app=xw.App(visible=False, add_book=True)
            bk=app.books.open('/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx')
            sht=bk.sheets.add()
            sht.name='sheet2'
            sht.pictures.add(fig, name='xbar plot', update=True, left=sht.range('A1').left, top=sht.range('A1').top, width=1000, height=800 * (highPoint+highPoint_r-lowPoint-lowPoint_r) / df.iloc[:,0].size)