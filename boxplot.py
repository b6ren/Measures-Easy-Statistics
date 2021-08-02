import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xlwings as xw
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as sc
import statistics
from tkinter import messagebox
import statistics
from tkinter import *
import math

class Boxplot:
    def submit_singleY(self, Column):
        if not Column.isalpha() or not Column.isupper() or len(Column)!=1:
            messagebox.showwarning("提示", "请输入大写字母")
        else:
            number = ord(Column)-65
            incidents = pd.DataFrame(pd.read_excel(
            '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx',
            sheet_name='Sheet3',
            header=0,
            )) 
            # 数据的 Pandas Series
            x = incidents.iloc[:,int(number)]
            column_name = incidents.columns.tolist()
            # 新建带有主副 y 轴的画布
            df = pd.DataFrame(x)
            fig = go.Figure()
            # 折线图主体
            fig.add_trace(go.Box(y=x,
                                name=column_name[int(number)],
                                line_color='RoyalBlue',
                                marker_color='RoyalBlue',
                                line=dict(width=1),
                                marker=dict(size=5)))
            # 设置布局
            fig.update_layout(hovermode='x',
                            title='箱线图',
                            showlegend=False,
                            width=1000, height=1000)
            fig.show()
            app=xw.App(visible=False, add_book=True)
            bk=app.books.open('/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx')
            sht=bk.sheets.add()
            sht.name='sheet2'
            sht.pictures.add(fig, name='箱线图', update=True, left=sht.range('A1').left, top=sht.range('A1').top, width=800, height=600)
    def submit_singleY_withGroups(self, Column, x):
        x = x.split(",")
        incidents = pd.DataFrame(pd.read_excel(
            '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx',
            sheet_name='Sheet3',
            header=0,
            )) 
        numberOfColumns = len(x)
        dic = {i:i for i in range(numberOfColumns)}
        i = 0
        for singleColumn in x:
            number = ord(singleColumn)-65
            # 数据的 Pandas Series
            targetDf = incidents.iloc[:,int(number)]
            dic[i] = targetDf.values
            i = i+1
        column_name = incidents.columns.tolist()
        df = pd.DataFrame(dic)
        number = ord(Column)-65
        xColumn = incidents.iloc[:,int(number)]
        # 画图
        fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
        # 折线图主体
        fig.add_trace(go.Box(y=df[0],
                            x=xColumn,
                            name=column_name[ord(x[0])-65],
                            line_color='RoyalBlue',
                            marker_color='RoyalBlue',
                            line=dict(width=1),
                            marker=dict(size=5)),
                    secondary_y=False)
        if numberOfColumns>1:
            fig.add_trace(go.Box(y=df[1],
                            x=xColumn,
                            name=column_name[ord(x[0])-65],
                            line_color='RoyalBlue',
                            marker_color='RoyalBlue',
                            line=dict(width=1),
                            marker=dict(size=5)),
                    secondary_y=False)
        # 设置布局
        fig.update_layout(hovermode='x',
                        title='箱线图',
                        showlegend=False,
                        width=1000, height=1000)

        fig.show()
        app=xw.App(visible=False, add_book=True)
        bk=app.books.open('/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx')
        sht=bk.sheets.add()
        sht.name='sheet2'
        sht.pictures.add(fig, name='箱线图', update=True, left=sht.range('A1').left, top=sht.range('A1').top, width=800, height=600)
    def submit_multipleY(self, Column):
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
        column_name = incidents.columns.tolist()
        df = pd.DataFrame(dic)
        # 画图
        fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])
        # 折线图主体
        columns = df.columns.tolist()
        for c in columns:
            fig.add_trace(go.Box(y=df[c],
                            name=column_name[c+1],
                            line_color='RoyalBlue',
                            marker_color='RoyalBlue',
                            line=dict(width=1),
                            marker=dict(size=5)),
                    secondary_y=False)
        # 设置布局
        fig.update_layout(hovermode='x',
                        title='箱线图',
                        showlegend=False,
                        width=1000, height=1000)
        fig.show()
        app=xw.App(visible=False, add_book=True)
        bk=app.books.open('/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx')
        sht=bk.sheets.add()
        sht.name='sheet2'
        sht.pictures.add(fig, name='箱线图', update=True, left=sht.range('A1').left, top=sht.range('A1').top, width=800, height=600)