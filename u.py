from tkinter import messagebox
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xlwings as xw
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import statistics
from tkinter import *
import math

class U:
    def submit(self, Column, Subgroup, r_value):
            if not Column.isalpha() or not Column.isupper() or len(Column)!=1:
                messagebox.showwarning("提示", "请输入大写字母")
            else:
                Subgroup = int(Subgroup)
                number = ord(Column)-65
                incidents = pd.DataFrame(pd.read_excel(
                '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx',
                sheet_name='Sheet3',
                header=0,
                )) 
                # 数据的 Pandas Series
                x = incidents.iloc[:,int(number)]

                # 创建移动极差的 list
                # 插入一个空值 mR 比 x 少1个自由度
                MR = [np.nan]

                # 计算移动极差
                i = 1
                for n in range(len(x) - 1):
                    MR.append(abs(x[n + 1] - x[n]))
                i += 1
                # 移动极差的 Pandas Series
                MR = pd.Series(MR)

                # 数据的 Pandas Dataframe
                data = pd.concat([x, MR], axis=1)
                data.columns = ['x', 'MR']

                # 计算 MR-bar
                mr_bar = statistics.mean(data['MR'][1:])

                # 计算 X-bar 及 MR-s
                #x_bar = statistics.mean(data['x'])
                if r_value.get() == "Not Estimate":
                    x_bar = e_mean.get()
                    ubar = e_mean.get()
                    mr_s = e_sd.get()
                else:
                    x_bar = statistics.mean(data['x'])
                    ubar = statistics.mean(data['x'])
                    mr_s = mr_bar / Subgroup

                # 8 Rules
                def rules(data, cl, ucl, ucl_b, ucl_c, lcl, lcl_b, lcl_c):
                    n = len(data)
                    ind = np.array(range(n))
                    obs = np.arange(1, n + 1)

                    # rule 1
                    ofc1 = data[(data > ucl) | (data < lcl)]
                    ofc1_obs = obs[(data > ucl) | (data < lcl)]

                    # rule 2
                    ofc2_ind = []
                    for i in range(n - 2):
                        d = data[i:i + 3]
                        index = ind[i:i + 3]
                        if ((d > ucl_b).sum() == 2) | ((d < lcl_b).sum() == 2):
                            ofc2_ind.extend(index[(d > ucl_b) | (d < lcl_b)])
                    ofc2_ind = list(sorted(set(ofc2_ind)))
                    ofc2 = data[ofc2_ind]
                    ofc2_obs = obs[ofc2_ind]

                    # rule 3
                    ofc3_ind = []
                    for i in range(n - 4):
                        d = data[i:i + 5]
                        index = ind[i:i + 5]
                        if ((d > ucl_c).sum() == 4) | ((d < lcl_c).sum() == 4):
                            ofc3_ind.extend(index[(d > ucl_c) | (d < lcl_c)])
                    ofc3_ind = list(sorted(set(ofc3_ind)))
                    ofc3 = data[ofc3_ind]
                    ofc3_obs = obs[ofc3_ind]

                    # rule 4
                    ofc4_ind = []
                    for i in range(n - 8):
                        d = data[i:i + 9]
                        index = ind[i:i + 9]
                        if ((d > cl).sum() == 9) | ((d < cl).sum() == 9):
                            ofc4_ind.extend(index)
                    ofc4_ind = list(sorted(set(ofc4_ind)))
                    ofc4 = data[ofc4_ind]
                    ofc4_obs = obs[ofc4_ind]

                    # rule 5
                    ofc5_ind = []
                    for i in range(n - 6):
                        d = data[i:i + 7]
                        index = ind[i:i + 7]
                        if all(u <= v for u, v in zip(d, d[1:])) | all(u >= v for u, v in zip(d, d[1:])):
                            ofc5_ind.extend(index)
                    ofc5_ind = list(sorted(set(ofc5_ind)))
                    ofc5 = data[ofc5_ind]
                    ofc5_obs = obs[ofc5_ind]

                    # rule 6
                    ofc6_ind = []
                    for i in range(n - 7):
                        d = data[i:i + 8]
                        index = ind[i:i + 8]
                        if (all(d > ucl_c) | all(d < lcl_c)):
                            ofc6_ind.extend(index)
                    ofc6_ind = list(sorted(set(ofc6_ind)))
                    ofc6 = data[ofc6_ind]
                    ofc6_obs = obs[ofc6_ind]

                    # rule 7
                    ofc7_ind = []
                    for i in range(n - 14):
                        d = data[i:i + 15]
                        index = ind[i:i + 15]
                        if all(lcl_c < d) and all(d < ucl_c):
                            ofc7_ind.extend(index)
                    ofc7_ind = list(sorted(set(ofc7_ind)))
                    ofc7 = data[ofc7_ind]
                    ofc7_obs = obs[ofc7_ind]

                    # rule 8
                    ofc8_ind = []
                    for i in range(n - 13):
                        d = data[i:i + 14]
                        index = ind[i:i + 14]
                        diff = list(v - u for u, v in zip(d, d[1:]))
                        if all(u * v < 0 for u, v in zip(diff, diff[1:])):
                            ofc8_ind.extend(index)
                    ofc8_ind = list(sorted(set(ofc8_ind)))
                    ofc8 = data[ofc8_ind]
                    ofc8_obs = obs[ofc8_ind]

                    return ofc1, ofc1_obs, ofc2, ofc2_obs, ofc3, ofc3_obs, ofc4, ofc4_obs, ofc5, ofc5_obs, ofc6, ofc6_obs, ofc7, ofc7_obs, ofc8, ofc8_obs

                
                # 控制图绘制：
                print(data['x'])
                print(data['x']/Subgroup)
                results = data['x']/Subgroup
                ubar = results.sum()/len(data['x'])
                xUCL = ubar + 3*math.sqrt(ubar/Subgroup)
                if (ubar - 3*math.sqrt(ubar/Subgroup)) < 0:
                    xLCL = float(0)
                else:
                    xLCL = ubar - 3*math.sqrt(ubar/Subgroup)
                print(xUCL)
                print(xLCL)
                xUCL_b = x_bar + 3 * mr_s * (2 / 3)
                xUCL_c = x_bar + 3 * mr_s * (1 / 3)
                xLCL_c = x_bar - 3 * mr_s * (1 / 3)
                xLCL_b = x_bar - 3 * mr_s * (2 / 3)
                # CL Mask
                x_arr = np.array(data['x'])
                _, ind1, _, ind2, _, ind3, _, ind4, _, ind5, _, ind6, _, ind7, _, ind8 \
                    = rules(x_arr, x_bar, xUCL, xUCL_b, xUCL_c, xLCL, xLCL_b, xLCL_c)
                ind_x = list(set(ind1).union(set(ind2)).union(set(ind3)).union(set(ind4)).union(
                    set(ind5)).union(set(ind6)).union(set(ind7)).union(set(ind8)))
                mask_cl = []
                for i in range(len(x_arr)):
                    if i + 1 in ind_x:
                        mask_cl.append(True)
                    else:
                        mask_cl.append(False)
                # 新建带有主副 y 轴的画布
                fig = make_subplots(rows=1, cols=1, subplot_titles=("U图"), specs=[[{'secondary_y': True}]])
                # 带条件的颜色列表
                colors_1 = ['RoyalBlue' if x == False else 'crimson' for x in mask_cl]
                # 折线图主体
                fig.add_trace(go.Scatter(x=np.arange(1, len(data['x']) + 1), y=results,
                                        mode='lines+markers',
                                        line_color='RoyalBlue',
                                        marker_color=colors_1,
                                        line=dict(width=1),
                                        marker=dict(size=5),
                                        name='x'),
                            row=1, 
                            col=1,
                            secondary_y=False)
                # 设置布局
                lowPoint = np.minimum(min(results),xLCL)
                highPoint = np.maximum(max(results),xUCL)
                print(highPoint)
                print(lowPoint)
                fig.update_layout(hovermode='x',
                                title='U控制图',
                                showlegend=False,
                                width=1200, height=1200)
                # 设置 x 轴
                fig.update_xaxes(title='样本',
                                tick0=0, dtick=10,
                                ticks='outside', tickwidth=1, tickcolor='black',
                                range=[0, len(data['x'])],
                                zeroline=False,
                                showgrid=False,
                                row=1, 
                                col=1,)
                # 设置主 y 轴
                fig.update_yaxes(title='U',
                                ticks='outside', tickwidth=1, tickcolor='black',
                                range=[lowPoint, highPoint],
                                nticks=5,
                                showgrid=False,
                                secondary_y=False,
                                row=1, 
                                col=1,)
                # UCL 辅助线
                fig.add_shape(type='line',
                            line_color='crimson',
                            line_width=1,
                            x0=0, x1=len(data['x']), xref='x1', y0=xUCL, y1=xUCL, yref='y2',
                            secondary_y=True,
                            row=1, 
                            col=1,)
                # 均值辅助线
                fig.add_shape(type='line',
                            line_color='LightSeaGreen',
                            line_width=1,
                            x0=0, x1=len(data['x']), xref='x1', y0=ubar, y1=ubar, yref='y2',
                            secondary_y=True,
                            row=1, 
                            col=1,)
                # LCL 辅助线
                fig.add_shape(type='line',
                            line_color='crimson',
                            line_width=1,
                            x0=0, x1=len(data['x']), xref='x1', y0=xLCL, y1=xLCL, yref='y2',
                            secondary_y=True,
                            row=1, 
                            col=1,)
                # 设置副 y 轴 为了方便标记界限值
                fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                                range=[lowPoint, highPoint],
                                ticktext=['LCL=' + str(np.round(xLCL, 3)),
                                        'u-bar=' + str(np.round(ubar, 3)),
                                        'UCL=' + str(np.round(xUCL, 3))],
                                tickvals=[xLCL, ubar, xUCL],
                                showgrid=False,
                                secondary_y=True,
                                row=1, 
                                col=1,)

                app=xw.App(visible=False, add_book=True)
                bk=app.books.open('/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx')
                sht=bk.sheets.add()
                sht.name='sheet2'
                sht.pictures.add(fig, name='U plot', update=True, left=sht.range('A1').left, top=sht.range('A1').top, width=600, height=600)