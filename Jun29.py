import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xlwings as xw
import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as sc
import statistics
from tkinter import *

def main():
    window = Tk()
    window.title("I-MR Control Chart")
    window.geometry("350x200")

    def clicked():
        window_sign_up = Toplevel(window)
        window_sign_up.geometry('300x200')
        window_sign_up.title('Properties Window')
        incidents = pd.DataFrame(pd.read_excel(
            '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Jun29/每日新增BUG数的控制图.xlsx',
            sheet_name='其他监控PPB',
            header=5,
        ))
        # 数据的 Pandas Series
        y = incidents[~incidents['每日发现缺陷数'].isin([0])].reset_index()
        x = y['每日发现缺陷数']

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

        # 计算 X-bar
        x_bar = statistics.mean(data['x'])

        # 计算 MR-bar
        mr_bar = statistics.mean(data['MR'][1:])

        # 计算 MR-s
        # d2(2) = 1.128
        mr_s = mr_bar / 1.128

        # 计算 xUCL & xLCL
        # 为了实现异常点判定 此处额外计算 B 区和 C 区的界限
        xUCL = x_bar + 3 * mr_s
        xUCL_b = x_bar + 3 * mr_s * (2 / 3)
        xUCL_c = x_bar + 3 * mr_s * (1 / 3)
        xLCL_c = x_bar - 3 * mr_s * (1 / 3)
        xLCL_b = x_bar - 3 * mr_s * (2 / 3)
        xLCL = x_bar - 3 * mr_s

        # 计算 mrUCL & mrLCL
        # 同理 计算 B 区和 C 区的界限
        # D4(2) = 3.267
        # mrUCL = 3.267 * mr_bar
        # d3(2) = 0.852
        mrUCL = mr_bar + 3 * 0.852 * mr_s
        mrUCL_b = mr_bar + 3 * 0.852 * mr_s * (2 / 3)
        mrUCL_c = mr_bar + 3 * 0.852 * mr_s * (1 / 3)
        mrLCL_c = mr_bar - 3 * 0.852 * mr_s * (1 / 3)
        mrLCL_b = mr_bar - 3 * 0.852 * mr_s * (2 / 3)  # mrLCL_b < 0
        mrLCL = 0

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
        # x 控制图绘制：

        # 导入 Plotly 包

        # x chart

        # 新建带有主副 y 轴的画布
        fig = make_subplots(specs=[[{'secondary_y': True}]])
        # 带条件的颜色列表
        colors_1 = ['RoyalBlue' if x == False else 'crimson' for x in mask_cl]
        # 折线图主体
        fig.add_trace(go.Scatter(x=np.arange(1, len(data['x']) + 1), y=data['x'],
                                mode='lines+markers',
                                line_color='RoyalBlue',
                                marker_color=colors_1,
                                line=dict(width=1),
                                marker=dict(size=5),
                                name='x'),
                    secondary_y=False)
        # 设置布局
        fig.update_layout(hovermode='x',
                        title='x chart',
                        showlegend=False,
                        width=1000, height=500)
        # 设置 x 轴
        fig.update_xaxes(title='Sample',
                        tick0=0, dtick=10,
                        ticks='outside', tickwidth=1, tickcolor='black',
                        range=[0, len(data['x'])],
                        zeroline=False,
                        showgrid=False)
        # 设置主 y 轴
        fig.update_yaxes(title='x',
                        ticks='outside', tickwidth=1, tickcolor='black',
                        range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
                        nticks=5,
                        showgrid=False,
                        secondary_y=False)
        # UCL 辅助线
        fig.add_shape(type='line',
                    line_color='crimson',
                    line_width=1,
                    x0=0, x1=len(data['x']), xref='x1', y0=xUCL, y1=xUCL, yref='y2',
                    secondary_y=True)
        # UCL_b 辅助线
        fig.add_shape(type='line',
                    line_color='LightSeaGreen',
                    line_dash='dot',
                    line_width=1,
                    x0=0, x1=len(data['x']), xref='x1', y0=xUCL_b, y1=xUCL_b, yref='y2',
                    secondary_y=True)
        # UCL_c 辅助线
        fig.add_shape(type='line',
                    line_color='LightSeaGreen',
                    line_dash='dot',
                    line_width=1,
                    x0=0, x1=len(data['x']), xref='x1', y0=xUCL_c, y1=xUCL_c, yref='y2',
                    secondary_y=True)
        # 均值辅助线
        fig.add_shape(type='line',
                    line_color='LightSeaGreen',
                    line_width=1,
                    x0=0, x1=len(data['x']), xref='x1', y0=x_bar, y1=x_bar, yref='y2',
                    secondary_y=True)
        # LCL_c 辅助线
        fig.add_shape(type='line',
                    line_color='LightSeaGreen',
                    line_dash='dot',
                    line_width=1,
                    x0=0, x1=len(data['x']), xref='x1', y0=xLCL_c, y1=xLCL_c, yref='y2',
                    secondary_y=True)
        # LCL_b 辅助线
        fig.add_shape(type='line',
                    line_color='LightSeaGreen',
                    line_dash='dot',
                    line_width=1,
                    x0=0, x1=len(data['x']), xref='x1', y0=xLCL_b, y1=xLCL_b, yref='y2',
                    secondary_y=True)
        # LCL 辅助线
        fig.add_shape(type='line',
                    line_color='crimson',
                    line_width=1,
                    x0=0, x1=len(data['x']), xref='x1', y0=xLCL, y1=xLCL, yref='y2',
                    secondary_y=True)
        # 设置副 y 轴 为了方便标记界限值
        fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                        range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
                        ticktext=['LCL=' + str(np.round(xLCL, 3)),
                                'x-bar=' + str(np.round(x_bar, 3)),
                                'UCL=' + str(np.round(xUCL, 3))],
                        tickvals=[xLCL, x_bar, xUCL],
                        showgrid=False,
                        secondary_y=True)

        fig.show()
        
    btn = Button(window, text="I-MR Control Chart", command=clicked)
    btn.grid(column=1, row=0)

    window.mainloop()   

if __name__ == "__main__":
    xw.Book("Jun29.xlsm").set_mock_caller()
    main()
