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
from tkinter import *
from tkinter import ttk
from tkinter import messagebox

def main():

    window = Tk()
    window.title("I-MR控制图")
    window.geometry("465x390")

    def clicked():
        window_sign_up = Toplevel(window)
        window_sign_up.geometry("575x510")
        window_sign_up.title('属性')

        def show_options():
            if l_mean['state']==NORMAL:
                l_mean['state']=DISABLED
                l_sd['state']=DISABLED
                e_mean['state']=DISABLED
                e_sd['state']=DISABLED
            else:
                l_mean['state']=NORMAL
                e_mean['state']=NORMAL
                l_sd['state']=NORMAL
                e_sd['state']=NORMAL
        def show_options_e():
            l_mean['state']=DISABLED
            e_mean['state']=DISABLED
            l_sd['state']=DISABLED
            e_sd['state']=DISABLED

        def submit():
            Column = e_column.get()
            print (Column.isalpha())
            if not Column.isalpha() or not Column.isupper() or len(Column)!=1:
                messagebox.showwarning("提示", "请输入大写字母")
            else:
                number = ord(Column)-65
                """wb = xw.Book.caller()
                sheet = wb.sheets[0]
                index1 = sheet.range((1,1),(1,15)).value
                index2 = Series(index1)
                incidents = sheet.range((2,1),(sheet.used_range.last_cell.row,15)).value
                incidents = pd.DataFrame(incidents,columns=index2)
                print(wb)
                print(incidents)
                incidents = pd.DataFrame(pd.read_excel(wb,
                sheet_name='其他监控PPB',
                header=5,
                ))
                """
                incidents = pd.DataFrame(pd.read_excel(
                '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx',
                sheet_name='其他监控PPB',
                header=5,
                )) 
                # 数据的 Pandas Series
                #y = incidents[~incidents.iloc[:,int(Column)].isin([0])].reset_index()
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
                    mr_s = e_sd.get()
                else:
                    x_bar = statistics.mean(data['x'])
                    mr_s = mr_bar / 1.128

                # 计算 MR-s
                # d2(2) = 1.128
                # mr_s = mr_bar / 1.128

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
                fig = make_subplots(rows=2, cols=1, subplot_titles=("I图", "MR图"), specs=[[{'secondary_y': True}],[{'secondary_y': True}]])
                #fig = make_subplots(specs=[[{'secondary_y': True, 'rows': 2, 'cols': 1, 'subplot_titles': ["I图", "MR图"]}]])
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
                            row=1, 
                            col=1,
                            secondary_y=False)
                # 设置布局
                fig.update_layout(hovermode='x',
                                title='I-MR控制图',
                                showlegend=False,
                                width=1000, height=1000)
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
                fig.update_yaxes(title='I',
                                ticks='outside', tickwidth=1, tickcolor='black',
                                range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
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
                            x0=0, x1=len(data['x']), xref='x1', y0=x_bar, y1=x_bar, yref='y2',
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
                                range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
                                ticktext=['LCL=' + str(np.round(xLCL, 3)),
                                        'x-bar=' + str(np.round(x_bar, 3)),
                                        'UCL=' + str(np.round(xUCL, 3))],
                                tickvals=[xLCL, x_bar, xUCL],
                                showgrid=False,
                                secondary_y=True,
                                row=1, 
                                col=1,)

                #fig.show()

                # MR Mask
                mr_arr = np.array(data['MR'][1:])
                _, ind1, _, ind2, _, ind3, _, ind4, _, ind5, _, ind6, _, ind7, _, ind8 \
                    = rules(mr_arr, mr_bar, mrUCL, mrUCL_b, mrUCL_c, mrLCL, mrLCL_b, mrLCL_c)
                ind_mr = list(set(ind1).union(set(ind2)).union(set(ind3)).union(set(ind4)).union(set(ind5)).union(set(ind6)).union(set(ind7)).union(set(ind8)))
                mask_mr = [False]
                for i in range(len(mr_arr)):
                    if i + 1 in ind_mr:
                        mask_mr.append(True)
                    else:
                        mask_mr.append(False)
                # MR chart
                #fig = make_subplots(specs=[[{'secondary_y': True}]])
                colors_2 = ['RoyalBlue' if x == False else 'crimson' for x in mask_mr]
                fig.add_trace(go.Scatter(x=np.arange(1, len(data['x'] + 2)), y=data['MR'], 
                    mode='lines+markers', 
                    line_color='RoyalBlue',
                    marker_color=colors_2, 
                    line=dict(width=1), 
                    marker=dict(size=5), 
                    name='x'), 
                    row=2, 
                    col=1,
                    secondary_y=False)
                """ fig.update_layout(hovermode='x', 
                    title='I-MR控制图', 
                    showlegend=False, 
                    width=1000, height=1000) """
                fig.update_xaxes(title='样本', 
                    tick0=0, dtick=10, 
                    ticks='outside', tickwidth=1, tickcolor='black', 
                    range=[0, len(data['x'])], 
                    zeroline=False, 
                    showgrid=False,
                    row=2, 
                    col=1,)
                fig.update_yaxes(title='MR', 
                    ticks='outside', tickwidth=1, tickcolor='black',
                    range=[mrLCL, mrUCL + mrUCL * 0.1], 
                    nticks=5, 
                    showgrid=False, 
                    secondary_y=False,
                    row=2, 
                    col=1,)
                fig.add_shape(type='line', 
                    line_color='crimson', 
                    line_width=1, 
                    x0=0, x1=len(data['x']), xref='x1', y0=mrUCL, y1=mrUCL, yref='y2', 
                    secondary_y=True,
                    row=2, 
                    col=1,)
                fig.add_shape(type='line', 
                    line_color='LightSeaGreen', 
                    line_width=1, 
                    x0=0, x1=len(data['x']), xref='x1', y0=mr_bar, y1=mr_bar, yref='y2', 
                    secondary_y=True,
                    row=2, 
                    col=1,)
                fig.add_shape(type='line', 
                    line_color='crimson', 
                    line_width=1, 
                    x0=0, x1=len(data['x']), xref='x1', y0=mrLCL, y1=mrLCL, yref='y2', 
                    secondary_y=True,
                    row=2, 
                    col=1,)
                fig.update_yaxes(ticks='outside', tickwidth=1, tickcolor='black',
                    range=[mrLCL, mrUCL + mrUCL * 0.1], 
                    ticktext=['LCL=' + str(np.round(mrLCL, 3)), 
                        'MR-bar=' + str(np.round(mr_bar, 3)), 
                        'UCL=' + str(np.round(mrUCL, 3))], 
                    tickvals=[mrLCL, mr_bar, mrUCL], 
                    showgrid=False, 
                    secondary_y=True,
                    row=2, 
                    col=1,)

                app=xw.App(visible=False, add_book=True)
                bk=app.books.open('/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx')
                sht=bk.sheets.add()
                sht.name='sheet2'
                sht.pictures.add(fig, name='I-MR plot', update=True, left=sht.range('A1').left, top=sht.range('A1').top, width=500, height=500)
                #bk.save()
                #bk.close()
              
        notebook = ttk.Notebook(window_sign_up)

        frameOne = Frame(notebook)
        frameTwo = Frame(notebook)
        frameThree = Frame(notebook)
        
        notebook.add(frameOne, text='数据')
        notebook.add(frameTwo, text='选项')
        notebook.add(frameThree, text='显示')
        notebook.grid(column=2, row=0)

        l_column =Label(frameOne,text='列(A,B,C...): ')
        l_column.grid(row=0,sticky=W, padx=15, pady=20)
        e_column =Entry(frameOne)
        e_column.grid(row=0,column=1,sticky=E)
        btn = Button(frameOne, text="确定", command=submit)
        btn.grid(column=1, row=3, sticky=E)

        l_mean =Label(frameTwo,text='平均值：',state=DISABLED)
        l_mean.grid(row=3,sticky=W, padx=15)
        l_sd =Label(frameTwo,text='标准差：',state=DISABLED)
        l_sd.grid(row=4,sticky=W, padx=15)
        #Section One
        
        
        r_value = StringVar()
        r_value.set("Estimate")
        #r_params = Radiobutton(frameTwo, text="Estimate the mean and the standard deviation from the data", variable=r_value, value="Estimate")
        r_params = Radiobutton(frameTwo, text="根据数据估算平均值和标准差", variable=r_value, value="Estimate", command=show_options_e)
        r_params.grid(row=1,sticky=W, padx=15)
        #r_params_second = Radiobutton(frameTwo, text="Specify the value for one or both parameters", variable=r_value, value="Not Estimate")
        r_params_second = Radiobutton(frameTwo, text="指定一个或两个参数的值", variable=r_value, value="Not Estimate", command=show_options)
        r_params_second.grid(row=2,sticky=W, padx=15)
        l_params =Label(frameTwo,text='参数：')
        l_params.grid(row=0,sticky=W, padx=15, pady=20)
        
        
        e_mean =Entry(frameTwo,state=DISABLED)
        e_mean.grid(row=3,column=1,sticky=E)
        
        
        e_sd =Entry(frameTwo,state=DISABLED)
        e_sd.grid(row=4,column=1,sticky=E)
        #Section Two
        """ l_omit =Label(frameTwo,text='Omit the following observations when estimating parameters(optional): ')
        l_omit.grid(row=5,sticky=W)
        e_omit =Entry(frameTwo)
        e_omit.grid(row=5,column=1,sticky=E) """
        btn = Button(frameTwo, text="确定", command=submit)
        btn.grid(column=1, row=6, sticky=E)

        #Display
        #Section One
        longtext='''全部检验：
        1. 1个点，距离中心线大于3个标准差
        2. 连续9点在中心线同一侧
        3. 连续6个点，全部递增或全部递减
        4. 连续14个点，上下交错
        5. 3个点中有2个点，距离中心线（同侧）大于2个标准差
        6. 5个点中有4个点，距离中心线（同侧）大于1个标准差
        7. 连续15个点，距离中心线（任一侧）1个标准差以内
        8. 连续8个点，距离中心线（任一侧）大于1个标准差
        '''
        l_tests =Label(frameThree,text = longtext, justify = "left")
        l_tests.grid(row=0,sticky=W, padx=15, pady=20)
        r_value_display = StringVar()
        r_value_display.set("testOne")
        r_tests_second = Radiobutton(frameThree, text="执行全部特殊检验", variable=r_value_display, value="all tests")
        r_tests_second.grid(row=3,sticky=W, padx=15)
        r_tests = Radiobutton(frameThree, text="仅执行检验1", variable=r_value_display, value="testOne")
        r_tests.grid(row=2,sticky=W, padx=15)
        r_tests_third = Radiobutton(frameThree, text="仅执行检验1、5、6、8", variable=r_value_display, value="part of tests")
        r_tests_third.grid(row=4,sticky=W, padx=15)
        r_tests_four = Radiobutton(frameThree, text="不执行检验", variable=r_value_display, value="no tests")
        r_tests_four.grid(row=5,sticky=W, padx=15)
        l_tests =Label(frameThree,text='特殊检验：')
        l_tests.grid(row=1,sticky=W, padx=15)
        #Section Two
        c_value_1 = "1sd"
        c_value_2 = "2sd"
        l_additional =Label(frameThree,text='显示额外的在中心线上下的标准差限制：')
        l_additional.grid(row=6,sticky=W, padx=15)
        c_additional = Checkbutton(frameThree, text="1标准差", variable=c_value_1, command=show_options)
        c_additional.grid(row=7,sticky=W, padx=15)
        c_additional_second = Checkbutton(frameThree, text="2标准差", variable=c_value_2, command=show_options)
        c_additional_second.grid(row=8,sticky=W, padx=15)
        btn = Button(frameThree, text="确定", command=submit)
        btn.grid(column=1, row=9, sticky=E, padx=15, pady=20)

    #xbar控制图
    def clicked_xbar():
        window_sign_up = Toplevel(window)
        window_sign_up.geometry("575x510")
        window_sign_up.title('属性')

        def show_options():
            if l_mean['state']==NORMAL:
                l_mean['state']=DISABLED
                l_sd['state']=DISABLED
                e_mean['state']=DISABLED
                e_sd['state']=DISABLED
            else:
                l_mean['state']=NORMAL
                e_mean['state']=NORMAL
                l_sd['state']=NORMAL
                e_sd['state']=NORMAL
        def show_options_e():
            l_mean['state']=DISABLED
            e_mean['state']=DISABLED
            l_sd['state']=DISABLED
            e_sd['state']=DISABLED

        def submit():
            Column = e_column.get()
            Column = Column.split(",")
            incidents = pd.DataFrame(pd.read_excel(
                '/Users/boyaren/Documents/ExcelStatistics/ControlChart/Measures-Easy-Statistics/每日新增BUG数的控制图.xlsx',
                sheet_name='其他监控PPB',
                header=5,
                )) 
            numberOfColumns = len(Column)
            dic = {i:i for i in range(numberOfColumns)}
            i = 0
            for singleColumn in Column:
                """ print (Column.isalpha())
                if not Column.isalpha() or not Column.isupper() or len(Column)!=1:
                    messagebox.showwarning("提示", "请输入大写字母")
                else: """
                number = ord(singleColumn)-65
                # 数据的 Pandas Series
                #y = incidents[~incidents.iloc[:,int(Column)].isin([0])].reset_index()
                targetDf = incidents.iloc[:,int(number)]
                dic[i] = targetDf.values
                i = i+1

            #画图部分
            df = pd.DataFrame(dic)
            x1_bar = df.mean().mean()
            R1 = (df.max(axis=1) - df.min(axis=1)).sum()/numberOfColumns
            #df['sample_no'] = ['#'+str(i) for i in range(1,6)]
            results = df[dic.keys()].mean(axis=1)
            results.index = range(0,df.iloc[:,0].size)
            results.plot(xticks=results.index)


            """ x = {incidents.iloc[:,int(number)],
                incidents.iloc[:,int(number)]
                }

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
                mr_s = e_sd.get()
            else:
                x_bar = statistics.mean(data['x'])
                mr_s = mr_bar / 1.128

            # 计算 MR-s
            # d2(2) = 1.128
            # mr_s = mr_bar / 1.128

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
            fig = make_subplots(rows=2, cols=1, subplot_titles=("I图", "MR图"), specs=[[{'secondary_y': True}],[{'secondary_y': True}]])
            #fig = make_subplots(specs=[[{'secondary_y': True, 'rows': 2, 'cols': 1, 'subplot_titles': ["I图", "MR图"]}]])
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
                        row=1, 
                        col=1,
                        secondary_y=False)
            # 设置布局
            fig.update_layout(hovermode='x',
                            title='I-MR控制图',
                            showlegend=False,
                            width=1000, height=1000)
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
            fig.update_yaxes(title='I',
                            ticks='outside', tickwidth=1, tickcolor='black',
                            range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
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
                        x0=0, x1=len(data['x']), xref='x1', y0=x_bar, y1=x_bar, yref='y2',
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
                            range=[xLCL - xLCL * 0.02, xUCL + xUCL * 0.02],
                            ticktext=['LCL=' + str(np.round(xLCL, 3)),
                                    'x-bar=' + str(np.round(x_bar, 3)),
                                    'UCL=' + str(np.round(xUCL, 3))],
                            tickvals=[xLCL, x_bar, xUCL],
                            showgrid=False,
                            secondary_y=True,
                            row=1, 
                            col=1,)

            #fig.show() """
        
        
        notebook = ttk.Notebook(window_sign_up)

        frameOne = Frame(notebook)
        frameTwo = Frame(notebook)
        frameThree = Frame(notebook)
        
        notebook.add(frameOne, text='数据')
        notebook.add(frameTwo, text='选项')
        notebook.add(frameThree, text='显示')
        notebook.grid(column=2, row=0)

        l_column =Label(frameOne,text='列(A,B,C...): ')
        l_column.grid(row=0,sticky=W, padx=15, pady=20)
        e_column =Entry(frameOne)
        e_column.grid(row=0,column=1,sticky=E)
        """ value = StringVar()
        cbx = ttk.Combobox(frameOne, width=12, height = 8, textvariable = value)
        cbx.grid(row=0,column=1,sticky=E)
        columnList = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "G", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W","X","Y","Z","AA","AB","AC","AD","AE","AF","AG","AH","AI","AJ","AK","AL","AM","AN","AO","AP","AQ","AR","AS","AT","AU","AV","AW","AX","AY","AZ"]
        cbx["values"] = columnList """
        btn = Button(frameOne, text="确定", command=submit)
        btn.grid(column=1, row=3, sticky=E)

        l_mean =Label(frameTwo,text='平均值：',state=DISABLED)
        l_mean.grid(row=3,sticky=W, padx=15)
        l_sd =Label(frameTwo,text='标准差：',state=DISABLED)
        l_sd.grid(row=4,sticky=W, padx=15)
        #Section One
        
        
        r_value = StringVar()
        r_value.set("Estimate")
        #r_params = Radiobutton(frameTwo, text="Estimate the mean and the standard deviation from the data", variable=r_value, value="Estimate")
        r_params = Radiobutton(frameTwo, text="根据数据估算平均值和标准差", variable=r_value, value="Estimate", command=show_options_e)
        r_params.grid(row=1,sticky=W, padx=15)
        #r_params_second = Radiobutton(frameTwo, text="Specify the value for one or both parameters", variable=r_value, value="Not Estimate")
        r_params_second = Radiobutton(frameTwo, text="指定一个或两个参数的值", variable=r_value, value="Not Estimate", command=show_options)
        r_params_second.grid(row=2,sticky=W, padx=15)
        l_params =Label(frameTwo,text='参数：')
        l_params.grid(row=0,sticky=W, padx=15, pady=20)
        
        
        e_mean =Entry(frameTwo,state=DISABLED)
        e_mean.grid(row=3,column=1,sticky=E)
        
        
        e_sd =Entry(frameTwo,state=DISABLED)
        e_sd.grid(row=4,column=1,sticky=E)
        #Section Two
        """ l_omit =Label(frameTwo,text='Omit the following observations when estimating parameters(optional): ')
        l_omit.grid(row=5,sticky=W)
        e_omit =Entry(frameTwo)
        e_omit.grid(row=5,column=1,sticky=E) """
        btn = Button(frameTwo, text="确定", command=submit)
        btn.grid(column=1, row=6, sticky=E)

        #Display
        #Section One
        longtext='''全部检验：
        1. 1个点，距离中心线大于3个标准差
        2. 连续9点在中心线同一侧
        3. 连续6个点，全部递增或全部递减
        4. 连续14个点，上下交错
        5. 3个点中有2个点，距离中心线（同侧）大于2个标准差
        6. 5个点中有4个点，距离中心线（同侧）大于1个标准差
        7. 连续15个点，距离中心线（任一侧）1个标准差以内
        8. 连续8个点，距离中心线（任一侧）大于1个标准差
        '''
        l_tests =Label(frameThree,text = longtext, justify = "left")
        l_tests.grid(row=0,sticky=W, padx=15, pady=20)
        r_value_display = StringVar()
        r_value_display.set("testOne")
        r_tests_second = Radiobutton(frameThree, text="执行全部特殊检验", variable=r_value_display, value="all tests")
        r_tests_second.grid(row=3,sticky=W, padx=15)
        r_tests = Radiobutton(frameThree, text="仅执行检验1", variable=r_value_display, value="testOne")
        r_tests.grid(row=2,sticky=W, padx=15)
        r_tests_third = Radiobutton(frameThree, text="仅执行检验1、5、6、8", variable=r_value_display, value="part of tests")
        r_tests_third.grid(row=4,sticky=W, padx=15)
        r_tests_four = Radiobutton(frameThree, text="不执行检验", variable=r_value_display, value="no tests")
        r_tests_four.grid(row=5,sticky=W, padx=15)
        l_tests =Label(frameThree,text='特殊检验：')
        l_tests.grid(row=1,sticky=W, padx=15)
        #Section Two
        c_value_1 = "1sd"
        c_value_2 = "2sd"
        l_additional =Label(frameThree,text='显示额外的在中心线上下的标准差限制：')
        l_additional.grid(row=6,sticky=W, padx=15)
        c_additional = Checkbutton(frameThree, text="1标准差", variable=c_value_1, command=show_options)
        c_additional.grid(row=7,sticky=W, padx=15)
        c_additional_second = Checkbutton(frameThree, text="2标准差", variable=c_value_2, command=show_options)
        c_additional_second.grid(row=8,sticky=W, padx=15)
        btn = Button(frameThree, text="确定", command=submit)
        btn.grid(column=1, row=9, sticky=E, padx=15, pady=20)
        
           
    btn = Button(window, text="I-MR控制图", command=clicked)
    btn.grid(column=1, row=0, padx=15, pady=20)
    btn = Button(window, text="Xbar控制图", command=clicked_xbar)
    btn.grid(column=2, row=0, padx=15, pady=20)
    btn = Button(window, text="Xbar-R控制图", command=clicked)
    btn.grid(column=3, row=0, padx=15, pady=20)
    btn = Button(window, text="Xbar-S控制图", command=clicked)
    btn.grid(column=4, row=0, padx=15, pady=20)

    window.mainloop()   

if __name__ == "__main__":
    xw.Book("每日新增BUG数的控制图.xlsx").set_mock_caller()
    main()
