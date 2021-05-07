import seaborn as sns
import CONSTANTS as cs
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import os

def plot (df, parameter, chamber, show = True, save = False, save_folder = ""):
    # 출력 화면크기 설정
    plt.figure(figsize=(16,8), dpi=100)
    # 제목 설정
    plt.title(parameter)
    #plot_windows(chamber)
    # 그래프 그리기
    plt.plot(df.time, df[parameter])
    # 라벨 표시
    plt.legend(loc='upper right')

    # 그래프를 모니터 화면에 직접 보고 싶을 때
    if show == True : plt.show()

    # png 파일로 그래프 결과를 저장하고 싶을 때
    if save == True and save_folder!="": 
        os.makedirs(save_folder+"/plot", exist_ok=True)
        plt.savefig(os.path.join(save_folder+"/plot", parameter+'.png'), bbox_inches='tight', pad_inches=0.0)

# group별로 쪼개서 여러번 plot
def multiple_plot(df, parameter, chamber, group_by = None, each = False, show = True, save = False, save_folder = ""):
    if each == False : 
        plt.figure(figsize=(16,8), dpi=100)
        plt.title(parameter)
        for group in list(set(df[group_by])):
            plt.plot(df.loc[df[group_by]==group,"time"], df.loc[df[group_by]==group,parameter], label = group) 
        plot_windows(chamber)
        plt.legend(loc='upper right')
        plt.show()
    else : 
        for group in list(set(df[group_by])):
            plt.figure(figsize=(16,5), dpi=100)
            plt.title(parameter+" : group by "+str(group))
            plot_windows(chamber)
            plt.plot(df.loc[df[group_by]==group,"time"], df.loc[df[group_by]==group,parameter], label = group) 
            plt.legend(loc='upper right')
            plt.show()

def scatter_plot(df, parameter, chamber, group_by = None, show = True, save = False, save_folder = ""):
    plt.figure(figsize=(16,8), dpi=100)
    plt.title(parameter)
    #plot_windows(chamber)

    if group_by == "recipe" : 
        if chamber in ["C", "D"] : palette = cs.chCD_colors
        else : palette = "deep"
    elif group_by == "step" : palette = "deep"
    else : palette = None

    # hue: 그룹화의 기준이 될 변수, palette: 색상 설정
    sns.scatterplot(x='time', y=parameter,  hue=group_by, palette = "deep" ,data=df)
    plt.legend(loc='upper right')

    if show == True : plt.show()
    if save == True and save_folder!="": 
        os.makedirs(save_folder+"/scatter_by_"+group_by, exist_ok=True)
        plt.savefig(os.path.join(save_folder+"/scatter_by_"+group_by, parameter+'.png'), bbox_inches='tight', pad_inches=0.0)

def histogram(df, parameter, show = True, save = False, save_folder = ""):
    plt.figure(figsize=(16,8), dpi=100)
    plt.title(parameter)
    df[parameter].hist()
    if show == True : plt.show()
    if save == True and save_folder!="": 
        os.makedirs(save_folder+"/histogram", exist_ok=True)
        plt.savefig(os.path.join(save_folder+"/histogram", parameter+'.png'), bbox_inches='tight', pad_inches=0.0)

def density_plot(df, parameter, chamber, show = True, save = False, save_folder = ""):
    plt.figure(figsize=(16,5), dpi=100)
    plt.title(parameter)
    df[parameter].plot(kind='kde', color='grey')
    plt.show()

def box_plot(df, parameter, chamber, group_by = None, show = True, save = False, save_folder = ""):
    plt.figure(figsize=(16,8), dpi=100)
    plt.title(parameter)
    sns.boxplot(x=group_by, y=parameter, data=df)

    if show == True : plt.show()
    if save == True and save_folder!="": 
        os.makedirs(save_folder+"/box_plot_by_"+group_by, exist_ok=True)
        plt.savefig(os.path.join(save_folder+"/box_plot_by_"+group_by, parameter+'.png'), bbox_inches='tight', pad_inches=0.0)



def heatmap (df, chamber, show = True, save = False, save_folder = ""):
    plt.figure(figsize=(12,10), dpi=100)
    plt.title("CH_"+chamber+" correlation heatmap")

    # annot: True일때, correlation 값을 표시, fmt: annot 표시 형식, cmap: 색상 타입 (seaborn heatmap페이지 참고)
    sns.heatmap(data = df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')

    if show == True : plt.show()
    if save == True and save_folder!="": 
        os.makedirs(save_folder+"/correlation_heatmap", exist_ok=True)
        plt.savefig(os.path.join(save_folder+"/correlation_heatmap", 'correlation_heatmap.png'), bbox_inches='tight', pad_inches=0.0)
        corr = df.corr()
        corr.to_csv(os.path.join(save_folder+"/correlation_heatmap", "CH_"+chamber+"correlation.csv"), sep=',')

def plot_windows(chamber, all = False):
    with open('./99. src/SPUTTER_PM/combined_windows.json', 'r') as f:
        json_data = json.load(f)
    if all == True :
        chambers = ["1", "C", "D"]
        for chamber in chambers:
            i=0
            for window in json_data['CH_'+chamber]:
                span_start = pd.Timestamp(window[0])
                span_end = pd.Timestamp(window[1])
                plt.axvspan(span_start, span_end, facecolor='red', alpha=0.5, label = "failure interval_"+chamber if i==0 else "")
                i+=1
    else:     
        i=0
        for window in json_data['CH_'+chamber]:
            span_start = pd.Timestamp(window[0])
            span_end = pd.Timestamp(window[1])
            plt.axvspan(span_start, span_end, facecolor='red', alpha=0.5, label = "failure interval" if i==0 else "")
            i+=1
        

#plt.grid() 
