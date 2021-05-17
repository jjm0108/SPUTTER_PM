import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

from sklearn.model_selection import train_test_split
import CONSTANTS as cs
import visualization as vs
from scipy.stats import mode

# 이어지는 step 공정 하나로 묶어주는 함수
def summarize_by_step(df, chamber):
    print(df['step'].value_counts())
    STEP = 0
    START = None

    # 데이터 요약 시 필요 없는 파라미터는 제거
    unwanted = ["slot", "step", "stepname"]
    params = [param for param in list(df.select_dtypes('number').columns) if param not in unwanted]

    columns = ['start','end', 'step', 'recipe'] + params

    temp=pd.DataFrame(columns=params)
    summary_df = pd.DataFrame(columns=columns)
    print('total data length : ', len(df))

    for index in df.index:
        # 요약 진행 상황 표시
        if index % 10000 == 0: print('processing ...', index)

        row = df.loc[index][params]

        # 첫번째 데이터에 대한 처리
        if index <1 : 
            START = df.loc[index]['time']
            temp = temp.append(row, ignore_index=True)

        # step이 달라졌을 때
        elif df.loc[index]['step'] != STEP : 
            # 평균(mean)으로 요약
            if len(temp) > 1 : temp = temp.mean()

            # start: 해당 step의 시작 시간, end: 해당 step의 종료 시간
            temp['start'] = START
            temp['end'] = df.loc[index-1]['time']
            temp['step'] = STEP
            temp['recipe'] = df.loc[index-1]['recipe']
            summary_df = summary_df.append(temp, ignore_index=True)

            START = df.loc[index]['time']
            temp = pd.DataFrame(columns=params).append(row, ignore_index=True)
        else:
            temp = temp.append(row, ignore_index=True)
        STEP = df.loc[index]['step']
    
    # 요약한 데이터를 csv 파일로 출력
    summary_df.to_csv('CH_'+chamber+" summary_by_step.csv", sep=',')

    return summary_df

def summary_analysis(data, chamber, param):
    step_l = list(set(data['step']))
    print(data['step'].value_counts())

    diff={param:[], "current":[], "previous":[]}
    index_l ={}
    # 각 step별 데이터들의 index 리스트
    for step in step_l:
        index_l[step] = list(data[data['step']==step].index)


    for index in data.index:
        if index % 1000 == 0: print('processing ...', index)
        step = data.loc[index]['step']

        # 동일한 step을 가진 데이터 중 이전 주기 데이터
        previous_index = index_l[step].index(index)-1
        if previous_index<0 : previous_index=0
        previous_index = index_l[step][previous_index]

        current_value = data.loc[index][param]
        previous_value = data.loc[previous_index][param]
        
        diff["current"].append(current_value)
        diff["previous"].append(previous_value)

        diff[param].append(current_value- previous_value)
        

    diff_df=pd.DataFrame(diff)
    
    diff_df['time']=data['time']
    diff_df['step']=data['step']
    print(diff_df)
    vs.plot (diff_df, param, chamber, show = True, save = False, save_folder = "")

def str2date(str_date):
    # 함수 선언 : 시간 값 string -> datetime으로 변환
    # e.g. '2021-04-15 오후 5:30' -> '2021-04-15 17:30:00' 으로 변경됨'
    kor2eng_date = str_date

    if str_date.find('오전') > -1 :
        kor2eng_date = str_date.replace("오전", "AM")

    elif str_date.find('오후') > -1 :
        kor2eng_date = str_date.replace("오후", "PM")

    return pd.to_datetime(kor2eng_date, format='%Y-%m-%d %p %I:%M:%S')

# 결측값, 기초적인 기술통계를 알아보는 함수
def data_analysis(df):
    
    print('data : \n{}'.format(df))
    print('*'*50)
    print('data shape : {}'.format(df.shape))
    print('\ndata columns : \n{}'.format(df.columns))
    print('\ndata type : \n{}'.format(df.dtypes))

    print('*'*50)
    print('data null check : \n{}'.format(df.isnull().sum().sort_values(ascending=False)))
    
    print('*'*50)
    print('data describe : \n{}'.format(df.describe()))

    # 결측값이 들어있는 행 전체 삭제하기(delete row with NaN)
    df.dropna(axis=0)
    # 결측값이 들어있는 열 전체 삭제하기 (delete column with NaN)
    df.dropna(axis=1)

    # 특정 행 또는 열을 대상으로 결측값이 들어있으면 제거 (delete specific row or column with missing values)
    df['stepname'].dropna()

# 특정 레시피 제거, 결측값 제거 등 전처리를 진행하는 함수
def preprocessing(df):
    # 특정 레시피 제거
    idx = df[df['recipe']=='TIW_SHUT_A101'].index
    df=df.drop(idx)

    # step -1 제거
    idx = df[df['step']==-1].index
    df=df.drop(idx)

# 정상 구간을 계산, 시각화하는 함수
def normal_range(df):
    for parameter in df.select_dtypes('number').columns :
    #for parameter in ["Ch 1 Onboard Cryo Temp 2nd Stage"] :
        value_count = df[parameter].value_counts(normalize=True)
        normal_range = []
        target_percent = 0.995
        sum_percent = 0 
        for value, percent in value_count.items():
            if sum_percent < target_percent :
                sum_percent+=percent
                normal_range.append(value)
            else: break
        print(parameter, "normal range :",min(normal_range),"~",max(normal_range))
        #print(value_count)

        plt.figure(figsize=(16,8), dpi=100)
        plt.title(parameter+" : percent ="+str(target_percent))
        col = np.where(df[parameter]> max(normal_range) or df[parameter]< min(normal_range),'red','grey')
        plt.scatter(df.time, df[parameter], color=col)
        #vs.plot_windows()
        #plt.hlines(y=min(normal_range), color='grey', linewidth=3.0, xmin=df.time[0], xmax=df.time[len(df.time)-1], label = "normal range")
        #plt.hlines(y=max(normal_range), color='grey', linewidth=3.0, xmin=df.time[0], xmax=df.time[len(df.time)-1])
        plt.axhspan(min(normal_range), max(normal_range), facecolor='grey', alpha=0.5, label ='normal range')
        plt.legend(loc='upper right')

        
        save_folder = 'normal_range_plot'
        os.makedirs(save_folder, exist_ok=True)
        #plt.savefig(os.path.join(save_folder, parameter+'(red_grey).png'), bbox_inches='tight', pad_inches=0.0)
        plt.show()
        #df[parameter] = df[parameter]-percentile
        #df.loc[df[parameter] <= 0] = 0
        #vs.plot(df[df[parameter]> percentile], parameter)
        #vs.plot(df, parameter, percentile=True)

    #output_f = "./99. src/summary data/filtered(difference)/percentile_difference_"+str(percent)+".csv"
    #df.to_csv(output_f)

def latency_analysis(df):
    max_latency = 100
    save_path = '4. 개인 분석 자료/latency_analysis'
    os.makedirs(save_path, exist_ok=True)

    for param_A in df.select_dtypes('number').columns :
        result_df = pd.DataFrame(columns=["parameter","latency", "high_corr"])
        for param_B in df.select_dtypes('number').columns :
            corr_df = pd.DataFrame(df[param_A])
            for i in range(max_latency):
                corr_df[str(i)]=df[param_B].shift(i)

            corr_df = corr_df.corr()
            #print(corr_df)
            highest_corr = corr_df[param_A].sort_values(ascending=False)[1]
            lowest_corr = corr_df[param_A].sort_values(ascending=True)[1]
            corr = lowest_corr if abs(lowest_corr)>abs(highest_corr) else highest_corr
    
            try : latency =  corr_df[corr_df[param_A]==corr].index.values[0]
            except IndexError : latency = 0
            result_df = result_df.append({"parameter":param_B, "latency": latency, "high_corr":corr}, ignore_index=True)

        print(result_df)
        
        result_df.to_csv(os.path.join(save_path, param_A+'.csv'), sep=",")

    #sns.heatmap(data = corr_df.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
    #plt.show()

def DynamicTimeWarping(df):
    import numpy as np
    from scipy.spatial.distance import euclidean

    from fastdtw import fastdtw

    param = 'Ch 1 MFC 2 Flow x1000'
    x = df[param]
    for parameter in df.select_dtypes('number').columns :
        y = df[parameter]
        distance, path = fastdtw(x, y, dist=euclidean)
        print(param+" and "+parameter+"'s dtw: "+str(distance))

# 다양한 시각화를 진행할 수 있는 함수
def visualization(df, save_folder, chamber, show=True, save = False):
    # 실수 형식의 데이터만 분석
    for parameter in df.select_dtypes('number').columns :
        #line plot (by time)
        vs.plot(df, parameter, chamber, show=show, save = save, save_folder = save_folder)

        # scatter plot (by recipe)
        vs.scatter_plot(df, parameter, chamber, group_by = "Ch "+chamber+" Running Recipe", show = show, save = save, save_folder = save_folder)

        # scatter plot (by step)
        vs.scatter_plot(df, parameter, chamber, group_by = "Ch "+chamber+" Step Number", show = show, save = save, save_folder = save_folder)

        # box plot (by step)
        vs.box_plot(df, parameter, chamber, group_by = "Ch "+chamber+" Step Number", show = show, save = save, save_folder = save_folder)

        # historam
        vs.histogram(df, parameter, show=show, save=save, save_folder = save_folder)

# 상관계수를 계산, 시각화하는 함수
def correlation(df, chamber, save_folder):
    print(df.corr())

    #heatmap plot (correlation check)
    vs.heatmap(df, chamber, show = False, save = True, save_folder = save_folder)

# 고장 유형 : target과 shutter disc의 접촉으로 인해 DC 전압이 0이 될 때, 문제 발생 가능
def DC_voltage_checking(df, chamber):
    print(df.columns)
    param = 'PVD Ch 1 DC Voltage'
    print(df.loc[df[param]==0])


def periodity_analysis(chamber):
    parameter = 'Ch C MFC 2 Flow x1000'
    df = pd.read_csv(parameter+" summary_by_step.csv")
    
    df['time'] = pd.to_datetime(df['time'])
    print(df)
    for step in df.columns[1:]:
        vs.plot (df, step, chamber, show = True, save = False, save_folder = "")


def DataAnalysis():
    parameters = [param for param in data.select_dtypes('number').columns if param not in ['slot', 'step', 'stepname','Ch 1 Wafer Source Slot', 'Ch 1 Step Number' ]]
    # #print(parameters)
    
    # param_1 = [
    #     'Ch 1 Pressure In mtorr', 
    #     'Ch 1 Pressure In ntorr',
    #     'Ch 1 Pressure In utorr' 

    #     # 'Ch 1 Onboard Cryo Temp 1st Stage',
    #     # 'Ch 1 Onboard Cryo Temp 2nd Stage'

    #     # 'PVD Ch 1 DC Current', 
    #     # 'PVD Ch 1 DC Voltage', 
    #     # 'PVD Ch 1 DC Power Actual', 

    #     # 'Ch 1 Foreline Pressure', 
    #     # 'Vent Gas Pressure', 
    #     # 'Slit Pneumatic Air Pressure', 
        
    #     # 'Ch 1 MFC 1 Flow x1000', 
    #     # 'Ch 1 MFC 2 Flow x1000'
    
    #     ]
    # param_C = [
    #     'Ch C Pressure In utorr', 
    #     'Ch C Pressure In ntorr', 
    #     'Ch C Pressure In mtorr',

    #     'Ch C DC Bias', 
        
    #     'Ch C RF1 Forward Power', 
    #     'Ch C RF1 Reflected Power', 
    #     'Ch C RF2 Forward Power', 
    #     'Ch C RF2 Reflected Power', 

    #     'Ch C MFC 1 Flow x1000', 
    #     'Ch C MFC 2 Flow x1000', 
        
    #     'Ch C Foreline Pressure', 
    #     'Vent Gas Pressure',  
    #     'Slit Pneumatic Air Pressure'

    #     # 'Chamber C Preventive Maintenance RF Used'
    # ]

    # if chamber == "1":
    #     X = data[param_1]
    # elif chamber == "C":
    #     X = data[param_C]
    # else:
    #     X = data[param_D]

    # split_percent = 0.4
    # x_train, x_test = data_split(data, chamber, split_percent, X, middle=False, visualization = True )
    # x_train = X[:int(len(X)*0.1)]
    # x_test = X[int(len(X)*0.2):int(len(X)*0.4)]
    
    # print("data splitted")

    # #AnomalyDetection(data, chamber, "IsolationForest", split_percent, x_train, x_test, scoring=False, contamination =0.0001, show_params = False, show= True, save=False)
   

# 학습 데이터와 테스트 데이터로 구분하는 함수
def data_split(df, chamber, percent, x, y = None, middle=False, visualization = False, ):

    if middle == True:
        before_x = x[:int(len(x)*percent)]
        after_x = x[-int(len(x)*percent):]
        x_train = pd.concat([before_x, after_x])
        x_test = x[int(len(x)*percent):-int(len(x)*percent)]

        if y!=None and not y.empty:
            before_y = y[:int(len(y)*percent)]
            after_y = y[-int(len(y)*percent):]
            y_train = pd.concat([before_y, after_y])
            y_test = y[int(len(y)*percent):-int(len(y)*percent)]
            return x_train, x_test, y_train, y_test

        # data split visualization
        if visualization == True:
            plt.figure(figsize=(16,8), dpi=100)
            time = df['time']
            param = x.columns[0]
            plt.title(param+" Data Split : "+str(percent))
            plt.plot(time[:int(len(time)*percent)], before_x[param], color ='grey')
            plt.plot(time[int(len(time)*percent):-int(len(time)*percent)], x_test[param])
            plt.plot(time[-int(len(time)*percent):], after_x[param], color ='grey')
            vs.plot_windows(chamber)
            plt.show()
    else:
        x_train = x[:int(len(x)*percent)]
        x_test = x[int(len(x)*percent):]

        if y!=None and not y.empty:
            y_train = y[:int(len(y)*percent)]
            y_test = y[int(len(y)*percent):]
            return x_train, x_test, y_train, y_test
        
        # data split visualization
        if visualization == True:
            plt.figure(figsize=(16,8), dpi=100)
            time = df['time']
            param = x.columns[0]
            plt.title(param+" Data Split : "+str(percent))
            plt.plot(time[:int(len(time)*percent)], x_train[param], color ='grey')
            plt.plot(time[int(len(time)*percent):], x_test[param])
            vs.plot_windows(chamber)
            plt.show()

    ## scikit-learn easy method
    #x_train, x_test = train_test_split(x, test_size=0.25, random_state=0, shuffle=False)

    return x_train, x_test

def RandomForest(df, chamber):
    time = df['time']
    # 연속 분포
    X = df[['Ch 1 Pressure In utorr','Ch 1 Pressure In mtorr']]
    Y = df[['Ch 1 MFC 2 Flow x1000'] ]

    # # scikit-learn easy method
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42, shuffle=False)
    percent = 0.2
    x_train, x_test, y_train, y_test = data_split(data, chamber, percent, X, Y)

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(x_train)

    # print(x_train)
    # print(x_train_scaled)


    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(n_estimators = 100, random_state = 0)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)

    score = regr.score(x_test, y_test)
    print('정확도: %.2f'%score)

    plt.figure(figsize=(16,8), dpi=100)
    time = df['time']
    plt.title("Random Forest prediction score : "+str(score))
    plt.plot(time[int(len(time)*percent):-int(len(time)*percent)], y_test, c='red', label="label")
    plt.plot(time[int(len(time)*percent):-int(len(time)*percent)], y_pred, c='grey', label="prediction")
    plt.legend(loc="upper right")

    vs.plot_windows(chamber)
    plt.show()

def AnomalyDetection(df, chamber, model, percent, x_train, x_test, scoring=True, contamination =0.001, show_params = False, show= True, save=False):
    slicing = int(len(df)*percent)

    if model == "extendedIsolationForest":
        import eif as iso
        # ExtensionLevel=0 is the same as regular Isolation Forest
        clf = iso.iForest(x_train.values, ntrees=200, sample_size=256, ExtensionLevel=1)
        print("fitting finished")
        train_pred = clf.compute_paths(X_in=x_train.values)
        test_pred = clf.compute_paths(X_in=x_test.values)
        print("scoring finished")
    
    else:
        if model == "IsolationForest":
            from sklearn.ensemble import IsolationForest
            # contamination : the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the sample
            clf = IsolationForest(n_estimators=50, contamination= contamination, random_state=0)

        elif model == "LocalOutlierFactor":
            from sklearn.neighbors import LocalOutlierFactor
            # If you really want to use neighbors.LocalOutlierFactor for novelty detection, 
            # i.e. predict labels or compute the score of abnormality of new unseen data, 
            # you can instantiate the estimator with the novelty parameter set to True before fitting the estimator.
            clf = LocalOutlierFactor(n_neighbors=5, novelty=True)

        elif model == "OneClassSVM":
            from sklearn.svm import OneClassSVM
            clf = OneClassSVM(gamma='auto')
        else:
            clf = None
            print("model selection error")
            return 

        clf.fit(x_train)
        print("fitting finished")

        # pred = -(clf.predict(x_test)) # predict: Returns 1 for outliers and -1 for inliers
        # pred = -(clf.score_samples(x_test[features])) # score_samples : Returns anomaly score 0~1 (0 for normal, 1 for anomal)

        if scoring == True: 
            train_pred = -(clf.score_samples(x_train))
            test_pred = -(clf.score_samples(x_test))
        else :
            train_pred = -(clf.predict(x_train))
            test_pred = -(clf.predict(x_test))  

        print("scoring finished")

          

    # result visualization
    time = df['time']
    if show_params ==  True :  
        fig, ax = plt.subplots(len(x_train.columns)+1,1, figsize=(16,8),constrained_layout=True)
        for i in range(len(x_train.columns)):
            param = x_train.columns[i]
            ax[i].set_title(param)
            ax[i].plot(time[:slicing], x_train[param][:slicing], color="grey", label="training")
            ax[i].plot(time[-slicing:], x_train[param][-slicing:], color="grey")
            ax[i].plot(time[slicing:-slicing], x_test[param], label="test")
            ax[i].legend(loc='upper right')

        ax[len(X.columns)].set_title('anomaly result from '+model)
        ax[len(X.columns)].plot(time[:slicing], train_pred[:slicing], c='grey', label="training")
        ax[len(X.columns)].plot(time[-slicing:], train_pred[-slicing:], c='grey')
        ax[len(X.columns)].plot(time[slicing:-slicing], test_pred, c='red', label="prediction")
        vs.plot_windows(chamber)
        ax[len(X.columns)].legend(loc='upper right')
    else:
        plt.figure(figsize=(16,8), dpi=100)

        plt.title('anomaly score from '+model)
        plt.plot(time[:slicing], train_pred[:slicing], c='grey', label="training")
        plt.plot(time[-slicing:], train_pred[-slicing:], c='grey')
        plt.plot(time[slicing:-slicing], test_pred, c='red', label="prediction")
        vs.plot_windows(chamber)
        plt.legend(loc='upper right') 
        
        

    # 그래프를 모니터 화면에 직접 보고 싶을 때
    if show == True : plt.show()

    # png 파일로 그래프 결과를 저장하고 싶을 때
    elif save == True :
        save_path = '4. 개인 분석 자료/plot result/CH'+chamber+"/isolation_forest"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, param+'.png'), bbox_inches='tight', pad_inches=0.0)

def ImportFDCData(chamber, original=True):
    if original == True:
        # data read
        data = pd.read_csv("./3. SPU01-FDC DATA/CH_"+chamber+"/SPU-01 CH_"+chamber+" FDC DATA.csv")
        # time-series data (시계열 데이터인 경우, 타임스탬프를 시간 형식으로 바꿔주는 작업 필요)
        data['time'] = pd.to_datetime(data['time'],format='%Y-%m-%d %p %I:%M:%S')

    else:
        # data read
        data = pd.read_csv("./CH_"+chamber+" summary_by_step.csv")
        data['time'] = pd.to_datetime(data['start'])
    return data

def DB_connecting(recent=10000):
    # pip install pymysql

    import pymysql
    import pandas as pd

    #필요한 기본 DB 정보
    host = "192.168.100.203" #접속할 db의 host명
    user = "root" #접속할 db의 user명
    port = 3306
    pw = "nepes" #접속할 db의 password
    db = "sputter_db" #접속할 db의 table명 (실제 데이터가 추출되는 table)


    #DB에 접속
    conn = pymysql.connect( host= host,
                            user = user,
                            port = port,
                            password = pw,
                            db = db)


    #실제 사용 될 sql쿼리 문
    sql = "SELECT * FROM sputter_db.`SPU-01-DETAIL`;"



    df = pd.read_sql_query(sql, conn)
    if len(df)>= recent : df= df[-recent:]
    print("DB connection finished")

    #db 접속 종료
    conn.close()

    # null data 삭제
    nonNull_params = []

    for column in df.columns:
        if df[column].isnull().sum() != len(df):
            nonNull_params.append(column)

    df = df[nonNull_params]

    # parameter naming (숫자로 되어있는 열을 공정 파라미터로 바꿔줌)
    columns = {"TIMESTAMP":"time"}
    varID_df = pd.read_csv("./parameter_variableID.csv")
    for param in df.columns[4:]:
        columns[param] = str(varID_df[varID_df["VARIABLE_ID"]==int(param)]["PARAMETER_NAME"].values[0])
    df.rename(columns=columns, inplace = True)

    # Timestamp 형식으로 바꾸기
    df['time'] = pd.to_datetime(df['time'])
    
    return df


if __name__ == '__main__':
    import datetime
    now = str(datetime.datetime.now())

    # 분석할 데이터의 챔버를 설정
    Chamber_list = ["C", "D", "1", "2", "3", "4"]
    chamber = "1"

    # DB 데이터 사용 시
    recent = 300000 # 최근 몇 개까지의 데이터를 가져올지 설정
    data = DB_connecting(recent=recent)
    save_path = '8. DB raw 데이터 분석 자료/'
    os.makedirs(save_path+"RAW DATA", exist_ok=True)
    data.to_csv(save_path+"RAW DATA/"+now+".csv" , sep=',')
    

    # FDC csv 파일 사용 시
    # data = ImportFDCData(chamber, original=True)

    for chamber in Chamber_list:
        # 분석 결과 저장 파일 설정
        save_folder = save_path+'CH_'+chamber+"/"+now
        os.makedirs(save_folder, exist_ok=True)

        # 특정 챔버 데이터 가져오기
        param_by_chamber=['time']
        for param in data.columns:
            if "Chamber "+chamber in param or "Ch "+chamber in param: param_by_chamber.append(param)

        df = data[param_by_chamber]
        print(df.columns)

        visualization(df, save_folder, chamber, show= False, save=True)

   

