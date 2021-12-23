import matplotlib
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn import preprocessing
import matplotlib.pyplot as plt
import warnings
import os
from scipy.stats import ttest_ind
import FinanceDataReader as fdr
import yfinance as yf
import streamlit as st
import plotly.express as px
import altair as alt
from PIL import Image

# PS C:\bankrupcy> streamlit run streamlit2.py

st.set_page_config(page_title="Security Selection to Backtest", page_icon=":bar_chart:", layout="wide")

st.title("Security Selection and Market Prediction")
st.write("""
    ### : Using ***K2-Score + F-Score*** and ***Machine Learning*** 
""")
image= Image.open('static/img/main_photo.jpg')
# image= Image.open('img/dl_predic.jpeg')
# image= Image.open('img/ml_predic.png')
st.image(image, use_column_width=True)

st.write("""
***
""")

filename = 'static/data/Kospi_all_Z_F_Score_03(2018,2019).csv'
data = pd.read_csv(filename, encoding='cp949')
# print(data)

data1 = data.rename(columns=
                    {"구분": "회사명",
                     "년도": "거래소코드",
                     "2018": "자산(2018)",
                     "2018.1": "유동자산(2018)",
                     "2018.2": "유동부채(2018)",
                     "2018.3": "이익잉여금(결손금)(2018)",
                     "2018.4": "매출액(수익)(2018)",
                     "2018.5": "매출원가(2018)",
                     "2018.6": "판매비와 관리비(물류원가 등 포함)(2018)",
                     "2018.7": "자본(2018)",
                     "2018.8": "부채(2018)",
                     "2018.9": "당기순이익(2018)",
                     "2018.10": "영업활동으로 인한 현금흐름(간접법)(2018)",
                     "2018.11": "부채비율(2018)",
                     "2018.12": "유동비율(2018)",
                     "2018.13": "발행한 주식총수(2018)",
                     "2018.14": "종가(원)(2018)",
                     "2019": "자산(2019)",
                     "2019.1": "유동자산(2019)",
                     "2019.2": "유동부채(2019)",
                     "2019.3": "이익잉여금(결손금)(2019)",
                     "2019.4": "매출액(수익)(2019)",
                     "2019.5": "매출원가(2019)",
                     "2019.6": "판매비와 관리비(물류원가 등 포함)(2019)",
                     "2019.7": "자본(2019)",
                     "2019.8": "부채(2019)",
                     "2019.9": "당기순이익(2019)",
                     "2019.10": "영업활동으로 인한 현금흐름(간접법)(2019)",
                     "2019.11": "부채비율(2019)",
                     "2019.12": "유동비율(2019)",
                     "2019.13": "발행한 주식총수(2019)",
                     "2019.14": "종가(원)(2019)"},
                    )
print(data1)
data1.describe()
data2 = data1.dropna(axis=0)
# print(data2)
data2.reindex()
data3 = data2.isnull().sum()
# print(data3)
data3 = data2.drop(index=1)
# print(data3)
# data3.info()
for i in range(1, 32):
    if '비율' in data3.columns[i]:
        data3 = data3.astype({data2.columns[i]: 'float'})
    else:
        data3 = data3.astype({data2.columns[i]: 'int64'})

# data3.info()
st.write("""
    ## Automation
    : Security Selection to Backtest
""")

st.write("""
***
""")

st.write("""
    ### K2-Score (부실예측)
    : 부실기업: K2 <-2.30 유보기업: -2.30=< K2 =< 0.75 건전기업: K2 > 0.75
""")

    # ALTMAN K2
data3['X1'] = np.log10(data3['자산(2019)'])
data3['X2'] = np.log10((data3['매출액(수익)(2019)'] / data3['자산(2019)']))
data3['X3'] = data3['이익잉여금(결손금)(2019)'] / data3['자산(2019)']
data3['x4'] = data3['자본(2019)'] / data3['부채(2019)']

data3['K2-Score'] = -18.696 + (1.501 * data3['X1']) + (2.706 * data3['X2']) + (19.760 * data3['X3']) + (
        1.146 * data3['x4'])

# print(data3)
data4 = data3.sort_values(by=['K2-Score'], axis=0, ascending=False)
# print(data4)
data4.loc[:, ['회사명', 'K2-Score']]

st.write("""
    ### F-Score (재무건정성)
    : 9가지 항목 각1점 9점 만점
""")

    # F-Score variables
def y1(a):
    if i >= 0:
        return 1
    else:
        return 0

data4['y1'] = data4.apply(lambda x: y1(x["당기순이익(2018)"]), axis=1)

def y2(a):
    if i >= 0:
        return 1
    else:
        return 0

data4['y2'] = data4.apply(lambda x: y2(x["영업활동으로 인한 현금흐름(간접법)(2018)"]), axis=1)

def y3(a, b, c, d):
    if (a / b) > (c / d):
        return 1
    else:
        return 0

data4['y3'] = data4.apply(lambda x: y3(x["당기순이익(2019)"], x['자산(2018)'], x['당기순이익(2019)'], x['자산(2019)']), axis=1)

def y4(a, b):
    if a > b:
        return 1
    else:
        return 0

data4['y4'] = data4.apply(lambda x: y4(x['영업활동으로 인한 현금흐름(간접법)(2019)'], x['당기순이익(2019)']), axis=1)

def y5(a, b, c, d):
    if b == 0 or d == 0:
        return 0
    elif (a / b) - a > (c / d) - c:
        return 1
    else:
        return 0

data4['y5'] = data4.apply(lambda x: y5(x["매출액(수익)(2018)"], x["매출원가(2018)"], x["매출액(수익)(2019)"], x["매출원가(2019)"]),
                          axis=1)

def y6(a, b):
    if a > b:
        return 1
    else:
        return 0

data4['y6'] = data4.apply(lambda x: y6(x['부채(2018)'], x['부채(2019)']), axis=1)

def y7(a, b):
    if a < b:
        return 1
    else:
        return 0

data4['y7'] = data4.apply(lambda x: y7(x['유동비율(2018)'], x['유동비율(2019)']), axis=1)

def y8(a, b):
    if (a - b) >= 0:
        return 1
    else:
        return 0

data4['y8'] = data4.apply(lambda x: y8(x['발행한 주식총수(2019)'], x['발행한 주식총수(2018)']), axis=1)

def y9(a, b, c, d):
    if (a / b) > (c / d):
        return 1
    else:
        return 0

data4['y9'] = data4.apply(lambda x: y9(x['매출액(수익)(2018)'], x['자산(2018)'], x['매출액(수익)(2019)'], x['자산(2019)']),
                          axis=1)

data4['F-Score'] = data4['y1'] + data4['y2'] + data4['y3'] + data4['y4'] + data4['y5'] + data4['y6'] + data4['y7'] + \
                   data4['y8'] + data4['y9']

print(data4)
data4.loc[:, ['회사명', 'F-Score']]

st.write("""
    ### Market Cap (시가총액 참고)
    : 주식총수 x 종가(2020.12. 단위:원)
""")

# MARKET CAPITALIZATION(시총)
data4['시가총액'] = data4['발행한 주식총수(2019)'] * data4['종가(원)(2019)']
# print(data4)
data5 = data4.sort_values(by=['K2-Score'], axis=0)
# print(data5)
score_sum = data5.loc[:, ['회사명', 'K2-Score', 'F-Score', '시가총액']]  # K2스코어 기준 내림차순
# print(score_sum) 개

# K2스코어 기업 정리분류 부실기업: K2 <-2.30 유보기업: -2.30=< K2 =< 0.75 건전기업: K2 > 0.75:

bins = [-float("inf"), -2.3, 0.75, float("inf")]
labels = ['부실', '유보', '건전']
score_sum['부실/유보회사 합'] = pd.cut(score_sum['K2-Score'], bins=bins, labels=labels)
# print(score_sum['부실/유보회사 합'])
score_sum['부실/유보회사 합'].value_counts()
# print(score_sum['부실/유보회사 합'].value_counts()) 개
data5['부실/유보회사 합'] = data5['K2-Score'] <= 0.75
# print(data5['부실/유보회사 합']) 개
data6 = data5.sort_values(['K2-Score'], ascending=False)
# print(data6)
# data6.loc[:, ['회사명', 'F-Score']]
# print(data6.loc[:, ['회사명', 'F-Score']]) 개
score_sum
# print(score_sum)
F_over_6 = score_sum[score_sum['F-Score'] >= 6][['회사명', 'F-Score']]  # F스코어 6점 이상 회사들
# F_over_6.head()
# print(F_over_6.head()) 개
YB = score_sum[score_sum['부실/유보회사 합'] == '유보'][['회사명', '부실/유보회사 합']]
BS = score_sum[score_sum['부실/유보회사 합'] == '부실'][['회사명', '부실/유보회사 합']]

BS_YB = pd.concat([BS, YB], axis=0)

# BS_YB  # 부실회사+유보회사 aka K2-Score <= 0.75
# print(BS_YB)
F6_BSYB = pd.merge(F_over_6, BS_YB, on=['회사명'], how='inner')

# F6_BSYB  # F스코어 6점 이상 & 부실/유보(K2-Score <= 0.75) 한 회사들의 합
# print(F6_BSYB)

st.write("""
    ***
""")

# st.write("""
#     ### T-test
# """)
#
# # T- test
# poor_comp = score_sum[score_sum['부실/유보회사 합'] == '부실']
# defered_comp = score_sum[score_sum['부실/유보회사 합'] == '유보']
# defered_n_poor_comp = score_sum[score_sum['부실/유보회사 합'] != '건전']
# steady_comp = score_sum[score_sum['부실/유보회사 합'] == '건전']
# print(len(poor_comp))
# print(len(defered_n_poor_comp))
# print(len(steady_comp))
#
# tTestDiffVar_p_vs_dp = ttest_ind(poor_comp['시가총액'], defered_n_poor_comp['시가총액'], equal_var=False)
# tTestDiffVar_p_vs_d = ttest_ind(poor_comp['시가총액'], defered_comp['시가총액'], equal_var=False)
# tTestDiffVar_p_vs_s = ttest_ind(poor_comp['시가총액'], steady_comp['시가총액'], equal_var=False)
# print("poor vs defered_poor : ", tTestDiffVar_p_vs_dp)
# print("poor vs defered : ", tTestDiffVar_p_vs_d)
# print("poor vs steady : ", tTestDiffVar_p_vs_s)
#
# st.write("""
# ***
# """)

st.write("""
    ### Backtest 10yrs 
    : Buy and Hold Strategy (Benchmark-KOSPI)
""")

# 백테스팅(Buy and Hold전략) 10년 / 3년 / 3년 / 1년간 벡테스팅 진행
# Buy & Hold
# 전략: 주식을 매수한 후 장기보유하는 투자전략
#
# 투자전략: 100 % 일시매수후일정기간경과후전량매도
# 아래 df_1 = yf.download(stock[k], start='2009-01-01', end='2019-12-31')함수에서 기간을
#
# 10년: 2009 - 01 - 01~ 2019 - 12 - 31
# 5년: 2015 - 01 - 01~ 2019 - 12 - 31
# 3년: 2017 - 01 - 01~ 2019 - 12 - 31
# 1년: 2019 - 01 - 01~ 2019 - 12 - 31 으로 변경 후
#
# CAGR - 연복리수익률
# MDD - 최대손실가능수익률
# Sharpe
# Ratio(펀드수익률 - 벤치마크수익률) / (해당펀드수익률표준편차)를 계산.그 후 벤치마크(KOSPI지수) 이상의 기록을 가진 기업을 추출

# Backtest 10yrs
bt = "static/data/Backtesting(2009~2019)test.xlsx"
A = pd.read_excel(bt, dtype={'거래소코드': str}, engine='openpyxl')
A['거래소코드'] = A['거래소코드'].astype(str) + '.KS'

print(A)

def df2dict(A):

    df_dict = []

    for k, v in zip(A.iloc[:, 0], A.iloc[:, 1]):  # 행 전체 열은 0,1번열 즉 회사명, 거래소코드 라네! .....
        temp = {}
        temp[k] = v
        df_dict.append(temp)

    return df_dict

stocks = df2dict(A)
# print(stocks) 개

len(stocks)
# print(len(stocks)) 개

myColumns = ['회사명', 'CAGR(%)', 'Sharpe', 'MDD(%)']
df = pd.DataFrame(columns=myColumns)
# print(df)

for stock in stocks:
    for k in stock.keys():
        df_1 = yf.download(stock[k], start='2009-01-01', end='2019-12-31')
        price_df = df_1.loc[:, ['Adj Close']].copy()
        price_df['daily_rtn'] = price_df[
            'Adj Close'].pct_change()  # Buy & Hold Strategy 누적 곱을 계산 한 사례  판다스 cumprod() 함수 사용
        price_df['st_rtn'] = (1 + price_df['daily_rtn']).cumprod()
        historical_max = price_df['Adj Close'].cummax()
        daily_drawdown = price_df['Adj Close'] / historical_max - 1.0
        historical_dd = daily_drawdown.cummin()
        CAGR = price_df.loc['2019-12-30', 'st_rtn'] ** (252. / len(price_df.index)) - 1
        Sharpe = np.mean(price_df['daily_rtn']) / np.std(price_df['daily_rtn']) * np.sqrt(252.)
        MDD = historical_dd.min()

        CAGR_1 = round(CAGR * 100, 2)
        Sharpe_1 = round(Sharpe, 2)
        MDD_1 = round(-1 * MDD * 100, 2)

        data = [[k, CAGR_1, Sharpe_1, MDD_1]]
        df_m = pd.DataFrame(data, columns=myColumns)
        df = df.append(df_m, ignore_index=True)

#    print('CAGR : ',round(CAGR*100,2),'%')
#    print('Sharpe : ',round(Sharpe,2))
#    print('MDD : ',round(-1*MDD*100,2),'%')
# print(df)
df

df2 = pd.merge(A, df)
# df2

df3 = df2.sort_values(by='CAGR(%)', ascending=False)
# df3.head(10)
# print(df3.head(10)) 개

df4 = df3[df3['CAGR(%)'] >= 6.12]
# df4.head(5)
# print(df4.head(5)) 개

df5 = df4['F Score'].value_counts()
# df5
# print(df5) 개

df5.plot(kind='bar', title='10Y')

df3.to_excel('static/data/toExcel/Backtesting_end.xlsx')

st.write("""
***
""")
#
# st.write("""
#     ### BenchMark: KOSPI
# """)
#
# # 벤치마크 : 코스피지수
# df1_krx = fdr.StockListing('KRX')
# len(df1_krx)
# # print(len(df1_krx)) 개
#
# price_df1 = fdr.DataReader('KS11', '2009-01-01', '2019-12-31')  # 티커 및 기간설정
# price_df1['Close'].plot()
#
# price_df1['daily_rtn'] = price_df1['Close'].pct_change()
# #  Buy & Hold Strategy 누적 곱을 계산 한 사례  판다스 cumprod() 함수 사용
# price_df1['st_rtn'] = (1 + price_df1['daily_rtn']).cumprod()
# ## mdd  필요 자료
# historical_max = price_df1['Close'].cummax()
# daily_drawdown = price_df1['Close'] / historical_max - 1.0
# historical_dd = daily_drawdown.cummin()
# historical_dd.plot()
#
# CAGR = price_df1.loc['2019-12-30', 'st_rtn'] ** (252. / len(price_df1.index)) - 1
# Sharpe = np.mean(price_df1['daily_rtn']) / np.std(price_df1['daily_rtn']) * np.sqrt(252.)
# VOL = np.std(price_df1['daily_rtn']) * np.sqrt(252.)
# MDD = historical_dd.min()
# print('CAGR : ', round(CAGR * 100, 2), '%')
# print('Sharpe : ', round(Sharpe, 2))
# print('VOL : ',round(VOL*100,2),'%')
# print('MDD : ', round(-1 * MDD * 100, 2), '%')
#
# st.write("""
# ***
# """)

st.write("""
    ### Backtest 5yrs
    : Buy and Hold Strategy (Benchmark-KOSPI)
""")

# Backtest 5yrs
for stock in stocks:
    for k in stock.keys():
        df_2 = yf.download(stock[k], start='2015-01-01', end='2019-12-31')
        price_df2 = df_2.loc[:, ['Adj Close']].copy()
        price_df2['daily_rtn'] = price_df2[
            'Adj Close'].pct_change()  # Buy & Hold Strategy 누적 곱을 계산 한 사례  판다스 cumprod() 함수 사용
        price_df2['st_rtn'] = (1 + price_df2['daily_rtn']).cumprod()
        historical_max2 = price_df2['Adj Close'].cummax()
        daily_drawdown2 = price_df2['Adj Close'] / historical_max2 - 1.0
        historical_dd2 = daily_drawdown2.cummin()
        CAGR = price_df2.loc['2019-12-30', 'st_rtn'] ** (252. / len(price_df2.index)) - 1
        Sharpe = np.mean(price_df2['daily_rtn']) / np.std(price_df2['daily_rtn']) * np.sqrt(252.)
        MDD = historical_dd2.min()

        CAGR_1 = round(CAGR * 100, 2)
        Sharpe_1 = round(Sharpe, 2)
        MDD_1 = round(-1 * MDD * 100, 2)

        data = [[k, CAGR_1, Sharpe_1, MDD_1]]
        df_m = pd.DataFrame(data, columns=myColumns)
        df6 = df.append(df_m, ignore_index=True)
df6
# print(df6)

#    print('CAGR : ',round(CAGR*100,2),'%')
#    print('Sharpe : ',round(Sharpe,2))
#    print('MDD : ',round(-1*MDD*100,2),'%')

df7 = pd.merge(A, df6)

# df7
# print(df7)

df8 = df7.sort_values(by='CAGR(%)', ascending=False)

df8.head(10)
# print(df8.head(10))

df9 = df8[df8['CAGR(%)'] >= 2.74]
# print(df9.head(5))

st.write("""
***
""")

st.write("""
    ### Backtest 3yrs
    : Buy and Hold Strategy (Benchmark-KOSPI)
""")

# Backtest 3yrs
for stock in stocks:
    for k in stock.keys():
        df_3 = yf.download(stock[k], start='2017-01-01', end='2019-12-31')
        price_df3 = df_3.loc[:, ['Adj Close']].copy()
        price_df3['daily_rtn'] = price_df3[
            'Adj Close'].pct_change()  # Buy & Hold Strategy 누적 곱을 계산 한 사례  판다스 cumprod() 함수 사용
        price_df3['st_rtn'] = (1 + price_df3['daily_rtn']).cumprod()
        historical_max3 = price_df3['Adj Close'].cummax()
        daily_drawdown3 = price_df3['Adj Close'] / historical_max3 - 1.0
        historical_dd3 = daily_drawdown3.cummin()
        CAGR = price_df3.loc['2019-12-30', 'st_rtn'] ** (252. / len(price_df3.index)) - 1
        Sharpe = np.mean(price_df3['daily_rtn']) / np.std(price_df3['daily_rtn']) * np.sqrt(252.)
        MDD = historical_dd3.min()

        CAGR_1 = round(CAGR * 100, 2)
        Sharpe_1 = round(Sharpe, 2)
        MDD_1 = round(-1 * MDD * 100, 2)

        data = [[k, CAGR_1, Sharpe_1, MDD_1]]
        df_m = pd.DataFrame(data, columns=myColumns)
        df10 = df.append(df_m, ignore_index=True)
df10

#    print('CAGR : ',round(CAGR*100,2),'%')
#    print('Sharpe : ',round(Sharpe,2))
#    print('MDD : ',round(-1*MDD*100,2),'%')
# print(df10)
df11 = pd.merge(A, df10)
# df11
# print(df11)
df12 = df11.sort_values(by='CAGR(%)', ascending=False)
# df12.head(10)
# print(df12.head(10))

df13 = df12[df12['CAGR(%)'] >= 2.83]
# df13.head(5)
# print(df13.head(5))

st.write("""
***
""")


st.write("""
    ### Backtest 1yr
    : Buy and Hold Strategy (Benchmark-KOSPI)
""")

# Backtest 1yr
for stock in stocks:
    for k in stock.keys():
        df_4 = yf.download(stock[k], start='2019-01-01', end='2019-12-31')
        price_df4 = df_4.loc[:, ['Adj Close']].copy()
        price_df4['daily_rtn'] = price_df4[
            'Adj Close'].pct_change()  # Buy & Hold Strategy 누적 곱을 계산 한 사례  판다스 cumprod() 함수 사용
        price_df4['st_rtn'] = (1 + price_df4['daily_rtn']).cumprod()
        historical_max4 = price_df4['Adj Close'].cummax()
        daily_drawdown4 = price_df4['Adj Close'] / historical_max4 - 1.0
        historical_dd4 = daily_drawdown4.cummin()
        CAGR = price_df4.loc['2019-12-30', 'st_rtn'] ** (252. / len(price_df4.index)) - 1
        Sharpe = np.mean(price_df4['daily_rtn']) / np.std(price_df4['daily_rtn']) * np.sqrt(252.)
        MDD = historical_dd4.min()

        CAGR_1 = round(CAGR * 100, 2)
        Sharpe_1 = round(Sharpe, 2)
        MDD_1 = round(-1 * MDD * 100, 2)

        data = [[k, CAGR_1, Sharpe_1, MDD_1]]
        df_m = pd.DataFrame(data, columns=myColumns)
        df14 = df.append(df_m, ignore_index=True)
df14

#    print('CAGR : ',round(CAGR*100,2),'%')
#    print('Sharpe : ',round(Sharpe,2))
#    print('MDD : ',round(-1*MDD*100,2),'%')
# print(df14)
df15 = pd.merge(A, df14)
# df15
# print(df15)

df16 = df15.sort_values(by='CAGR(%)', ascending=False)
# df16.head(10)
# print(df16.head(10))

df17 = df16[df16['CAGR(%)'] >= 9.58]
# df17.head(5)
# print(df17.head(5))

# st.write("""
# ***
# """)
#
#
# st.write("""
#     ### Download to Excel file
# """)

xlsx_dir = 'static/data/toExcel/BnH_comp_list.xlsx'
with pd.ExcelWriter(xlsx_dir) as writer:
    df4.to_excel(writer, sheet_name='10년_백테스트', index=False, encoding='utf-8-sig')
    df9.to_excel(writer, sheet_name='5년_백테스트', index=False, encoding='utf-8-sig')
    df13.to_excel(writer, sheet_name='3년_백테스트', index=False, encoding='utf-8-sig')
    df17.to_excel(writer, sheet_name='1년_백테스트', index=False, encoding='utf-8-sig')
