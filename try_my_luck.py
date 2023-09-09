import websocket, json
import pandas as pd
from binance.client import Client
from binance.enums import *
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import random
import talib
import re
import FinanceDataReader as fdr
import joblib

SOCKET = "wss://fstream.binance.com/ws/btcusdt@kline_1m"

closes=[]

client = Client('4ZCYiErLOgTRkEEhG8nXAxrUYhe946ZKPthKPmJh2qzdUgInYmWbNWJh6L5yRmfo', '0mDyZfR2fFQpvnDbiBB5tzF7efbZ2jn3LMxwodmSaz3VnaUHROFU08ZFfRbp71Zn')

# long_phase=1
# short_phase=1



def get_patterns(df):
    patterns = ['CDLCLOSINGMARUBOZU', 'CDLDOJI', 'CDLDRAGONFLYDOJI', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
                'CDLHANGINGMAN', 'CDLHIGHWAVE', 'CDLINVERTEDHAMMER', 'CDLLONGLEGGEDDOJI', 'CDLLONGLINE',
                'CDLMARUBOZU', 'CDLRICKSHAWMAN', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 'CDLTAKURI']

    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    # create columns for each pattern
    for candle in patterns:
        df[candle] = getattr(talib, candle)(op, hi, lo, cl)

    return df


def get_indicators(df):
    op = df['Open']
    hi = df['High']
    lo = df['Low']
    cl = df['Close']
    vo = df['Volume']

    df['ADX'] = talib.ADX(hi, lo, cl, timeperiod=14)
    df['APO'] = talib.APO(cl, fastperiod=12, slowperiod=26, matype=0)
    df['BOP'] = talib.BOP(op, hi, lo, cl)
    df['CCI'] = talib.CCI(hi, lo, cl, timeperiod=14)
    df['DX'] = talib.DX(hi, lo, cl, timeperiod=14)
    df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(cl, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MFI'] = talib.MFI(hi, lo, cl, vo, timeperiod=14)
    df['MOM'] = talib.MOM(cl, timeperiod=14)
    df['RSI'] = talib.RSI(cl, timeperiod=14)
    df['WILLR'] = talib.WILLR(hi, lo, cl, timeperiod=14)
    df['BOLUP'], _, df['BOLLOW'] = talib.BBANDS(cl, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['EMA'] = talib.EMA(cl, timeperiod=50)
    df['Trend'] = 0
    df['Trend'] = np.where(df['Close'] > df['EMA'], 1, 0)

    return df


def eight_trigram(df):
    high_high = df['High'] >= df['High'].shift(1)
    high_low = df['High'] <= df['High'].shift(1)
    close_high = df['Close'] >= df['Close'].shift(1)
    close_low = df['Close'] <= df['Close'].shift(1)
    low_high = df['Low'] >= df['Low'].shift(1)
    low_low = df['Low'] <= df['Low'].shift(1)

    BearHorn = high_high & close_low & low_low
    BearHarami = high_low & close_low & low_high
    BearHigh = high_high & close_low & low_high
    BearLow = high_low & close_low & low_low
    BullishHorn = high_high & close_high & low_low
    BullishHarami = high_low & close_high & low_high
    BullishHigh = high_high & close_high & low_high
    BullishLow = high_low & close_high & low_low

    df['EightTri'] = 0
    df['EightTri'] = np.where(BearHorn, 1, df['EightTri'])
    df['EightTri'] = np.where(BearHarami, 2, df['EightTri'])
    df['EightTri'] = np.where(BearHigh, 3, df['EightTri'])
    df['EightTri'] = np.where(BearLow, 4, df['EightTri'])
    df['EightTri'] = np.where(BullishHorn, 5, df['EightTri'])
    df['EightTri'] = np.where(BullishHarami, 6, df['EightTri'])
    df['EightTri'] = np.where(BullishHigh, 7, df['EightTri'])
    df['EightTri'] = np.where(BullishLow, 8, df['EightTri'])

    df.loc[df.index[0], 'EightTri'] = random.randrange(1, 8)

    return df


def order(side, quantity, symbol, order_type, timeInForce, PRICE):
    try:
        print("{} 주문실행".format(side))
        order = client.futures_create_order(symbol=symbol, side=side, quantity=quantity, type=order_type, timeInForce=timeInForce, price=PRICE)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True


def order_market(symbol, side, quantity, order_type):
    try:
        print("{} 주문실행".format(side))
        order = client.futures_create_order(symbol=symbol, side=side, quantity=quantity, type=order_type)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True


def order_stop(symbol, side, stopPrice, quantity, order_type):
    try:
        print("{} 주문실행".format(side))
        order = client.futures_create_order(symbol=symbol, side=side, stopPrice=stopPrice, quantity=quantity, type=order_type)
        print(order)
    except Exception as e:
        print("an exception occured - {}".format(e))
        return False

    return True




def on_open(ws):
    print('opened connection')

def on_close(ws):
    print('closed connection')


def on_message(ws, message):
    global closes
    global long_phase
    global short_phase

    json_message = json.loads(message)

    candle = json_message['k']

    is_candle_closed = candle['x']

    if is_candle_closed:
        print("일봉시작")
        # 변수 초기화, 모델 로드
        trader1 = joblib.load('./trader1.pkl')
        ONE_DAY = '1d'

        all_tickers = client.futures_symbol_ticker()

        print("코인데이터 파일 저장")
        symbols = [x['symbol'] for x in all_tickers]
        symbols = [symbol for symbol in symbols if symbol.endswith('USDT')]

        for symbol in symbols:
            columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df = pd.DataFrame(client.futures_klines(symbol=symbol, interval=ONE_DAY))
            df = df.iloc[:, 0:6]
            df.columns = columns
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.to_csv('./data/' + symbol + ONE_DAY + '.csv')

        print("시장지수 데이터프레임")
        GOLD = pd.DataFrame(fdr.DataReader('ZG', '2021'))
        NDAQ = pd.DataFrame(fdr.DataReader('IXIC', '2021'))
        US10YT = pd.DataFrame(fdr.DataReader('US10YT=X', '2021'))
        VIX = pd.DataFrame(fdr.DataReader('VIX', '2021'))

        print("코인 테스트 데이터 준비")
        cols = {'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
        scores = {"coin": [],
                  "score": [],
                  "today": []};
        path = "./Data/"

        file_list = os.listdir(path)
        file_list_csv = [file for file in file_list if file.endswith("USDT1d.csv")]

        p = re.compile('.+(?=USDT)')

        i = 0
        for symbol in file_list_csv:
            coin = p.search(file_list_csv[i]).group()

            df = pd.read_csv(path + symbol, parse_dates=['time'], index_col='time')
            df = df.rename_axis('Date')
            df = df.iloc[:, 1:6]
            df = df[['close', 'open', 'high', 'low', 'volume']]
            df.rename(columns=cols, inplace=True)
            df['Change'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100
            df = get_patterns(df)
            df = eight_trigram(df)
            df = get_indicators(df)
            df['Trend'] = 0
            df['Trend'] = np.where(df['Close'] > df['EMA'], 1, 0)

            df = df.merge(GOLD, on='Date', how='left', suffixes=('_' + coin, '_GLD'))
            list_df = ['NDAQ', 'US10YT', 'VIX']
            for x in list_df:
                df = df.merge(globals()[x].add_suffix('_' + str(x)),
                              on='Date',
                              how='left')

            df.Volume_GLD.fillna(0, inplace=True)
            df.Change_GLD.fillna(0, inplace=True)
            df.Volume_NDAQ.fillna(0, inplace=True)
            df.Change_NDAQ.fillna(0, inplace=True)
            df.Change_US10YT.fillna(0, inplace=True)
            df.Volume_VIX.fillna(0, inplace=True)
            df.Change_VIX.fillna(0, inplace=True)

            df = df.fillna(0)
            df.dropna(inplace=True)

            df['target'] = df['Close_' + coin].pct_change()
            df['target'] = np.where(df['target'] > 0, 1, -1)
            df['target'] = df['target'].shift(-1)
            df['target'].fillna(0, inplace=True)

            y_var = df['target']
            x_var = df.drop(['target', 'Volume_VIX'], axis=1)

            pred = trader1.predict(x_var)

            scores["coin"].append(file_list_csv[i])
            scores["score"].append(trader1.score(x_var, y_var))
            scores["today"].append(pred[-1])

            i += 1

        p2 = re.compile('.+(USDT)')
        df_rank = pd.DataFrame(scores)
        df_rank['today'] = np.where(df_rank['today'] == 1, 'long', 'short')
        rank = df_rank.sort_values('score', ascending=False)
        coins_temp = list(rank['coin'])
        today_coin = []

        for i in range(len(coins_temp)):
            today_coin.append(p2.search(coins_temp[i]).group())

        long_or_short = list(rank['today'])
        print("try my luck today : ", today_coin[:5], long_or_short[:5])

        print("계좌 및 포지션 불러오기")
        df_position = pd.DataFrame(client.futures_position_information())
        # position_entry = float(df_position.loc[:, 'entryPrice'])
        # leverage = int(df_position.iloc[0][6])
        # positionAmt = float(df_position.loc[:, 'positionAmt'])
        position_info = df_position.sort_values('positionAmt', ascending=False)
        current_position = list(position_info['positionAmt'])
        current_symbol = list(position_info['symbol'])
        long_symbol = current_symbol[0]
        short_symbol = current_symbol[-1]
        long_amount = float(current_position[0])
        short_amount = float(current_position[-1])

        # PNL = float(df_position.loc[:, 'unRealizedProfit'])

        print("포지션이 존재할 경우 당일 청산")
        if long_amount > 0 or short_amount < 0:
            # symbol = df_position.loc[:, 'symbol']
            if long_amount > 0:
                order_status = order_market(long_symbol, SIDE_SELL, long_amount, ORDER_TYPE_MARKET)
            elif short_amount < 0:
                order_status = order_market(short_symbol, SIDE_BUY, short_amount, ORDER_TYPE_MARKET)
            print(order_status)

            # 포지션 청산 후 오늘의 코인 배팅
            if long_or_short[0] == 'long':
                df = pd.DataFrame(client.futures_klines(symbol=today_coin[0], interval='1d'))
                last_price = float(df.iloc[-2][4])
                get_balance = pd.DataFrame(client.futures_account_balance())
                balance = float(get_balance.iloc[6][2])
                bid = balance / last_price * 0.95
                print("entry long")
                print("coin : {}".format(today_coin[0]))
                print("price : {}".format(last_price))
                print("amount : {}".format(bid))
                for i in range(3, -1, -1):
                    try:
                        if i == 0:
                            order_status = order_market(today_coin[0], SIDE_BUY, int(bid), ORDER_TYPE_MARKET)
                        else:
                            order_status = order_market(today_coin[0], SIDE_BUY, round(bid, i), ORDER_TYPE_MARKET)
                    except:
                        pass
                print(order_status)

            elif long_or_short[0] == 'short':
                df = pd.DataFrame(client.futures_klines(symbol=today_coin[0], interval='1d'))
                last_price = float(df.iloc[-2][4])
                get_balance = pd.DataFrame(client.futures_account_balance())
                balance = float(get_balance.iloc[6][2])
                bid = balance / last_price * 0.95
                print("entry short")
                print("coin : {}".format(today_coin[0]))
                print("price : {}".format(last_price))
                print("amount : {}".format(bid))
                for i in range(3, -1, -1):
                    try:
                        if i == 0:
                            order_status = order_market(today_coin[0], SIDE_SELL, int(bid), ORDER_TYPE_MARKET)
                        else:
                            order_status = order_market(today_coin[0], SIDE_SELL, round(bid, i), ORDER_TYPE_MARKET)
                    except:
                        pass
                print(order_status)

        # 포지션이 없는 경우 신규매수
        else:
            if long_or_short[0] == 'long':
                df = pd.DataFrame(client.futures_klines(symbol=today_coin[0], interval='1d'))
                last_price = float(df.iloc[-2][4])
                get_balance = pd.DataFrame(client.futures_account_balance())
                balance = float(get_balance.iloc[6][2])
                bid = balance / last_price * 0.95
                print("entry long")
                print("coin : {}".format(today_coin[0]))
                print("price : {}".format(last_price))
                print("amount : {}".format(bid))
                for i in range(3, -1, -1):
                    try:
                        if i == 0:
                            order_status = order_market(today_coin[0], SIDE_BUY, int(bid), ORDER_TYPE_MARKET)
                        else:
                            order_status = order_market(today_coin[0], SIDE_BUY, round(bid, i), ORDER_TYPE_MARKET)
                    except:
                        pass
                print(order_status)

            elif long_or_short[0] == 'short':
                df = pd.DataFrame(client.futures_klines(symbol=today_coin[0], interval='1d'))
                last_price = float(df.iloc[-2][4])
                get_balance = pd.DataFrame(client.futures_account_balance())
                balance = float(get_balance.iloc[6][2])
                bid = balance / last_price * 0.95
                print("entry short")
                print("coin : {}".format(today_coin[0]))
                print("price : {}".format(last_price))
                print("amount : {}".format(bid))
                for i in range(3, -1, -1):
                    try:
                        if i == 0:
                            order_status = order_market(today_coin[0], SIDE_SELL, int(bid), ORDER_TYPE_MARKET)
                        else:
                            order_status = order_market(today_coin[0], SIDE_SELL, round(bid, i), ORDER_TYPE_MARKET)
                    except:
                        pass
                print(order_status)

ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_close=on_close, on_message=on_message)
ws.run_forever()