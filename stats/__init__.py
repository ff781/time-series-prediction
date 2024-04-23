from amends import text, _dict, _pandas, functional, _torch, _requests, timer
from utils.model import stonks
import utils.model.stonks.meth
from utils.apiwrap import tradeogre, _coingecko
import numpy, pandas, torch, pyperclip
import pandas as pd


c1,m1 = stonks.data.source['coingecko']['info_and_market_intersection']


def volatility_and_positioning_analysis(market_data, n, window):
    coins = ['bitcoin', 'ethereum', 'catgirl', 'spx6900']
    coins = pandas.IndexSlice[:]

    total = n + window - 1

    coinage = market_data.iloc[-total:,].loc[:,(coins,'prices')]

    timer.start()
    vol = stonks.meth.rolling(coinage, stonks.meth.volatility, window=window)
    print('vol', timer.stop())

    timer.start()
    pos = stonks.meth.rolling(coinage, stonks.meth.positioning, window=window)
    print('pos', timer.stop())

    timer.start()
    weighted = stonks.meth.rolling(
        coinage,
        stonks.meth.weighted,
        window=window,
        fs=[stonks.meth.volatility, stonks.meth.positioning],
        weights=[1, -1],
    )
    print('weighted', timer.stop())

    timer.start()
    ratio = stonks.meth.rolling(
        coinage,
        stonks.meth.ratio,
        window=window,
        f1=stonks.meth.volatility,
        f2=stonks.meth.positioning,
    )
    print('ratio', timer.stop())

    return vol, pos, weighted, ratio, coinage