import functools
import os.path

import traceback, pdb
from . import base
from amends import _pandas, text, _dict, _datetime, functional, timer, _dill, data_proxy, _numpy
from utils.model import stonks
import utils.model.stonks.data
from sklearn.preprocessing import *
import dill
import posixpath
import pandas
import numpy, scipy




def clean_coin_data(
    min_consecutive_block_size=50,
    min_volume_usd=200000,
    max_log_price_swing=((.8, .5), (1, 3)),
    market_data=None,
    debug=0,
):
    global r, _index, price_data
    timer.start()
    r = stonks.data.source['coingecko']['info'].copy()
    print(f'{clean_coin_data.__name__}.0', timer.stop())
    if market_data is None:
        market_data = clean_market_data()
        #stonks.data.source['coingecko']['market'].copy()

    timer.start()
    # relevant columns
    r = _pandas.dumben(r)

    # scrap those with very low volume
    _index = r['market_data.total_volume.usd'] >= min_volume_usd
    before_len = len(r)
    catgirl_in = 'catgirl' in r.id[~_index].values
    r = r[_index]
    if debug:
        if catgirl_in:
            print('catgirl was dropped by:')
        print(f'filter by volume greater as {min_volume_usd} dropped {before_len-len(r)} rows')

    r = _pandas.AdvancedIndexingProxy(r)[[
        '^id$',
        'categories',
        'block_time_in_minutes',
        'hashing_algorithm',
        #'detail_platforms', not really correct, projects sometimes only list a few of all of their available platforms here, rather use categories column
        'links',
        #'sentiment_votes_',
        'tickers',
        #'watchlist_portfolio_users',
        #'market_cap_rank', # has na
        'community_data.twitter_followers', # has na => 0
        'community_data.telegram_channel_user_count', # has na => 0
        'developer_data.forks',
        'developer_data.stars',
        'developer_data.subscribers',
        'developer_data.total_issues',
        'developer_data.closed_issues',
        'developer_data.pull_requests_merged',
        'developer_data.pull_request_contributors',
        'developer_data.code_additions_deletions_4_weeks.additions', # has na => 0
        'developer_data.code_additions_deletions_4_weeks.deletions', # has na => 0
        'developer_data.commit_count_4_weeks',
        'genesis_date',
    ]]
    if debug: print(0, timer.stop())

    # filter all that have outlandish price swings, probably price manipulation
    for quantile_p, upper_bound in max_log_price_swing:
        timer.start()
        price_data = market_data.loc[:, (r.id,'prices')]
        _index = probably_not_too_manipulated = numpy.nanquantile(
            numpy.log(price_data / price_data.shift(1)).abs(),
            q=quantile_p,
            axis=0
        ) <= upper_bound

        before_len = len(r)
        catgirl_in = 'catgirl' in r.id[~_index].values
        r = r[probably_not_too_manipulated]
        if debug:
            if catgirl_in:
                print('catgirl was dropped by:')
            print(
                f'{clean_coin_data.__name__} filter by abs log volatility {quantile_p}-quantile less than {upper_bound} dropped {before_len-len(r)} rows in',
                timer.stop()
            )

    # drop all rows that have no tickers
    r = r[r['tickers'].apply(lambda a: getattr(a, 'size', 0)!=0)]
    if debug: print(1, timer.stop())

    # remove all columns only composed of na
    r = r[r.columns[~r.isna().all(axis=0)]]
    if debug: print(2, timer.stop())

    # links columns, filter: non string iterable
    _index = _pandas.aip(r).getaindex(['links'])
    r = r[r.columns[
        r[_index]
        .applymap(functional.it.is_non_string_iterable)
        .all(axis=0)
        .reindex(r.columns, fill_value=True)
    ]]
    if debug: print(4, timer.stop())

    # links columns: reduce to count
    _index = _pandas.aip(r).getaindex(['links'])
    r[_index] = r[_index].applymap(lambda a: numpy.count_nonzero(a))
    if debug: print(5, timer.stop())

    # convert market data to numeric, datetime
    _index = _pandas.aip(r).getaindex(['market_data', 'developer_data', 'genesis_date', 'links', 'sentiment_votes'])
    r[_index] = _pandas.coerce_num_dt_type(r[_index].copy())
    if debug: print(3, timer.stop())

    # geck_genesis_date based on first market datapoint from coingecko
    # r['geck_genesis_date'] = r.id.map(market_data.attrs['lower_bound_by_ticker'])
    # if debug: print(6, timer.stop())

    # detail_platforms onehot, '' = has its own platform
    # r = _pandas.expand_to_binary_vector(r, 'detail_platforms', debug=debug)
    # if debug: print(7, timer.stop())

    # sentiment votes, na = 0
    # _index = _pandas.aip(r).getindex(['sentiment_votes_'])
    # r[_index] = r[_index].fillna(0)
    # if debug: print(9,timer.stop())

    return r


def transform_coin_data(
    clean_coin_data,
    include_tickers=0, # extremely bloated
    debug=0,
):
    global r, _index
    r = clean_coin_data.copy()

    timer.start()
    r['hashing_algorithm'] = r['hashing_algorithm'].fillna('none')
    r = _pandas.expand_to_typ_vector(r, 'hashing_algorithm', debug=debug)
    if debug: print(69,timer.stop())

    timer.start()
    r = _pandas.expand_to_typ_vector(r, 'categories', debug=debug)
    if debug: print(8,timer.stop())


    if include_tickers:
        timer.start()
        # tickers: filter by is_anomaly=false,is_stale=false,trust_score in [green,yellow]
        # tickers become tuple of (market.identifier, target_coin_id, is_anomaly, is_stale, trust_score)
        def filter_tickers(list_of_tickers):
            def ticker_predicate(ticker):
                return ticker["trust_score"] in ["green", "yellow"]

            def ticker_cleaner(ticker):
                #ticker = _dict.partial(ticker, indices=["market", "target_coin_id", "is_anomaly", "is_stale", "trust_score"])
                ticker = _dict.partial(ticker, indices=["market", "target_coin_id"])
                ticker["market_id"] = ticker.pop("market")["identifier"]
                return ticker

            def ticker_flattener(ticker):
                #ticker = tuple(_dict.select(ticker, indices=["market_id", "target_coin_id", "is_anomaly", "is_stale", "trust_score"]))
                ticker = tuple(_dict.select(ticker, indices=["market_id", "target_coin_id"]))
                return ticker

            r = list(map(
                lambda a: ticker_flattener(a),
                map(
                    ticker_cleaner,
                    filter(
                        ticker_predicate,
                        list_of_tickers
                    )
                )
            ))
            r = _dict.count_occurences(r)
            r = _dict.stringify_keys(r)
            return r
        r['tickers'] = r['tickers'].apply(filter_tickers)
        if debug: print(f'{transform_coin_data.__name__}.a',timer.stop())

        # tickers onehot
        r = _pandas.expand(r, 'tickers', fill_value=0)
    else:
        r = r.drop(columns=['tickers'])


    # fillna
    # _index = _pandas.aip(r).getindex(["market_data.*date",])
    # r[_index] = r[_index].fillna(r[_index].max())
    _index = _pandas.aip(r).getaindex(["tickers", "community_data", "developer_data", "market_data"])
    r[_index] = r[_index].fillna(0)
    # r['market_cap_rank'] = r['market_cap_rank'].fillna(69696)
    r['genesis_date'] = r['genesis_date'].fillna(r['genesis_date'].max())

    # convert to num
    r = _pandas.coerce_num_type(r, errors='ignore')

    global a, b
    # normalize all except id, categorical columns
    _index = _pandas.aip(r).getaindex(['^(?!hashing_algorithm)(?!categories)(?!tickers)(?!id)'])
    r[_index] = MinMaxScaler().fit_transform(r[_index])

    r = r.reset_index(drop=True)

    return r


def clean_market_data():
    r = stonks.data.source['coingecko']('market', 'clean').copy()

    r = r.reset_index()
    r = r[r.bitcoin.prices.notna()]

    return r


def transform_market_data(clean_market_data, debug=0):
    global r
    r = clean_market_data.copy()

    r = r.fillna(numpy.nan)
    r[r==0] = numpy.nan

    # r = add_helpers_to_market_data(r, debug=debug)
    # r = add_super_helpers_to_market_data(r, debug=debug)

    if 'timestamp' in r:
        r = r.set_index('timestamp')

    # normalize market_caps and total_volumes by dividing by highest market_cap/total_volume
    for col_name in ['market_caps', 'total_volumes']:
        _index = r.columns.get_level_values(level=1).isin((col_name,))
        r.attrs.setdefault('max', {})[col_name] = numpy.nanmax(r.loc[:,_index])
        r.loc[:,_index] /= r.attrs['max'][col_name]

    # all 0 == nan
    # r = r.fillna(0)
    r = r.reset_index()

    return r

def post_coin_clean_market_data(market_data, coin_data):
    global r
    r = market_data

    r = r.set_index('timestamp')

    r = r[coin_data.id]

    r = r.reset_index()

    r = add_helpers_to_market_data(r)
    r = add_super_helpers_to_market_data(r)

    return r

def get_coin_and_market_data(clear_cache=0, debug=0):
    cps = [
        base.p("train_coin_data.data"),
        base.p("train_market_data.data"),
    ]
    if clear_cache:
        for p in cps:
            if os.path.isfile(p):
                os.remove(p)

    datas = list(map(lambda a: _dill.load(a) if os.path.isfile(a) else None, cps))


    if (a:=datas[1]) is None:
        datas[1] = clean_market_data()

    if datas[0] is None:
        datas[0] = transform_coin_data(clean_coin_data(market_data=datas[1], debug=debug),)
        _dill.dump(datas[0], cps[0])

    if a is None:
        datas[1] = transform_market_data(datas[1], debug=debug)

    if a is None:
        datas[1] = post_coin_clean_market_data(datas[1], datas[0])
        _dill.dump(datas[1], cps[1])

    datas[1] = datas[1].set_index('timestamp')

    return datas


def add_helpers_to_market_data(market_data, debug=0):
    global r
    r = market_data

    r = r.set_index('timestamp')

    timer.start()
    r.attrs['upper_bound'] = (~r[::-1].isna()).idxmax(axis=0)
    r.attrs['lower_bound'] = (~r.isna()).idxmax(axis=0)
    if debug: print(f'{add_helpers_to_market_data.__name__}.2', timer.stop())

    timer.start()
    r.attrs['upper_bound_by_ticker'] = r.attrs['upper_bound'].groupby(level=0).min()
    r.attrs['lower_bound_by_ticker'] = r.attrs['lower_bound'].groupby(level=0).max()
    if debug: print(f'{add_helpers_to_market_data.__name__}.3', timer.stop())

    timer.start()
    r.attrs['assumed'] = r.attrs['upper_bound'] - r.attrs['lower_bound']
    r.attrs['actual'] = (~r.isna()).sum(axis=0)
    r.attrs['actual_by_ticker'] = r.attrs['actual'].groupby(level=0).min()
    if debug: print(f'{add_helpers_to_market_data.__name__}.4', timer.stop())

    r.attrs['tickers'] = r.columns.get_level_values(level=0)[::3]

    r = r.reset_index()

    return r


def add_super_helpers_to_market_data(market_data, debug=0):
    global r
    r = market_data

    r = r.set_index('timestamp')

    def _(sub):
        s = sub
        a = (
            (~sub.iloc[:,0].isna()) &
            (~sub.iloc[:,2].isna())
        )
        return a
    timer.start()
    r.attrs[f'bool_by_ticker'] = r.groupby(level=0, axis=1).apply(_)
    if debug: print(f'{add_super_helpers_to_market_data.__name__}.b', timer.stop())


    timer.start()
    r.attrs[f'bool_sum'] = r.attrs[f'bool_by_ticker'].sum(axis=1)
    if debug: print(f'{add_super_helpers_to_market_data.__name__}.a', timer.stop())

    for _ph in ['raw', 'grouped']:

        timer.start()
        r.attrs[f'{_ph}_blocks_by_ticker'] = r.attrs['bool_by_ticker'].apply(
            lambda a: _numpy.index_binary_category_consecutive_blocks(
                a.values,
                format=_ph,
            ),
        )
        if debug: print(f'{add_super_helpers_to_market_data.__name__}.{_ph}', timer.stop())


    timer.start()
    r.attrs['block_max_by_ticker'] = r.attrs['grouped_blocks_by_ticker'].apply(lambda a: max(a, default=0))
    if debug: print(f'{add_super_helpers_to_market_data.__name__}.c', timer.stop())

    r = r.reset_index()

    return r