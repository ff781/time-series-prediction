# %reload_ext autoreload
# %autoreload 2

# +
import datetime
import glob
import posixpath
from amends import _dict, timer, exceptional, text, _requests, _pandas, functional, _os
from amends._requests import proxies
from utils.apiwrap import _coingecko

import time
import functools
import os
import pandas
import numpy
import tqdm
import warnings

warnings.filterwarnings(action='once', category=pandas.errors.PerformanceWarning)

# +
output_dir = r'J:\loli\data\stonks\coingecko_raped'
cache_file = posixpath.join(output_dir, 'geck.cache')
all_coins_data_file = posixpath.join(output_dir, f'coins.feather')
all_coins_market_data_file = posixpath.join(output_dir, f'coins_market_data.feather')
coins_historical_market_data_dir = posixpath.join(output_dir, f'market_data')
coin_historical_market_data_file_f = lambda coin_id: posixpath.join(coins_historical_market_data_dir, f'{coin_id}.feather')

os.makedirs(coins_historical_market_data_dir, exist_ok=True)


# -
timer.start()
geck = _coingecko.geck(
    cache_file=cache_file,
    request_delay=.69,
    request_timeout=.69,
    cache_timeout_seconds=None,
    proxy_manager=proxies.ProxyManager(
        known_proxies=proxies.smartproxy_bought()
    )
)
print(f'{timer.stop()} seconds to start the 𓆈')


def fetch_all_coins_data(batch_size=2**8):
    fresh = 0
    if os.path.isfile(all_coins_data_file):
        try:
            old_coins_data = _pandas.load(all_coins_data_file)
        except Exception as e:
            raise
            exceptional.clean(e)
            fresh = 1
        else:
            all_coin_ids = geck.get_all_coin_ids(fresh=1)
            missing_coin_ids = tuple(filter(lambda a: a not in old_coins_data['id'].values, all_coin_ids))
    else:
        fresh = 1
    if fresh:
        old_coins_data = pandas.DataFrame()
        missing_coin_ids = geck.get_all_coin_ids(fresh=1)
    
    all_coins_data = old_coins_data
    for coin_ids in tqdm.tqdm(list(functional.iteration.batches(missing_coin_ids, batch_size=batch_size))):
        coins_data = geck.get_all_coins_data(coin_ids=coin_ids)
        all_coins_data = pandas.concat([all_coins_data, coins_data], axis=0, ignore_index=True)

        try:
            _pandas.save(_pandas.dumben(all_coins_data), all_coins_data_file)
        except Exception as e:
            print(exceptional.clean(e))
            return all_coins_data
            
    return all_coins_data


def update_coins_historical_market_data(coin_ids=None, obsolescence_days=5, debug=0):
    if coin_ids is None:
        coin_ids = _pandas.load(all_coins_data_file).id
    coin_historical_market_data_files = list(map(coin_historical_market_data_file_f, coin_ids))
    n0 = len(coin_ids)
    data = list(zip(coin_ids, coin_historical_market_data_files))
    
    counts=[0, 0, 0]
    def _(_):
        _,filename = _
        if os.path.isfile(filename):
            if _os.last_mod_sec(filename) < obsolescence_days*24*60*60:
                counts[1] += 1
                return False
            else:
                counts[0] += 1
        return True
    data = list(filter(_, data))
    
    if debug:
        print(f'{counts[0]}/{counts[0]+counts[1]}/{n0}:obsolete/exist/total ids')
        print(f'{len(data)}/{n0} coin_ids will have market data refetched')

    for coin_id, coin_historical_market_data_file in tqdm.tqdm(data):

        try:
            coin_historical_market_data = geck.get_coin_market_data(coin_id, days=365, fresh=1)
            _pandas.save(_pandas.dumben(coin_historical_market_data), coin_historical_market_data_file)
        except Exception as e:
            print(exceptional.clean(e))
            continue

def clean_and_consolidate_market_data():
    global r
    r = []
    for path, deers, files in os.walk(coins_historical_market_data_dir):
        for file in tqdm.tqdm(files):
            full_file = posixpath.join(path, file)
            coin_id, _ = os.path.splitext(file)

            data = _pandas.load(full_file)

            data['timestamp'] -= data['timestamp'] % 86400000
            data = data.set_index('timestamp')
            data = data[~data.index.duplicated(keep='first')]
            data = _pandas.prefix_columns(data, f'{coin_id}.')

            r.append(data)

    # concat all coins market data into one table
    timer.start()
    r = _pandas.concat_along_1_optimized(r)
    print(timer.stop())

    # unindex timestamp for featherization
    r = r.reset_index()
    r = r.rename(columns={'index': 'timestamp'})

    # drop illegal timestamp==0 row
    r = r[r.timestamp!=0]

    # sort by timestamp
    r = r.sort_values(by=['timestamp'], ignore_index=True)
    _pandas.save(_pandas.dumben(r), all_coins_market_data_file)

    return r


a = fetch_all_coins_data(2**8)

update_coins_historical_market_data(obsolescence_days=1, debug=1)

clean_and_consolidate_market_data()



