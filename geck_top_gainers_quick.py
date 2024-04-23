# %load_ext autoreload
# %autoreload 2

from utils.apiwrap import _coingecko
from amends import _dict, timer
import functools
from amends._requests import proxies



output_dir = r'J:\loli\data\stonks\coingecko_raped'
cache_file = posixpath.join(output_dir, 'geck.cache')
timer.start()
geck = _coingecko.geck(
    cache_file=cache_file,
    request_delay=1.25,
    cache_timeout_seconds=None,
    proxy_manager=proxies.ProxyManager(
        known_proxies=proxies.smartproxy_bought()
    )
)
print(timer.stop())
geck = _coingecko.CoinGecko(proxy_manager=p)
print(timer.stop())



r = geck.get_top_movers_data()

f = r.loc[:29,['id','market_data.price_change_percentage_24h','categories']]
f

e = _dict.sort_by_values(_dict.count_occurences(functools.reduce(lambda a,b:a+b,r.loc[:29,'categories'])))
e


