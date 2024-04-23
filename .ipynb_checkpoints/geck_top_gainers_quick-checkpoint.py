# %load_ext autoreload
# %autoreload 2

from utils.apiwrap import _coingecko
from amends import _dict, timer
import functools
from amends._requests import proxies



timer.start()
p = proxies.ProxyManager(known_proxies=proxies.smartproxy_bought())
print(timer.stop())
geck = _coingecko.CoinGecko(proxy_manager=p)
print(timer.stop())



r = geck.get_top_movers_data()

r['pcp'] = r.market_data.apply(lambda a:a['price_change_percentage_24h'])
r.loc[:29,['id','pcp','categories']]
_dict.sort_by_values(_dict.count_occurences(functools.reduce(lambda a,b:a+b,r.loc[:29,'categories'])))


