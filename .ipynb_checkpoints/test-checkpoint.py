# %reload_ext autoreload
# %autoreload 2
from amends import _datetime, _dict, _pandas, text, functional, timer, _torch, _numpy
from utils.apiwrap import _coingecko
import ai
from utils.model import stonks
import pandas, torch, torchviz
import numpy
import functools, itertools
import warnings

ai.train.main(
    clear_cache=0,

    model_path='model0',

    num_epochs=2**6,
    #num_batches=2**15,
    batch_size=2**6,
    learning_rate=8e-7,
    gradient_clipping=1,
    l2_reg_lammda=None,#.0001,

    dataset_path='dataset0',
    dataset_use_cache=1,
    seq_length=60,
    n_top_gainers=2**4,
    num_samples=2**20,

    sanity_check=1,
    debug_period=2**9+1,
    debug=1,
)



a = _torch.stats(ai.train._['model']().state_dict())

b=_torch.module.stats(ai.train._['model'](), level=0)

c=_torch.module.stats(ai.train._['model']().top_gainers_coin_data_m.module, level=0)
c

d=_torch.module.stats(ai.train._['model']().top_losers_coin_data_m.module, level=0)
d





ai.train.example(
    'bitcoin',
    predict_last=69,
    show_predicted_shifted=1,
    lookback=500,
    scale='log',
    viz_architecture=1,
    hide_tops=1,
    prices_shift=0
)


viz = ai.train.viz()
viz

c,m = ai.train._['data'](clear_cache=0)



c0, c15 = torch.load(ai.base.p('checkpoint_0')), torch.load(ai.base.p('checkpoint_15'))
a = _torch.diff(c0['model_state_dict'], c15['model_state_dict'])



m_0 = ai.train.model
m_0_0 = m_0.target_coin_meta_module
m_1 = ai.train.make_model(4344, 'cuda')
m_1_0 = m_1.target_coin_meta_module
batch=ai.train.batch_
x,y=batch

_torch.flat_reduce(lambda a,b:any([a,b]),x,key_f=_torch.is_any_nan)

m_0(x), y

m, m2

m2(x[0])



list(ai.train.model.state_dict().values())[1]

b['nem'][703]

timer.start()
coin_data = stonks.data.source['coingecko']['info']
market_data = stonks.data.source['coingecko']['market']
print(timer.stop())

timer.start()
c0 = ai.data.clean_coin_data()
c1 = ai.data.transform_coin_data(c0)
print(timer.stop())

timer.start()
m0 = ai.data.clean_market_data()
m1 = ai.data.transform_market_data(m0)
print(timer.stop())

cc = ai.data.get_coin_data(1, 0)
mm = ai.data.get_market_data(1, 0)




