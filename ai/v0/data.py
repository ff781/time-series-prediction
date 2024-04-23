import itertools
import pdb
import warnings

import numpy
import torch
from amends import _torch, _hashlib
from amends.data_structures import consecutive_block_finder
from sklearn.preprocessing import MinMaxScaler



class FixedSeqLenDataset(_torch.CacheDataset, _torch.SliceableDataset):
    def __init__(
        self,
        clean_coin_data,
        clean_market_data,
        num_samples=2**15,
        marker_coin_id='bitcoin',
        seq_length=30,
        n_top_gainers=2**4,
        salt=0,
        noise_std=.25,
        debug=0,
    ):
        super().__init__(
            num_samples=num_samples,
            debug=debug
        )

        clean_market_data = clean_market_data
        clean_market_data = clean_market_data.loc[:,clean_coin_data.id]

        self.clean_coin_data = clean_coin_data
        self.clean_market_data = clean_market_data
        self.marker_coin_id = marker_coin_id

        self.seq_length = seq_length
        self.salt = salt
        self.noise_std = noise_std
        self.n_top_gainers = n_top_gainers

    def __len__(self):
        return self.num_samples

    def _fetch_sample(self, i):
        seq_length = self.seq_length

        seed1 = _hashlib.phash((self.salt,i))
        random = numpy.random.default_rng(seed1)

        potential_target_coin_ids = self.clean_market_data.attrs['block_max_by_ticker'].index[
            self.clean_market_data.attrs['block_max_by_ticker'] >= seq_length
        ]

        for try_index in itertools.count():
            target_coin_id = random.choice(potential_target_coin_ids)

            block_indices_of_seq_length = consecutive_block_finder.min_n_long_blocks(
                self.clean_market_data.attrs
                ['grouped_blocks_by_ticker']
                [target_coin_id],
                seq_length
            )

            target_market_start_index = random.choice(
                block_indices_of_seq_length
            )
            target_market_stop_index = target_market_start_index + seq_length

            #prices_shift = random.normal(0, self.noise_std)
            prices_shift = 0

            try:
                return tuple(get_data(
                    clean_coin_data=self.clean_coin_data,
                    clean_market_data=self.clean_market_data,
                    target_coin_id=target_coin_id,
                    market_start_index=target_market_start_index,
                    market_stop_index=target_market_stop_index,
                    marker_coin_id=self.marker_coin_id,
                    n_top_gainers=self.n_top_gainers,
                    prices_shift=prices_shift,
                ))
            except OverflowError:
                global max_fail_index
                globals().setdefault('max_fail_index', -1)
                if try_index > max_fail_index:
                    print(f'getting sample fail#{try_index}, retrying')
                    max_fail_index = try_index
                continue


def get_data(
    clean_coin_data,
    clean_market_data,
    target_coin_id,
    market_start_index,
    market_stop_index,
    marker_coin_id,
    n_top_gainers=2**4,
    n_top_market_caps=None,
    prices_shift=0,
    nan_check=1,
):
    global t_coin_row_, t_coin_data, t_coin_id, t_market_data_all, t_coin_data, t_market_data, m_market_data, y, t_market_data_y
    global top_gainers_market_data_indices, top_gainers_coin_ids, tickers, all_market_data
    global tmstarti, tmstopi, ccd, cmd, tci, gains, gains_argsort
    tmstarti = market_start_index
    tmstopi = market_stop_index
    ccd = clean_coin_data
    cmd = clean_market_data
    tci = target_coin_id

    def data(coin_id, offset):
        return finalized_coin_market_data(
            clean_coin_data=clean_coin_data,
            clean_market_data=clean_market_data,
            coin_id=coin_id,
            start=market_start_index,
            stop=market_stop_index + offset,
            offset=-2,
            prices_shift=prices_shift,
        )

    all_market_data = (
        clean_market_data
        .iloc[market_start_index: market_stop_index - 1, :]
    )
    all_market_data_ = torch.from_numpy(all_market_data.values)
    # gains of all coins
    gains = all_market_data_[-1, ::3] / all_market_data_[0, ::3]
    # sort from lowest to highest gains
    gains_argsort = numpy.argsort(gains)
    # only non nans count (otherwise max gains are nan)
    gains_argsort = gains_argsort[gains[gains_argsort]==gains[gains_argsort]]
    if len(gains_argsort) < 2 * n_top_gainers:
        raise OverflowError('not enough different non-nan tickers in time step')

    tickers = clean_market_data.attrs['tickers']

    top_gainers_market_data_indices = gains_argsort[-n_top_gainers:]
    top_gainers_coin_ids = tickers[top_gainers_market_data_indices]
    top_losers_market_data_indices = gains_argsort[:n_top_gainers]
    top_losers_coin_ids = tickers[top_losers_market_data_indices]

    top_gainers_coin_data, top_gainers_market_data = zip(*map(
        lambda a: data(a, -1),
        top_gainers_coin_ids,
    ))
    top_losers_coin_data, top_losers_market_data = zip(*map(
        lambda a: data(a, -1),
        top_losers_coin_ids,
    ))

    if n_top_market_caps is not None:
        market_caps = all_market_data_[-1, 2::3]
        market_caps_argsort = numpy.argsort(market_caps)
        top_market_cap_market_data_indices = market_caps_argsort[-n_top_market_caps:]
        top_market_cap_coin_ids = tickers[top_market_cap_market_data_indices]
        top_market_cap_coin_data, top_market_cap_market_data = zip(*map(
            lambda a: data(a, -1),
            top_market_cap_coin_ids,
        ))


    # target coin's data
    t_coin_data, t_market_data_y = data(
        target_coin_id,
        0,
    )
    t_market_data = t_market_data_y[:-1, :]
    y = t_market_data_y[-1, [0]]

    # marker has future prices to mark the way
    _, m_market_data = data(
        marker_coin_id,
        0,
    )

    r = (
        [
            (t_coin_data, t_market_data),
            m_market_data,
            (top_gainers_coin_data, top_gainers_market_data),
            (top_losers_coin_data, top_losers_market_data),
        ],
        (y, torch.tensor(prices_shift),)
    )
    if n_top_market_caps is not None:
        r[0].append((top_market_cap_coin_data, top_market_cap_market_data),)

    r = _torch.to(
        r,
        dtype=torch.float32,
    )

    if nan_check:
        has_nan = _torch.is_flat_any_nan(r)
        has_inf = _torch.is_flat_any_inf(r)
        if has_nan or has_inf:
            _ = '/'.join(map(lambda a: a[0], filter(lambda a: a[1], zip(['nan','inf'], [has_nan, has_inf]))))
            raise Exception(f'{_} values detected, opinion rejected\n{has_nan=}\n{has_inf=}')

    return r

def finalized_coin_market_data(
    clean_coin_data,
    clean_market_data,
    coin_id,
    start,
    stop,
    offset,
    prices_shift,
):
    global ccd, coin_row, cid, market_data, market_data_all, sa, sb
    ccd = clean_coin_data
    cid = coin_id
    sa,sb = start,stop

    coin_row = clean_coin_data.loc[clean_coin_data.id==coin_id, :].iloc[0,:]
    coin_data = torch.from_numpy(coin_row[coin_row.index != 'id'].values.astype(numpy.float32).copy())

    market_data_all = clean_market_data[coin_id]


    # normalize
    market_data = market_data_all.iloc[start: stop, :].values.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        market_data = torch.from_numpy(MinMaxScaler().fit_transform(market_data))
    # market_data = torch.from_numpy(market_data_all.iloc[start: stop, :].values.copy())
    # market_data = log_normal_columns_separately(market_data, offset)
    # market_data = _torch.inter_nan(market_data)
    market_data = torch.nan_to_num(market_data, 0)
    market_data[:,0] = market_data[:,0] + prices_shift

    return coin_data, market_data

def unshift(data, shift):
    data = data.clone()
    data[:,0] = data[:,0] - shift
    return data

def log_normal_columns_separately(array, offset):
    global arr
    arr = array
    # only prices should be log normed
    array[:,0] = _torch.log_wref(array[:, 0], offset)

    # rest already normalized over entire dataset, so no need
    # for i in range(1, array.shape[1]):
    #     array[:,i] = _torch.log_wref(array[:, i], -1)
    return array