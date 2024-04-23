import pdb

import math
import matplotlib.pyplot as pyplot
import matplotlib.colors as mcolors
from . import loss, modules
from amends import _pandas, timer, data_proxy, _torch, _datetime, meth, _dict, functional, _hashlib
from amends.data_structures import consecutive_block_finder, graph
import torch.utils.data
import scipy, numpy

from .loss import criterion, l2
from .data import FixedSeqLenDataset, get_data, unshift
from .modules import Predictor0





def make_sample(
    clean_coin_data,
    clean_market_data,
    target_coin_id,
    n_top_gainers=2**4,
    date_or_ts=None,
    lookback=None,
    marker_id='bitcoin',
    prices_shift=0,
):
    if date_or_ts is None:
        date_or_ts = clean_market_data.index[-1]

    last_ts = _datetime.to_timestamp(date_or_ts, scale=1000) // meth.ms_per_day * meth.ms_per_day
    index = numpy.argmax(clean_market_data.index == last_ts)

    r = consecutive_block_finder.max_pre_or_succ_block(
        clean_market_data.attrs['raw_blocks_by_ticker'][target_coin_id],
        index,
    )
    if r is None:
        return

    block_index, start_index, stop_index = r
    if lookback is not None:
        start_index = max(start_index, (stop_index - lookback))

    # print(f'{block_index=} {start_index=} {stop_index=}')

    return get_data(
        clean_coin_data=clean_coin_data,
        clean_market_data=clean_market_data,
        target_coin_id=target_coin_id,
        market_start_index=start_index,
        market_stop_index=stop_index,
        marker_coin_id=marker_id,
        n_top_gainers=n_top_gainers,
        prices_shift=prices_shift,
    )


def eval(model, x, y=None, l2_reg_lammda=.0001, loss_name='simp_nll'):
    y_ = model(x)

    loss_value = None
    if y is not None:
        loss_value = loss.criterion(lammda=l2_reg_lammda, model=model, loss_name=loss_name)(
            y_,
            y,
        )

    return y_, loss_value


def eval_and_plot(
    model,
    x,
    y=None,
    scale=None,
    l2_reg_lammda=.0001,
    predict_last=100,
    hide_tops=0,
    debug=0,
    **k,
):
    y_, loss = eval(model, x, y, l2_reg_lammda)
    fig, ys_y = plot_it(
        x,
        y=y,
        model=model,
        scale=scale,
        predict_last=predict_last,
        hide_tops=hide_tops,
        return_prediction_labels=1,
        debug=debug,
        **k
    )
    return y_, loss, fig, ys_y


def slice_market_data(market_data, *a):
    return _torch.map_(lambda b: b[:, slice(*a)], market_data)

def slice_x_market_data(x, *a):
    [
        (t_coin_data, t_market_data),
        m_market_data,
        (top_gainers_coin_data, top_gainers_market_data),
        (top_losers_coin_data, top_losers_market_data),
    ] = x

    t_market_data_clip = slice_market_data(t_market_data, *a)
    top_gainers_market_data_clip = slice_market_data(top_gainers_market_data, *a)
    top_losers_market_data_clip = slice_market_data(top_losers_market_data, *a)
    return (
        (t_coin_data, t_market_data_clip),
        m_market_data,
        (top_gainers_coin_data, top_gainers_market_data_clip),
        (top_losers_coin_data, top_losers_market_data_clip),
    )

def plot_it(
    x,
    y_=None,
    y=None,
    y__=None,
    model=None,
    predict_last=100,
    fixed_seq_length=None,
    hide_tops=0,
    show_before_predict=7,
    show_predicted_shifted=0,
    scale='log',
    return_prediction_labels=0,
    plot_type='plot',
    debug=0
):
    global px, py, real_values, predicted_values, marker_values, color_tuple, ys_
    global t_coin_data, t_market_data,m_market_data,top_gainers_coin_data, top_gainers_market_data,top_losers_coin_data, top_losers_market_data
    px = x
    py = y

    if scale=='log':
        normalization_f = lambda a: a
    elif scale=='ratio':
        normalization_f = torch.exp

    [
        (t_coin_data, t_market_data),
        m_market_data,
        (top_gainers_coin_data, top_gainers_market_data),
        (top_losers_coin_data, top_losers_market_data),
    ] = x

    batch_size = t_coin_data.shape[0]
    seq_length = t_market_data.shape[1] + 1
    if debug: print(f'{seq_length=}')
    predict_last = min(predict_last, seq_length)

    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)

    # calculate predicted valeus
    # (prediction_length, batch_size)
    predicted_means = predicted_means_fixed_seq_len = []
    if y_ is not None:
        ys_ = torch.tensor([y_])
        predicted_means = torch.stack([y_[:,[0]]] * 2)
    elif model is not None:
        # predict values using the model
        def _(start, stop):
            sliced_x = slice_x_market_data(x, start, stop)
            return model(
                sliced_x
            )

        timer.start()
        ys_ = torch.full(
            (predict_last, batch_size, 2),
            fill_value=torch.nan,
        )
        if debug: print(f'allocated predicted array in {timer.stop()}')

        timer.start()
        model.eval()
        with torch.no_grad():
            for i in range(predict_last):
                stop = - predict_last + i + 1
                if stop + seq_length <= 1:
                    break
                if stop == 0:
                    stop = None
                start = 0
                ys_[i] = _(start, stop)
        if debug: print(f'generated predictions in {timer.stop()}')
        predicted_means = ys_[...,[0]]

        if fixed_seq_length is not None:
            ys_fixed_seq_len = torch.full(
                (predict_last, batch_size, 2),
                fill_value=torch.nan,
            )
            timer.start()
            model.eval()
            with torch.no_grad():
                for i in range(predict_last):
                    stop = - predict_last + i + 1
                    if stop + seq_length <= 1:
                        break
                    if stop == 0:
                        stop = None

                    start = 0
                    if fixed_seq_length > 0:
                        start = stop - fixed_seq_length

                    ys_fixed_seq_len[i] = _(start, stop)
            if debug: print(f'generated predictions in {timer.stop()}')
            predicted_means_fixed_seq_len = ys_fixed_seq_len[...,[0]]

    else:
        print('model and predicted value passed were none, can\'t predict values')

    if y__ is not None:
        after_predicted_means = y__[:,0]

    def plot_ts(ax, alignment, values, show_max=None, *a, **k):
        values = _torch.smooth(values)
        if show_max is not None:
            values = values[-show_max:]
        top = seq_length + alignment

        x = list(range(top - len(values), top))
        y = normalization_f(values).tolist()

        k['label'] += f' ({values[-1]})'

        return getattr(ax, plot_type)(
            x,
            y,
            *a,
            **k,
        )

    # for i, color_tuple in enumerate(get_color_tuples((batch_size, 7))):
    #     (
    #         real_color,
    #         pred_color,
    #         marker_color,
    #         after_pred_color,
    #         top_gain_color,
    #         top_loss_color,
    #         pred_shft_color,
    #     ) = color_tuple
    for i in range(batch_size):

        predicted_values = predicted_means[:,i]
        num_predicted_values = len(predicted_values)

        if y is not None:
            real_values = torch.concatenate([t_market_data[i,:,0], y[i]])
            plot_ts(
                ax,
                0,
                real_values,
                show_max=num_predicted_values + show_before_predict,
                #color=real_color,
                label=f'actual#{i}'
            )

        if len(predicted_values):
            plot_ts(
                ax, 0,
                predicted_values,
                #color=pred_color,
                label=f'predicted#{i}'
            )

        if len(predicted_means_fixed_seq_len):
            predicted_values_fixed_seq_len = predicted_means_fixed_seq_len[:,i]
            plot_ts(
                ax,
                0,
                predicted_values_fixed_seq_len,
                #color=pred_shft_color,
                label=f'predicted fixed seq len#{i}'
            )

        if show_predicted_shifted:
            plot_ts(
                ax,
                -1,
                predicted_values,
                #color=pred_color,
                label=f'predicted shifted#{i}'
            )

        if y__ is not None:
            after_predicted_values = after_predicted_means[i]
            after_predicted_values = torch.full((2,), after_predicted_values.item())
            plot_ts(
                ax,
                0,
                after_predicted_values,
                #color=after_pred_color,
                label=f'original predicted#{i}'
            )

        if not hide_tops:
            plot_ts(
                ax,
                -1,
                top_gainers_market_data[-1][i, :, 0],
                show_max=num_predicted_values + show_before_predict,
                #color=top_gain_color,
                label=f'top gainer#{i}'
            )

            plot_ts(
                ax,
                -1,
                top_losers_market_data[0][i, :, 0],
                show_max=num_predicted_values + show_before_predict,
                #color=top_loss_color,
                label=f'top loser#{i}'
            )

        marker_values = m_market_data[i,:,0]
        plot_ts(
            ax,
            0,
            marker_values,
            show_max=num_predicted_values + show_before_predict,
            #color=marker_color,
            label=f'reference#{i}'
        )

    ax.legend()

    if return_prediction_labels:
        return fig, torch.cat((ys_, real_values[-ys_.shape[0]:].view(*ys_.shape[:-1], 1).to(ys_.device),), dim=-1)

    return fig

def get_color_tuples(shape, seed=0):
    choices=list(filter(
        lambda a: numpy.linalg.norm(mcolors.to_rgb(a)) < .88,
        mcolors.CSS4_COLORS,
    ))
    return numpy.random.default_rng(seed).choice(choices, math.prod(shape)).reshape(shape)