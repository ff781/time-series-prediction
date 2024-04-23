import os

from amends import _torch, timer

import pdb

from amends import functional, _pandas, _torch
import torch, numpy, pandas




class CoinMetaEmbedModule(torch.nn.Module):
    def __init__(
        self,
        columns,
        other_layers,
        hashing_algorithm_emb_dim,
        categories_layers,
        tickers_layers,
        activation_m,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.columns = columns
        self.hash_algorithm_indices = _pandas.aip(columns).getindex(['hashing_algorithm'])
        self.categories_indices = _pandas.aip(columns).getindex(['categories'])
        self.tickers_indices = _pandas.aip(columns).getindex(['tickers'])
        self.n_ticker_indices = self.tickers_indices.sum()
        self.other_indices = ~(self.hash_algorithm_indices | self.categories_indices | self.tickers_indices)


        _ = [
            categories_layers, other_layers,
        ]
        if self.n_ticker_indices > 0:
            _.append(tickers_layers)
        self._output_shape = (sum(map(lambda a: a[-1], _)) + hashing_algorithm_emb_dim),

        num_embeddings = self.hash_algorithm_indices.sum()
        self.hashing_algorithm_emb_dim = hashing_algorithm_emb_dim
        self.hash_algorithm_m = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=hashing_algorithm_emb_dim)
        self.categories_m = _torch.module.DeepLinear([self.categories_indices.sum()] + categories_layers, activation_m=activation_m)
        if self.n_ticker_indices > 0:
            self.tickers_m = _torch.module.DeepLinear([self.n_ticker_indices] + tickers_layers, activation_m=activation_m)

        self.other_m = _torch.module.DeepLinear([self.other_indices.sum()] + other_layers, activation_m=activation_m)

    def output_shape(self):
        return self._output_shape

    def forward(self, raw_data):
        global hash_algorithm_data, categories_data, tickers_data, other_data
        global ss
        ss = self

        hash_algorithm_data = raw_data[:, self.hash_algorithm_indices]
        categories_data = raw_data[:, self.categories_indices]
        if self.n_ticker_indices > 0:
            tickers_data = raw_data[:, self.tickers_indices]
        other_data = raw_data[:, self.other_indices]

        _ = [
            self.hash_algorithm_m(torch.argmax(hash_algorithm_data, dim=-1)),
            self.categories_m(categories_data),
        ]
        if self.n_ticker_indices > 0:
            _.append(self.tickers_m(tickers_data))
        _.append(self.other_m(other_data))

        _ = torch.cat(_, dim=-1)
        return _

class CoinMetaModule(torch.nn.Module):
    def __init__(
        self,
        columns,
        other_layers,
        hashing_algorithm_emb_dim,
        categories_layers,
        tickers_layers,
        combination_layers,
        activation_m,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.columns = columns

        self.embed_m = CoinMetaEmbedModule(
            columns=columns,
            other_layers=other_layers,
            hashing_algorithm_emb_dim=hashing_algorithm_emb_dim,
            categories_layers=categories_layers,
            tickers_layers=tickers_layers,
            activation_m=activation_m,
        )

        self.combination_m = _torch.module.DeepLinear(
            [numpy.prod(self.embed_m.output_shape())] + combination_layers,
            activation_m=activation_m,
        )

    def forward(self, raw_data):
        _ = self.embed_m(raw_data)

        return self.combination_m(_)

class CorrelationModule(torch.nn.Module):
    def __init__(
        self,
        t_meta_m,
        t_market_m,
        o_meta_m,
        o_market_m,
        combination_layers,
        activation_m,
    ):
        super().__init__()

        self.t_meta_m = t_meta_m
        self.t_market_m = t_market_m
        self.o_meta_m = o_meta_m
        self.o_market_m = o_market_m

        self.combination_m = _torch.module.DeepLinear(
            combination_layers,
            # [sum(
            #     list(
            #         map(
            #             lambda a: numpy.prod(a.output_shape()),
            #             [self.t_meta_m, self.t_market_m, self.o_meta_m, self.o_market_m]
            #         )
            #     )
            # )] +
            activation_m=activation_m,
        )

    def forward(self, _):
        (t_coin_data, t_market_data), (o_coin_data, o_market_data), = _

        corr_coef = _torch.corr_coef(t_market_data, o_market_data, dim=-2)
        corr_coef = torch.nan_to_num(corr_coef, 0)

        a = self.t_meta_m(t_coin_data)
        b = self.t_market_m(t_market_data)
        c = self.o_meta_m(o_coin_data)
        d = self.o_market_m(o_market_data)

        cc = torch.cat([corr_coef, a, b, c, d], dim=-1)

        r = self.combination_m(cc)

        return r


class CoinMarketModule(torch.nn.Module):
    def __init__(
        self,
        layers=None,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        if layers is None:
            layers = [torch.nn.LSTM(3, 10, 7, bidirectional=False, dropout=0)]

        self.main = torch.nn.Sequential(
            *layers,
        )

    def forward(self, data):
        return self.main(data)[0][:,-1]


def default_coin_module(columns, activation_m):
    return _torch.module.CatModule(
        module=[
            CoinMetaModule(
                columns=columns,
                other_layers=[
                    16,
                ],
                hashing_algorithm_emb_dim=32,
                categories_layers=[
                    128,
                    64,
                ],
                tickers_layers=[
                    512,256,64
                ],
                combination_layers=[
                    128,
                ],
                activation_m=activation_m,
            ),
            CoinMarketModule(
                [
                    torch.nn.LSTM(
                        input_size=3,
                        hidden_size=32,
                        bidirectional=False,
                        num_layers=2,
                        dropout=0,
                        batch_first=True
                    ),
                ],
            )
        ]
    )

def default_coin_meta_embed_module(columns, activation_m):
    return CoinMetaEmbedModule(
        columns=columns,
        other_layers=[
            16,
        ],
        hashing_algorithm_emb_dim=32,
        categories_layers=[
            128,
            64,
        ],
        tickers_layers=[
            512,256,64
        ],
        activation_m=activation_m,
    )
def default_correlation_module(columns, activation_m):
    return CorrelationModule(
        t_meta_m=default_coin_meta_embed_module(columns, activation_m),
        t_market_m=CoinMarketModule(
            [
                torch.nn.LSTM(
                    input_size=3,
                    hidden_size=32,
                    bidirectional=False,
                    num_layers=2,
                    dropout=0,
                    batch_first=True
                ),
            ],
        ),
        o_meta_m=default_coin_meta_embed_module(columns, activation_m),
        o_market_m=CoinMarketModule(
            [
                torch.nn.LSTM(
                    input_size=3,
                    hidden_size=32,
                    bidirectional=False,
                    num_layers=2,
                    dropout=0,
                    batch_first=True
                ),
            ],
        ),
        combination_layers=[
            # 3 + 2 * (64 + 64 + 32 + 16 + 32),
            3 + 2 * (64 + 32 + 16 + 32),
            256,
            128,
        ],
        activation_m=activation_m,
    )


class Predictor0(_torch.module.AbstractModule):

    """
    expected input, output shape
    (
        # input
        [
            (t_coin_data, t_market_data),
            m_market_data,
            (top_gainers_coin_data, top_gainers_market_data),
            (top_losers_coin_data, top_losers_market_data),

        ],
        # output
        y
    )
    """
    def __init__(
        self,
        columns,
        n_top_gainers,
        model_path=None,
        activation_m=torch.nn.ReLU(),
    ):
        super().__init__()

        self.columns = columns
        self.n_top_gainers = n_top_gainers
        self.model_path = model_path

        self.marker_module = CoinMarketModule(
            [
                torch.nn.LSTM(
                    input_size=3,
                    hidden_size=16,
                    bidirectional=False,
                    num_layers=1,
                    dropout=0,
                    batch_first=True
                ),
            ],
        )

        self.t_coin_m = default_coin_module(columns, activation_m=activation_m)

        # self.top_gainers_coin_data_m = _torch.module.StackModule(
        #     module=[
        #         default_coin_module(columns) for i in range(n_top_gainers)
        #     ]
        # )
        # self.top_losers_coin_data_m = _torch.module.StackModule(
        #     module=[
        #         default_coin_module(columns) for i in range(n_top_gainers)
        #     ]
        # )
        self.top_gainers_m = default_correlation_module(columns, activation_m=activation_m)
        self.top_losers_m = default_correlation_module(columns, activation_m=activation_m)

        # self.top_gainers_reduction_m = _torch.module.Unstack()
        # self.top_losers_reduction_m = _torch.module.Unstack()

        self.final_m = torch.nn.Sequential(
            _torch.module.DeepLinear(
                [
                    #16 + (2 * n_top_gainers + 1) * 64,
                    16 + (32 + 128) + 2 * 128,
                    .5,
                    .5,
                    .5,
                    2,
                ],
                activation_m=activation_m,
            ),
        )


    def forward(self, x):
        global xx, ss
        ss = self
        xx = (
            (t_coin_data, t_market_data),
            m_market_data,
            (top_gainers_coin_data, top_gainers_market_data),
            (top_losers_coin_data, top_losers_market_data),
        ) = x

        t_data = (t_coin_data, t_market_data)
        t_out = self.t_coin_m(t_data)
        marker = self.marker_module(m_market_data)

        g_out = tuple(map(lambda o_data: self.top_gainers_m((t_data, o_data)), zip(*(top_gainers_coin_data, top_gainers_market_data))))
        l_out = tuple(map(lambda o_data: self.top_losers_m((t_data, o_data)), zip(*(top_losers_coin_data, top_losers_market_data))))
        g_avg = torch.stack(g_out, dim=-1).mean(dim=-1)
        l_avg = torch.stack(l_out, dim=-1).mean(dim=-1)

        # c = self.top_gainers_reduction_m(self.top_gainers_coin_data_m(
        #     tuple(zip(*(top_gainers_coin_data, top_gainers_market_data)))
        # ))
        # d = self.top_losers_reduction_m(self.top_losers_coin_data_m(
        #     tuple(zip(*(top_losers_coin_data, top_losers_market_data)))
        # ))

        y_ = self.final_m(torch.cat([marker, t_out, g_avg, l_avg], dim=-1))

        return y_

