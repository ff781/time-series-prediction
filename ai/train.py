import itertools
import pdb

import torchviz
from IPython.display import display
import os.path
from . import v0
from . import data, base
from amends import _torch, timer, data_proxy, _dict, functional, _dill, exceptional, meth as amends_meth, _pandas
from amends import _torchview, _torchtest
from matplotlib import pyplot
import tqdm
import numpy, torch.utils.data
import ignite, random
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer



def make_sample(
    target_coin_id,
    date_or_ts=None,
    lookback=None,
    marker_id='bitcoin',
    n_top_gainers=2**4,
    prices_shift=0,
):
    return _torch.batch1(
        v0.make_sample(
            clean_coin_data=_['coin_data'](),
            clean_market_data=_['market_data'](),
            target_coin_id=target_coin_id,
            date_or_ts=date_or_ts,
            n_top_gainers=n_top_gainers,
            lookback=lookback,
            marker_id=marker_id,
            prices_shift=prices_shift,
        ),
    )

def viz():
    input_size = [(torch.Size([1, 2071]), torch.Size([1, 68, 3])), torch.Size([1, 69, 3]), ((torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071])), (torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]))), ((torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071]), torch.Size([1, 2071])), (torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3]), torch.Size([1, 68, 3])))]
    model = _['model']()
    x = make_sample(
        target_coin_id='bitcoin',
        lookback=69,
        n_top_gainers=model.n_top_gainers,
    )[0]
    pdb.set_trace()
    return _torchview.draw_graph(
        model,
        #input_data=x,
        input_size=input_size,
        device='meta',
    )

def example(
    target_coin_id,
    date_or_ts=None,
    lookback=None,
    marker_id='bitcoin',
    l2_reg_lammda=.0001,
    predict_last=100,
    scale='log',
    hide_tops=1,
    viz_architecture=0,
    prices_shift=0,
    debug=0,
    **k,
):
    global sample, vizdot
    model = _['model']().to(inference_device())
    sample = _torch.to(
        make_sample(
            target_coin_id=target_coin_id,
            date_or_ts=date_or_ts,
            lookback=lookback,
            marker_id=marker_id,
            n_top_gainers=model.n_top_gainers,
            prices_shift=prices_shift,
        ),
        inference_device()
    )
    x,y = sample
    y_, loss, fig, ys_y = v0.eval_and_plot(
        model,
        x,
        y,
        scale=scale,
        predict_last=predict_last,
        hide_tops=hide_tops,
        l2_reg_lammda=l2_reg_lammda,
        debug=debug,
        **k,
    )

    display(fig)
    pyplot.close(fig)
    print(f'prediction: {y_}')
    print(f'label: {y}')
    print(f'prediction/label all: {ys_y}')


def inference_device():
    return _torch.device()
def train_device():
    return _torch.device()

def train(
    model,
    dataset,
    validation_dataset,
    max_epochs=10,
    learning_rate=0.01,
    gradient_clipping=1,
    l2_reg_lammda=.0001,
    debug_period=2 ** 7,
    checkpoint_freq=.3,
    loss_name='simp_nll',
    sanity_check=1,
    batch_size=None,
    start_batch=None,
):
    global x, y, y_, batch_index, batch, batch_, loss, optimizer

    model = model.to(train_device())

    checkpoint_f = lambda epoch: base.p(f'checkpoint_{epoch}')
    checkpoint_period = int(max_epochs * checkpoint_freq)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_f = v0.criterion(model=model, lammda=l2_reg_lammda, loss_name=loss_name)

    if sanity_check:
        _torchtest.assert_vars_change(
            model,
            loss_f,
            optimizer,
            list(map(_torch.TensorWrap, _torch.batch1(dataset[0]))),
            train_device(),
        )
        print('passed sanity check')

    def get_dataloader(dataset, salt):
        salt += random.randint(0, 69)

        batch_sampler = dataset.recommended_sampler(batch_size=batch_size, salt=salt)
        num_batches=len(batch_sampler)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            #shuffle=True,
            #batch_size=batch_size,
            batch_sampler=batch_sampler,
            pin_memory=True,
        )
        return data_loader

    def validation_step(engine, batch):
        with torch.no_grad():
            x, y = _torch.to(batch, train_device())
            y_ = model(x)
            loss = loss_f(y_, y)

            engine.state.metrics['avg_loss'] + loss.item()

            #print('validation batch loss', loss.item())

            return loss.item()

    def process_function(engine, batch):
        global batch_, x, y, y_, loss, shift
        if _:=_torch.is_flat_any_nan_inf(batch):
            raise Exception(f'nan/inf values in input/label detected, opinion rejected\n{_}')
        batch_ = _torch.to(batch, train_device())
        x,(y,shift) = batch_

        optimizer.zero_grad()
        y_ = model(x)

        loss = loss_f(y_, (y,shift))
        loss.backward()

        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

        optimizer.step()

        return {
            'batch_loss': loss.item(),
            'epoch_loss_avg': engine.state.metrics['avg_loss'].mean(default=0),
            'epoch_loss_var': engine.state.metrics['avg_loss'].var(default=0),
        }

    trainer = Engine(process_function)
    evaluator = Engine(validation_step)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_batch_info(engine, log_interval=debug_period, start_from=0):
        epoch = engine.state.epoch
        iteration = engine.state.iteration

        engine.state.metrics['avg_loss'] + loss.item()

        if iteration >= (1 + start_from) and iteration % log_interval == (1 + start_from):
            print(
                f"Iteration [{iteration}]: Epoch [{epoch}], Batch [{(iteration - 1) % len(engine.state.dataloader)}] - Batch Loss: {loss.item():.5f}"
            )
            y__ = model(x)

            #print(f'grad_individual_abs_sums: ', list(map(lambda a:a.grad.abs().sum().item(),model.parameters())))
            # grad_norm = _torch.grad_norm(model.parameters())
            # engine.state.metrics['avg_grad_norm'] + grad_norm
            # print(f'gradient norm: {grad_norm}')

            # l2 = v0.l2(model.parameters())
            # print(f'l2: {l2}')

            #print(f'grad: {next(model.parameters()).grad}')

            print(f'shift {shift[:4]}')

            print(f'real:\n{y[:4]}')
            print(f'predicted:\n{y_[:4]}')
            print(f'predicted after optim:\n{y__[:4]}')

            if shift[:4].abs().sum().item() != 0:
                unshifted_y = v0.unshift(y, shift)
                unshifted_y_ = v0.unshift(y_, shift)
                unshifted_y__ = v0.unshift(y__, shift)
                print(f'real unshifted:\n{unshifted_y[:4]}')
                print(f'predicted unshifted:\n{unshifted_y_[:4]}')
                print(f'predicted unshifted after optim:\n{unshifted_y__[:4]}')

            #print(f'input:\n{x[0]}')
            n_plot = 1
            slice = _torch.slice(batch_, n_plot)
            fig = v0.plot_it(
                x=slice[0],
                y=slice[1][0],
                model=model,
                y__=y_[:n_plot],
                predict_last=44,
                scale='log',
                debug=1,
            )
            model.train()
            display(fig)
            pyplot.close(fig)
        if _torch.is_any_nan(loss):
            raise Exception('nan loss')

    @trainer.on(Events.EPOCH_STARTED)
    def _(engine):
        engine.state.metrics['avg_loss'] = amends_meth.Average()
        engine.state.metrics['avg_grad_norm'] = amends_meth.Average()

        print(f"Epoch {engine.state.epoch} started")
        new_dataloader = get_dataloader(dataset, salt=engine.state.epoch)
        engine.set_data(new_dataloader)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_end(engine):
        print(f"Epoch {engine.state.epoch}")

        print(f"Epoch Loss: {engine.state.metrics['avg_loss']}")

        print(f"Epoch Gradient Norm: {engine.state.metrics['avg_grad_norm']} ")

        log_validation()

        print('-'*16)

        if engine.state.epoch % checkpoint_period == 1 or engine.state.epoch == max_epochs:
            checkpoint_path = checkpoint_f(engine.state.epoch)
            torch.save(
                {
                    'epoch': engine.state.epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_loss': engine.state.metrics['avg_loss'],
                    'avg_grad_norm': engine.state.metrics['avg_grad_norm'],
                },
                checkpoint_path,
            )

    def log_validation():
        evaluator.state.metrics['avg_loss'] = amends_meth.Average()

        model.eval()
        evaluator.run(data=get_dataloader(validation_dataset, random.randint(0, 69)))
        model.train()

        metrics = evaluator.state.metrics
        print(f"Validation Loss {metrics['avg_loss']}\n")

    ProgressBar(persist=True).attach(trainer, output_transform=lambda x: x)

    trainer.run(data=get_dataloader(dataset, 0), max_epochs=max_epochs)


def make_model(model_path, columns, n_top_gainers):
    return v0.Predictor0.readfrom(
        path=model_path,
        make_f=lambda: v0.Predictor0(
            columns,
            n_top_gainers=n_top_gainers
        ),
        debug=1,
    )


def clear():
    global coin_data, market_data, dataset, model
    coin_data = market_data = dataset = model = None
clear()

_ = src = data_proxy.DistinctFunctionsBranchingSource()

@src.function(name='coin_data')
def load_coin_data(clear_cache=0, debug=1):
    return load_preprocessed_data(clear_cache=clear_cache, debug=debug)[0]
@src.function(name='market_data')
def load_market_data(clear_cache=0, debug=1):
    r = load_preprocessed_data(clear_cache=clear_cache, debug=debug)[1]
    return r
@src.function(name='data')
def load_preprocessed_data(clear_cache=0, debug=1):
    global coin_data, market_data

    if clear_cache or coin_data is None or market_data is None:
        a = timer.Timer()
        a.start()
        coin_data, market_data = data.get_coin_and_market_data(clear_cache=clear_cache, debug=debug)
        if debug: print('fetch coin & market data', a.stop())

    return coin_data, market_data
@src.function(name='model')
def load_model(columns=None, n_top_gainers=2**4, model_path=None, debug=1):
    global model
    if model_path is None:
        model_path = 'model0'

    full_model_path = base.p(model_path)

    if columns is None:
        columns = _['coin_data']().columns[_['coin_data']().columns!='id']

    if model is None:
        model = make_model(
            columns=columns,
            n_top_gainers=n_top_gainers,
            model_path=full_model_path
        ).to(train_device())

    return model
@src.function(name='dataset')
def load_dataset(
    dataset_path=None,
    dataset_use_cache=1,
    num_samples=2 ** 15,
    num_batches=2 ** 15,
    batch_size=2 ** 6,
    seq_length=30,
    n_top_gainers=2**4,
    clear_cache=0,
    debug=1
):
    global dataset
    if dataset_path is None:
        dataset_path = 'dataset0'

    full_dataset_path = base.p(dataset_path)

    if dataset is None or clear_cache:
        timer.start()
        dataset = make_dataset(
            dataset_deer=full_dataset_path,
            num_samples=num_samples,
            num_batches=num_batches,
            batch_size=batch_size,
            seq_length=seq_length,
            n_top_gainers=n_top_gainers,
            dataset_use_cache=dataset_use_cache,
            clear_cache=clear_cache,
            debug=debug
        )
        if debug: print('made dataset in', timer.stop())

    return dataset



def make_dataset(
    dataset_deer,
    dataset_use_cache,
    num_samples,
    seq_length,
    n_top_gainers,
    num_batches=None,
    batch_size=None,
    clear_cache=0,
    debug=0
):
    if dataset_use_cache and not clear_cache and dataset_deer is not None and os.path.isfile(dataset_deer):
        return torch.load(dataset_deer)
    return _torch.DiskDataset(
        dataset_deer=dataset_deer if dataset_use_cache else None,
        load_f=lambda i, num_samples: v0.FixedSeqLenDataset(
            clean_coin_data=_['coin_data'](),
            clean_market_data=_['market_data'](),
            num_samples=num_samples,
            n_top_gainers=n_top_gainers,
            salt=i,
            seq_length=seq_length,
            debug=0,
        ),
        num_per_dataset_samples=2**16,
        num_total_samples=num_samples,
    )


def load_basics(
    clear_cache,
    dataset_path,
    dataset_use_cache,
    model_path,
    num_samples,
    num_batches,
    batch_size,
    n_top_gainers,
    seq_length,
    debug=0
):
    global model, dataset, coin_data, market_data

    coin_data, market_data = load_preprocessed_data(clear_cache=clear_cache, debug=debug)

    model = load_model(model_path=model_path, n_top_gainers=n_top_gainers, debug=debug)

    dataset = load_dataset(
        dataset_path=dataset_path,
        num_samples=num_samples,
        num_batches=num_batches,
        batch_size=batch_size,
        dataset_use_cache=dataset_use_cache,
        seq_length=seq_length,
        n_top_gainers=n_top_gainers,
        clear_cache=clear_cache,
        debug=debug,
    )

    return coin_data, market_data, model, dataset

def main(
    clear_cache=0,

    model_path=None,

    num_epochs=2**3,
    batch_size=2**6,
    learning_rate=0.01,
    gradient_clipping=1,
    l2_reg_lammda=None,
    loss_name='simp_nll',

    dataset_path=None,
    dataset_use_cache=1,
    seq_length=60,
    n_top_gainers=2**4,
    num_samples=2**16,

    sanity_check=1,
    debug_period=2**13,
    debug=0,

    # obsolete params
    num_batches=2 ** 15,
    start_batch=None,
):
    global coin_data, market_data, model, dataset

    coin_data, market_data, model, dataset = load_basics(
        clear_cache=clear_cache,
        dataset_path=dataset_path,
        model_path=model_path,
        num_samples=num_samples,
        num_batches=num_batches,
        dataset_use_cache=dataset_use_cache,
        batch_size=batch_size,
        n_top_gainers=n_top_gainers,
        seq_length=seq_length,
        debug=debug,
    )

    validation_dataset = make_dataset(
                    dataset_deer=base.p('validation_dataset'),
                    dataset_use_cache=1,
                    num_samples=int(.1 * num_samples),
                    seq_length=seq_length,
                    n_top_gainers=n_top_gainers,
                    clear_cache=clear_cache,
                    debug=debug,
                )

    while 1:
        try:
            timer.start()
            model.train()
            train(
                model,
                dataset,
                max_epochs=num_epochs,
                learning_rate=learning_rate,
                debug_period=debug_period,
                sanity_check=sanity_check,
                start_batch=start_batch,
                gradient_clipping=gradient_clipping,
                l2_reg_lammda=l2_reg_lammda,
                batch_size=batch_size,
                loss_name=loss_name,
                validation_dataset=validation_dataset,
            )
        except KeyboardInterrupt:
            1
        except Exception as e:
            print(exceptional.like_python(e))
        finally:
            print(f'GG, trained in {timer.stop()/60/60} hours')

            model.writeback(base.p(model_path), mode='state_dict', debug=1)
            if dataset_use_cache:
                dataset.writeback()
                validation_dataset.writeback()

            print('wrote back model and dataset')

            break