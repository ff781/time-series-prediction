# +
# # %reload_ext autoreload
# # %autoreload 2
# -

from amends import _pandas, text
from utils.model import stonks
import optuna
import pandas
import numpy
import torch




class AmnesicApexPredictor(torch.nn.Module):
    def __init__(
        self,
        num_input_days=2**5,
        num_days=3,
    ):
        super().__init__()
        # normalize all price values by multiplying with outstanding

        # input data from input days
        # detail_platforms onehot
        # links
        # sentiment_votes_up_percentage
        # sentiment_votes_down_percentage
        # watchlist_portfolio_users
        # status_updates
        # tickers
        # market_data.ath.btc
        # market_data.ath.usd
        # market_data.ath_date.btc
        # market_data.ath_date.usd
        # market_data.atl.btc
        # market_data.atl.usd
        # market_data.atl_date.btc
        # market_data.atl_date.usd
        # genesis_date
        # first geck price timestamp



def objective(trial):
    # Define hyperparameter search space using trial object
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])

    # Example: Set up your model, train it, evaluate it, return the evaluation score
    model = Model()  # Assuming Model() is your model class
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    train_loader, val_loader = get_data_loaders(batch_size)  # Define your data loaders

    for epoch in range(NUM_EPOCHS):
        train(model, optimizer, train_loader)  # Implement your training loop
    val_score = evaluate(model, val_loader)  # Implement your evaluation method

    # Return the score to minimize (or maximize)
    return val_score  # Assuming it's a loss to minimize

def study():
    study = optuna.create_study(direction='minimize')  # Use 'maximize' for metrics you want to maximize
    study.optimize(objective, n_trials=100)  # Specify the number of trials

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


coin_data[filter(lambda a:a.startswith('links',),coin_data.columns)].loc[4,'links.blockchain_site'].size


coin_data = stonks.data.source['coingecko']['info']
market_data = stonks.data.source['coingecko']['market']

_pandas.scale_display(10)

print(coin_data[coin_data.id=='bitcoin'])
