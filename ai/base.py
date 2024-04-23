
import os
import posixpath





tmp_root = "J:/loli/tmp"
path = os.path.splitdrive(os.getcwd())[1]
tmp_path = posixpath.join(tmp_root, path)
def full_path():
    return posixpath.join(tmp_path, version_path)
def p(rel_path):
    return posixpath.join(full_path(), rel_path)
def set_version_path(vp):
    global version_path
    version_path = vp
    os.makedirs(full_path(), exist_ok=True)
set_version_path('v0')



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

