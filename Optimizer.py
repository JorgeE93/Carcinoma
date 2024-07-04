import optuna
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from Carcinoma import CarcinomaTrainer, Config, trainloader, validloader, model, device, criterion


def objective(trial):
    # Define the hyperparameter search space
    base_lr = trial.suggest_loguniform('base_lr', 1e-5, 1e-2)
    max_lr = trial.suggest_loguniform('max_lr', 1e-3, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Update config with trial parameters
    config = Config()
    config.base_lr = base_lr
    config.max_lr = max_lr
    config.batch_size = batch_size

    # Update DataLoader with the new batch size
    trainloader.batch_size = batch_size
    validloader.batch_size = batch_size

    # Initialize model and optimizer with trial parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=5 * len(trainloader), mode='exp_range',
                         cycle_momentum=False)

    trainer = CarcinomaTrainer(criterion, optimizer, scheduler)
    avg_valid_loss, avg_valid_acc = trainer.fit(model, trainloader, validloader, epochs=config.epochs)

    return avg_valid_loss  # We want to minimize validation loss


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_trial = study.best_trial

    print(f"Best trial: Value: {best_trial.value}, Params: {best_trial.params}")

    # Save the best hyperparameters
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(str(best_trial.params))
