# Optimizer.py
import optuna
from Carcinoma import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables
config = Config()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Data Preparation
data_frame = DataFrame()
dcm_reader = DCMReader(data_frame)
PIL_images = dcm_reader.load_images()
train_dataset, val_dataset = split(PIL_images, train_size=0.8, shuffle=True)


def create_dataloaders(batch_size):
    trainset = MyDataset(train_dataset, train_transform(config))
    validset = MyDataset(val_dataset, valid_transform(config))

    train_weights = get_weights(trainset)
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=4)

    return trainloader, validloader


def objective(trial):
    # Define the hyperparameter search space
    base_lr = trial.suggest_loguniform('base_lr', 1e-5, 1e-2)
    max_lr = trial.suggest_loguniform('max_lr', 1e-3, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Create new DataLoaders with the new batch size
    trainloader, validloader = create_dataloaders(batch_size)

    # Initialize model and optimizer with trial parameters
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=256, out_features=4)
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=5 * len(trainloader), mode='exp_range',
                         cycle_momentum=False)

    trainer = CarcinomaTrainer(criterion, optimizer, scheduler, device, data_frame)
    avg_valid_loss, _ = trainer.fit(model, trainloader, validloader, epochs=config.epochs)

    return avg_valid_loss  # We want to minimize validation loss


def run_optuna():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_trial = study.best_trial

    logger.info(f"Best trial: Value: {best_trial.value}, Params: {best_trial.params}")

    # Save the best hyperparameters
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(str(best_trial.params))


if __name__ == "__main__":
    run_optuna()
