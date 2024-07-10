# Run_Model.py
import os
from Carcinoma import *
from Optimizer import run_optuna


def main():
    # Check if hyperparameters have already been optimized
    if not os.path.exists("best_hyperparameters.txt"):
        run_optuna()

    # Read best hyperparameters from the file
    with open("best_hyperparameters.txt", "r") as f:
        best_params = eval(f.read())

    # Update config with best hyperparameters
    config = Config()
    config.base_lr = best_params["base_lr"]
    config.max_lr = best_params["max_lr"]
    config.batch_size = best_params["batch_size"]

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and visualize images
    data_frame = DataFrame()
    dcm = DCMReader(data_frame)
    images = dcm.load_images()
    PIL_images = []
    for img, target in tqdm(images):
        im = Image.fromarray(np.uint16(img)).convert("L")
        PIL_images.append([im, target])

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    for ax, (img, target) in zip(axs.flatten(), PIL_images[:10]):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Target: {target}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Shuffle files
    train_dataset, val_dataset = split(PIL_images, train_size=0.8, shuffle=True)

    trainset = MyDataset(train_dataset, train_transform(config))
    validset = MyDataset(val_dataset, valid_transform(config))
    print(f"Trainset Size: {len(trainset)}")
    print(f"Validset Size: {len(validset)}")

    train_weights = get_weights(trainset)
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    trainloader = DataLoader(
        trainset, batch_size=config.batch_size, sampler=train_sampler, num_workers=4
    )
    validloader = DataLoader(validset, batch_size=config.batch_size, shuffle=True)
    print(f"No. of batches in trainloader: {len(trainloader)}")
    print(f"No. of Total examples: {len(trainloader.dataset)}")

    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(in_features=1280, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=256, out_features=4),
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_lr)
    scheduler = CyclicLR(
        optimizer,
        base_lr=config.base_lr,
        max_lr=config.max_lr,
        step_size_up=5 * len(trainloader),
        mode="exp_range",
        cycle_momentum=False,
    )
    trainer = CarcinomaTrainer(criterion, optimizer, scheduler, device, data_frame)
    trainer.fit(model, trainloader, validloader, epochs=config.epochs)
    torch.save(model.state_dict(), "EfficientNetCarcinomaModel_best.pth")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
