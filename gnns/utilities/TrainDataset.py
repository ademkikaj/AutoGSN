from models.gin_paper import GINPaper
from models.gin_default import GINDefault
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.utils.data.dataset import Subset
from utilities.ProcessedDataset import ProcessedDataset
from sklearn.model_selection import StratifiedKFold
import torch, torch.optim.lr_scheduler as lr_scheduler, numpy as np, os, time, json, random, operator

class TrainDataset:
    def __init__(
        self,
        seed,
        dataset_name,
        feature_name,
        labelled,
        nr_of_folds,
        epochs,
        batch_size,
        hidden_channel,
        learning_rate,
        layers,
        device,
        scheduler,
        step_size,
        decay,
        dropout,
    ) -> None:
        self.seed = seed
        self._seed_everything(self.seed)
        self.dataset_name = dataset_name
        self.nr_of_folds = nr_of_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_channel = hidden_channel
        self.learning_rate = learning_rate
        self.layers = layers
        self.device = device
        self.scheduler = scheduler
        self.step_size = step_size
        self.decay = decay
        self.dropout = dropout
        self.results_file = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.feature_name = feature_name
        self.labelled = labelled
        self.data_root_dir = os.path.join(
            "experiments",
            dataset_name,
            "data",
            "processed",
        )
        if self.feature_name == "No_Features":
            self.dataset = TUDataset(
                root=self.data_root_dir,
                name=self.dataset_name
            )
        else:
            self.dataset = ProcessedDataset(
                root=self.data_root_dir,
                name=self.feature_name + ("_labelled" if self.labelled else "_unlabelled"),
            )
        self.results_dir = os.path.join(
            "experiments",
            dataset_name,
            "results",
            self.feature_name
            + ("_labelled_" if self.labelled else "_unlabelled_")
            + str(self.seed)
            + ".json",
        )
        print("Training started %s%s in device %s" % (self.feature_name, ("_labelled" if self.labelled else "_unlabelled"), self.device))
        print("Number of node features %d "% self.dataset.num_node_features)
        self.results_file = open(self.results_dir, "w")
        start_time = time.time()
        self.dataset = self.dataset.shuffle()
        self.folds = self._get_stratified_folds()
        self.results_file.write('{ "Train": [ {\n')
        self._fold_epoch_train()
        self.results_file.write('"Elapsed time": %.2f' % (time.time() - start_time))
        self.results_file.write("}]}")
        self.results_file.close()
        print("Training finished %s%s" % (self.feature_name, ("_labelled" if self.labelled else "_unlabelled")))

    def _print_data(self):
        print(self.dataset)
        print(self.dataset[0].num_nodes)
        print(self.dataset[1].num_nodes)
        print(self.folds[0][0].dataset[0].num_nodes)
        print(self.folds[0][0].dataset[1].num_nodes)

    def _seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def _get_stratified_folds(self):
        skf = StratifiedKFold(n_splits=self.nr_of_folds)
        # skf = StratifiedKFold(n_splits=self.nr_of_folds, shuffle=True, random_state=self.seed)
        folds = []
        for train_index, test_index in skf.split(self.dataset, self.dataset.data.y):
            train = torch.utils.data.Subset(self.dataset, train_index)
            test = torch.utils.data.Subset(self.dataset, test_index)
            train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=False)
            folds.append((train_loader, test_loader))
        return folds

    def _get_folds(self):
        total_size = len(self.dataset)
        fraction = 1 / self.nr_of_folds
        seg = int(total_size * fraction)
        folds = []
        for i in range(self.nr_of_folds):
            trll = 0
            trlr = i * seg
            vall = trlr
            valr = i * seg + seg
            trrl = valr
            trrr = total_size

            train_left_indices = list(range(trll, trlr))
            train_right_indices = list(range(trrl, trrr))

            train_indices = train_left_indices + train_right_indices
            val_indices = list(range(vall, valr))

            train_set = Subset(self.dataset, train_indices)
            val_set = Subset(self.dataset, val_indices)

            train_loader = DataLoader(
                train_set, batch_size=self.batch_size, shuffle=True
            )
            test_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=True)
            folds.append((train_loader, test_loader))

        return folds

    def _train(self, model, train_loader, optimizer):
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            data.to(self.device)

            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.

            loss = self.criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def _test(self, model, test_loader):
        model.eval()

        correct = 0
        for data in test_loader:  # Iterate in batches over the training/test dataset.
            data.to(self.device)

            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.

        return correct / len(test_loader.dataset)

    def _fold_epoch_train(self):
        epochs_train_acc = {}
        epochs_test_acc = {}
        folds_train_acc = {}
        folds_test_acc = {}
        for f_index, fold in enumerate(self.folds):
            self.results_file.write('"Fold %d":' % f_index)
            train_acc_agg = []
            test_acc_agg = []
            model = GINPaper(
                in_channels=self.dataset.num_node_features,
                hidden_channels=self.hidden_channel,
                num_layers=self.layers,
                out=self.dataset.num_classes,
                dropout=self.dropout,
                seed=self.seed
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            if self.scheduler:
                scheduler = lr_scheduler.StepLR(
                    optimizer, step_size=self.step_size, gamma=self.decay
                )
            train_loader = fold[0]
            test_loader = fold[1]
            epoch_res_all = []
            for epoch in range(1, self.epochs):
                self._train(model, train_loader, optimizer)
                if self.scheduler:
                    scheduler.step()
                train_acc = self._test(model, train_loader)
                test_acc = self._test(model, test_loader)
                epoch_res_all.append({"Epoch": epoch, "Train Acc": (train_acc*100), "Test Acc": (test_acc*100)})
                train_acc_agg.append((train_acc*100))
                test_acc_agg.append((test_acc*100))
            folds_train_acc[f_index] = train_acc_agg
            folds_test_acc[f_index] = test_acc_agg
            self.results_file.write(json.dumps(epoch_res_all))
            self.results_file.write(",\n")
        for k, v in folds_train_acc.items():
            for i, elem in enumerate(v):
                try:
                    epochs_train_acc[i] = epochs_train_acc[i] + [elem]
                except:
                    epochs_train_acc[i] = [elem]
        
        for k, v in folds_test_acc.items():
            for i, elem in enumerate(v):
                try:
                    epochs_test_acc[i] = epochs_test_acc[i] + [elem]
                except:
                    epochs_test_acc[i] = [elem]
        for k, v in epochs_train_acc.items():
            epochs_train_acc[k] = (np.average(v), np.std(v))
        for k, v in epochs_test_acc.items():
            epochs_test_acc[k] = (np.average(v), np.std(v))

        self.results_file.write(
            '"Results Train Acc": "%.2f±%.2f",\n'
            % (max(epochs_train_acc.items(), key=operator.itemgetter(1))[1][0], max(epochs_train_acc.items(), key=operator.itemgetter(1))[1][1])
        )
        self.results_file.write(
            '"Results Test Acc": "%.2f±%.2f",\n'
            % (max(epochs_test_acc.items(), key=operator.itemgetter(1))[1][0], max(epochs_test_acc.items(), key=operator.itemgetter(1))[1][1])
        )
