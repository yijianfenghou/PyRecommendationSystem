import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from XDeepFM.dataset.movielens import MovieLens1MDataset
from XDeepFM.Model.xdfm import ExtremeDeepFactorizationMachineModel


def get_dataset(name, path):
    return MovieLens1MDataset(path)


def get_model(name, dataset):
    field_dims = dataset.fields_dims
    return ExtremeDeepFactorizationMachineModel(field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False,
                                                mlp_dims=(16, 16), dropout=0.2)


class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuacy = 0.0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuacy:
            self.best_accuacy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0.0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main(dataset_name, dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir):
    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, (
    train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
    model = get_model(model_name, dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(num_trials=2, save_path=f'{save_dir}/{model_name}.pt')
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc = test(model, valid_data_loader, device)
        print("epoch:", epoch_i, ' validation:auc:', auc)
        if not early_stopper.is_continuable(model, auc):
            print(f'validation: best auc: {early_stopper.best_accuacy}')
            break
    auc = test(model, test_data_loader, device)
    print(f'test auc: {auc}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='movie')
    parser.add_argument('--dataset_path', default='./dataset/ML1M/ratings.dat')
    parser.add_argument('--model_name', default='xdf')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='./chkpt')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
