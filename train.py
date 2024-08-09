import os
import random
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
sys.argv = ['run.py']
import math
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import torch.nn.functional as F
import argparse
from metrics import get_cindex, get_rm2, evaluate_regression
from MFFDTAmodel import mymodel
from utils import *
from log.train_logger import TrainLogger
from torch_geometric.data import InMemoryDataset
import warnings
warnings.filterwarnings("ignore")
import timm
import timm.optim
import timm.scheduler



class GNNDataset(InMemoryDataset):
    def __init__(self, root, index, types='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[index])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[index + len(self.raw_paths)])

    @property
    def raw_file_names(self):
        return ['davis_fold_0.csv', 'davis_fold_1.csv', 'davis_fold_2.csv', 'davis_fold_3.csv', 'davis_fold_4.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_fold_0.pt', 'processed_data_fold_1.pt', 'processed_data_fold_2.pt', 'processed_data_fold_3.pt','processed_data_fold_4.pt', 
        'processed_test_data_fold_0.pt', 'processed_test_data_fold_1.pt', 'processed_test_data_fold_2.pt', 'processed_test_data_fold_3.pt','processed_test_data_fold_4.pt']

    def download(self):
        pass


def val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y
            running_loss.update(loss.item(), label.size(0))

    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()

    return epoch_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sample_cold_drug_davis_five_cross', help='task name')
    parser.add_argument('--save_model', action='store_true', help='whether save model or not')
    parser.add_argument('--lr', type=float, default=9e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    params = dict(
        data_root="data",
        save_dir="save",
        dataset=args.dataset,
        save_model=args.save_model,
        lr=args.lr,
        batch_size=args.batch_size
    )

    logger = TrainLogger(params)
    logger.info(__file__)

    DATASET = params.get("dataset")
    save_model = params.get("save_model")
    data_root = params.get("data_root")
    batch_size = params.get("batch_size")
    fpath = os.path.join(data_root, DATASET)


    test_set_list = []
    set_list = []
    for index in range(5):
        test_set_list.append(GNNDataset(fpath, types='test', index=index))
        set_list.append(GNNDataset(fpath, types='train', index=index))

    num_workers = 8 if sys.platform == 'linux' else 0
    print('Number of workers: ', num_workers)

    for i in range(5):
        test_loader = DataLoader(test_set_list[i], batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        concat_dataset = ConcatDataset([it for j, it in enumerate(set_list) if i != j])
        train_loader = DataLoader(dataset=concat_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        drop_last=True
                                        )
        logger.info(f"Use dataset: {DATASET}")
        logger.info(f"Number of train: {len(train_loader)}")
        logger.info(f"Number of test: {len(test_loader)}")
        device = torch.device('cuda:0')
        model = mymodel().to(device)
        epochs = 3000
        steps_per_epoch = 50
        num_iter = math.ceil((epochs * steps_per_epoch) / len(train_loader))
        break_flag = False

        optimizer = optim.Adam(model.parameters(), lr=params.get("lr"))
        criterion = nn.MSELoss()
        
        scheduler = timm.scheduler.CosineLRScheduler(
        optimizer = optimizer,
        t_initial = 500,
        lr_min = 1e-5,
        warmup_t = 20,
        warmup_lr_init=3e-4
        )

        global_step = 0
        global_epoch = 0
        early_stop_epoch = 500

        running_loss = AverageMeter()
        running_cindex = AverageMeter()
        running_best_mse = BestMeter("min")
        running_best_ci = BestMeter("max")

        model.train()
        messageforCI_best = ''
        logger.info(f"Number of num_iter: {iter}")
        for _ in range(num_iter):
            if break_flag:
                break
            for data in train_loader:
                global_step += 1       
                data = data.to(device)
                pred = model(data)

                loss = criterion(pred.view(-1), data.y.view(-1))
                cindex = get_cindex(data.y.detach().cpu().numpy().reshape(-1), pred.detach().cpu().numpy().reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), data.y.size(0)) 
                running_cindex.update(cindex, data.y.size(0))

                if global_step % steps_per_epoch == 0:

                    global_epoch += 1

                    epoch_loss = running_loss.get_average()
                    epoch_cindex = running_cindex.get_average()
                    running_loss.reset()
                    running_cindex.reset()

                    test_loss, test_cindex, test_r2, epoch_pearsonr, epoch_r2score = test_result(model, criterion, test_loader, device)

                    msg = "epoch-%d, loss-%.4f, ci-%.4f, test_loss-%.4f, test_ci-%.4f, test_pr-%.4f, test_r2score-%.4f, test_rm2-%.4f, lr-%.8f" % (global_epoch, epoch_loss, epoch_cindex, 
                                                                                                                                                  test_loss, test_cindex, 
                                                                                                                                                  epoch_pearsonr, epoch_r2score, test_r2, 
                                                                                                                                                  optimizer.state_dict()['param_groups'][0]['lr'])
                        
                    epoch_msg = "epoch%dtest_loss%.4f" % (global_epoch, test_loss)
                    logger.info(msg)
                    if test_cindex > running_best_ci.get_best():
                        running_best_ci.update(test_cindex)
                        messageforCI_best = "test_loss:%.4f, test_ci:%.4f, test_pr-%.4f, test_r2score-%.4f, test_rm2:%.4f" % (test_loss, test_cindex, epoch_pearsonr, epoch_r2score, test_r2)
                        logger.info('######' + messageforCI_best + '######')
                        if save_model:
                            save_model_dict(model, logger.get_model_dir(), epoch_msg)
                    else:
                        count = running_best_ci.counter()
                        if count > early_stop_epoch :
                            logger.info(f"early stop in epoch {global_epoch}")
                            break_flag = True
                            logger.info('fold_' + str(i) + "'s result is: " + messageforCI_best)
                            break
                        if global_epoch > 3000 :
                            logger.info('fold_' + str(i) + "'s result is: " + messageforCI_best)
                            break_flag = True
                            break

                        

                    # if test_loss < running_best_mse.get_best():
                    #     running_best_mse.update(test_loss)
                    #     messageforCI_best = "test_loss:%.4f, test_ci:%.4f, test_pr-%.4f, test_r2score-%.4f, test_rm2:%.4f" % (test_loss, test_cindex, epoch_pearsonr, epoch_r2score, test_r2)
                    #     logger.info('######' + messageforCI_best + '######')
                    #     if save_model:
                    #         save_model_dict(model, logger.get_model_dir(), epoch_msg)
                    # else:
                    #     count = running_best_mse.counter()
                    #     if count > early_stop_epoch:
                    #         logger.info(f"early stop in epoch {global_epoch}")
                    #         break_flag = True
                    #         logger.info('fold_' + str(i) + "'s result is: " + messageforCI_best)
                    #         break
            scheduler.step(_)

def test_result(model, criterion, test_loader, device):
    criterion = nn.MSELoss()
    test_loss, test_cindex, test_r2, epoch_pearsonr, epoch_r2score = test_val(model, criterion, test_loader, device)
    return test_loss, test_cindex, test_r2,  epoch_pearsonr, epoch_r2score


def test_val(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []
    prsum = 0
    count = 0
    r2scoresum = 0
    for data in dataloader:
        data = data.to(device)

        with torch.no_grad():
            pred = model(data)
            loss = criterion(pred.view(-1), data.y.view(-1))
            label = data.y.cpu()
            pred = pred.cpu()
            metric_dict = evaluate_regression(label, pred)
            prsum += metric_dict['pearsonr']
            r2scoresum += metric_dict['r2']
            count += 1
            label = label.to(device)
            pred = pred.to(device)
            pred_list.append(pred.view(-1).detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            running_loss.update(loss.item(), label.size(0))

    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    epoch_pearsonr = prsum / count
    epoch_r2score = r2scoresum / count
    epoch_cindex = get_cindex(label, pred)
    epoch_r2 = get_rm2(label, pred)
    epoch_loss = running_loss.get_average()
    running_loss.reset()

    return epoch_loss, epoch_cindex, epoch_r2, epoch_pearsonr, epoch_r2score


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False



if __name__ == "__main__":
    setup_seed(23)
    if sys.platform == 'linux':
        with open(__file__,"r",encoding="utf-8") as f:
            for line in f.readlines():
                print(line)
    main()
