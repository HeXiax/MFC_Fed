import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import scipy.io as sio
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
import random
import wandb
from fast_pytorch_kmeans import KMeans as KMeansP
from torch.autograd import Variable
from tqdm import tqdm
from function import DiffLoss
from model_utils import read_mv_mnist
from os import path, makedirs
import copy
from model_utils import read_user_mv_mnist
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn.parameter import Parameter
from function import calculate_metrics


METRIC_PRINT = 'metrics: ' + ', '.join(['{:.4f}'] * 4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def cluster_acc(Y_pred, Y):
    from scipy.optimize import linear_sum_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i,j] for i,j in zip(ind[0], ind[1])])*1.0/Y_pred.size, w

class Net_Decoder(nn.Module):
    def __init__(self, in_dim, hiddens_dim):
        super(Net_Decoder, self).__init__()

        self.decoder_fc = torch.nn.Sequential(
            torch.nn.Linear(hiddens_dim[2]*2, hiddens_dim[1]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm1d(hiddens_dim[1]),
            torch.nn.Linear(hiddens_dim[1], hiddens_dim[0]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm1d(hiddens_dim[0]),
            torch.nn.Linear(hiddens_dim[0], in_dim))

    def forward(self, x):
        output = self.decoder_fc(x)
        return output

class Net(nn.Module):
    def __init__(self, in_dim, hiddens_dim, args):
        super(Net, self).__init__()

        self.shared_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hiddens_dim[0]),
            nn.LeakyReLU(0.2, inplace=True),  # wiki txt上效果好
            torch.nn.BatchNorm1d(hiddens_dim[0]),
            torch.nn.Linear(hiddens_dim[0], hiddens_dim[1]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm1d(hiddens_dim[1]),
            torch.nn.Linear(hiddens_dim[1], hiddens_dim[2]))

        self.cluster_layer = Parameter(torch.Tensor(10, hiddens_dim[2]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x):

        result = []
        shared_latent = self.shared_encoder(x)
        result.append(shared_latent)

        q = 1.0 / (1.0 + torch.sum(torch.pow(shared_latent.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        result.append(q)

        return result

class Net_Private(nn.Module):
    def __init__(self, in_dim, hiddens_dim):
        super(Net_Private, self).__init__()

        self.private_encoder = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hiddens_dim[0]),
            nn.LeakyReLU(0.2, inplace=True),  # wiki txt上效果好
            torch.nn.BatchNorm1d(hiddens_dim[0]),
            torch.nn.Linear(hiddens_dim[0], hiddens_dim[1]),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.BatchNorm1d(hiddens_dim[1]),
            torch.nn.Linear(hiddens_dim[1], hiddens_dim[2]))

        #self.private_encoder_pre_class = torch.nn.Linear(hiddens_dim[2], 10)

        self.cluster_layer = Parameter(torch.Tensor(10, hiddens_dim[2]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def forward(self, x):

        result = []
        private_latent = self.private_encoder(x)
        result.append(private_latent)

        q = 1.0 / (1.0 + torch.sum(torch.pow(private_latent.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        result.append(q)

        return result



def get_centroids(latent_z, nClusters):
    kmeans = KMeans(n_clusters = nClusters).fit(latent_z)
    return kmeans.cluster_centers_

def get_global_centroids(args, Nets, data, device):
    local_latent_z_ls = []
    local_centroids_ls = []
    for i in range(args.numusers):
        Nets[f'model_{i}'].eval()
        Nets[f'model_private_{i}'].eval()
        id, train_data = read_user_mv_mnist(i, data)
        trainloader = DataLoader(train_data, args.batch_size, shuffle=False)
        latent_z = []
        with torch.no_grad():
            for x, _ in trainloader:  # use the training set to get the latent features
                x = x.to(device)
                private_z, class_label_private = Nets[f'model_private_{i}'](x)
                result = Nets[f'model_{i}'](x)
                share_z, class_label_shared = result
                z = torch.cat([share_z, private_z], dim=1)
                latent_z.append(z)

        latent_z = torch.cat(latent_z).cpu().numpy()  
        local_latent_z_ls.append(latent_z)

        local_centroids = get_centroids(latent_z, args.k)
        local_centroids_ls.append(local_centroids)  
    global_centroids = np.mean(local_centroids_ls, axis=0)
    # local_centroids_all = np.concatenate(local_centroids_ls) 
    # global_centroids = get_centroids(local_centroids_all, args.k)  
    return global_centroids



def clustering_by_cosine_similarity(args, data, Nets, global_centroids, device):
    pseudo_labels = []
    for idx in range(args.numusers):
        Nets[f'model_{idx}'].eval()
        Nets[f'model_private_{idx}'].eval()
        id, train_data = read_user_mv_mnist(idx, data)
        train_loader = DataLoader(train_data, args.batch_size, shuffle=False)

        latent_z = []
        latent_private = []
        ground_truth = []
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                private_z, class_label_private = Nets[f'model_private_{idx}'](x)
                result = Nets[f'model_{idx}'](x)
                share_z, class_label_shared = result
                z = torch.cat([share_z, private_z], dim=1)
                latent_z.append(z)
                latent_private.append(private_z)
                ground_truth.append(y.cpu().numpy())
        latent_z_all = torch.cat(latent_z, 0).cpu().numpy()
        latent_private_all = torch.cat(latent_private, 0).cpu().numpy()
        ground_truth = np.concatenate(ground_truth)
        #pred_private = cosine_similarity(latent_private_all, global_centroids).argmax(1)
        pred1 = cosine_similarity(latent_z_all, global_centroids).argmax(1)
        q = 1.0 / (1.0 + np.sum(np.power(np.expand_dims(latent_z_all, axis=1) - global_centroids, 2), axis=2) / 1.0)
        q = np.power(q, (1.0 + 1.0) / 2.0)
        q = (q.T / np.sum(q, axis=1)).T
        #p = target_distribution(q)
        pred = q.argmax(1)
        pseudo_labels.append(pred)
        acc, nmi = cluster_acc(pred, ground_truth)[0], NMI(ground_truth, pred)
        print(f'acc: {acc: .4f} | nmi: {nmi: .4f}')

        acc1, nmi1 = cluster_acc(pred1, ground_truth)[0], NMI(ground_truth, pred1)
        print(f'acc_cos: {acc1: .4f} | nmi_cos: {nmi1: .4f}')
        # acc_p,nmi_p = cluster_acc(pred_private, ground_truth)[0], NMI(ground_truth, pred_private)
        # print(f'acc_p: {acc_p: .4f} | nmi_p: {nmi_p: .4f}')
    return pseudo_labels, acc, nmi

def pseudo_label(latent, centroids):
    q = 1.0 / (1.0 + torch.sum(torch.pow(latent.unsqueeze(1) - centroids, 2), 2) / 1)
    q = q.pow((1 + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()



def train_ae(args, config, data, num_sample, trial_dir, device):
    Nets = locals()
    global_model = Net(config['in_dim'], config['hiddens_dim'], args).to(device)
    loss_diff = DiffLoss().to(device)

    save_path = path.join(trial_dir, f"model_pretrain_mnist.pt")
    if not path.exists(save_path):
      
        for idx in range(args.numusers):
            Nets[f'model_{idx}'] = Net(config['in_dim'], config['hiddens_dim'], args).to(device)
            Nets[f'model_private_{idx}'] = Net_Private(config['in_dim'], config['hiddens_dim']).to(device)
            Nets[f'model_decoder_{idx}'] = Net_Decoder(config['in_dim'], config['hiddens_dim']).to(device)
            Nets[f'optim_{idx}'] = torch.optim.Adam(Nets[f'model_{idx}'].parameters(), lr=args.lr)
            Nets[f'optim_private_{idx}'] = torch.optim.Adam(Nets[f'model_private_{idx}'].parameters(), lr=args.lr)
            Nets[f'optim_decoder_{idx}'] = torch.optim.Adam(Nets[f'model_decoder_{idx}'].parameters(), lr=args.lr)

        print(f'pretraining on: {device}')
        for g_epoch in range(5): 
            global_model.eval()
            global_w = global_model.state_dict()
            train_sample = []
            print(f'global epoch: {g_epoch}')

            for idx in range(args.numusers):
                id, train_data = read_user_mv_mnist(idx, data)
                trainloader = DataLoader(train_data, args.batch_size, shuffle=True)
                len_dataloader = len(trainloader)
                train_sample.append(len(train_data))

                Nets[f'model_{idx}'].load_state_dict(copy.deepcopy(global_w))
                Nets[f'model_{idx}'].train()
                Nets[f'model_private_{idx}'].train()
                Nets[f'model_decoder_{idx}'].train()

                for epoch in range(1, args.local_epochs + 1): 
                    i = 0
                    total_loss = 0
                    latent_z = []
                    ys = []

                    for x, y in trainloader:

                        x = x.to(device)
                        y = y.to(device)
                        ys.append(y)

                        Nets[f'optim_{idx}'].zero_grad()
                        Nets[f'optim_private_{idx}'].zero_grad()
                        Nets[f'optim_decoder_{idx}'].zero_grad()

                        loss = 0

                        private_result = Nets[f'model_private_{idx}'](x)
                        
                        private_code, class_label_private = private_result

                        share_result = Nets[f'model_{idx}'](x)
                        share_code, class_label_shared = share_result

                        latent_private_share = torch.cat((private_code, share_code), dim=1)
                        x_recon = Nets[f'model_decoder_{idx}'](latent_private_share)
                        loss_recon = F.mse_loss(x, x_recon)
                        loss += loss_recon

                        latent_z.append(latent_private_share)

                        diff = args.beta_weight * loss_diff(private_code, share_code)  
                        loss += diff

                        loss.backward()
                        Nets[f'optim_{idx}'].step()
                        Nets[f'optim_private_{idx}'].step()
                        Nets[f'optim_decoder_{idx}'].step()


                        i += 1
                        total_loss += loss.item()

                    print(f'epoch: {epoch}, total_loss: {total_loss/len_dataloader:.4f}')
                    latent_z = torch.cat(latent_z, dim=0).cpu().detach().numpy()
                    labels = torch.cat(ys, dim=0).cpu().detach().numpy()
                    kmeans = KMeans(n_clusters=args.k, n_init=20).fit(latent_z)
                    acc, nmi = cluster_acc(kmeans.labels_, labels)[0], NMI(kmeans.labels_, labels)
                    print(f'num_user:{idx} | epochs: {epoch} | acc: {acc: .4f} | nmi: {nmi: .4f}')

            # Averaging the local models' parameters to get global model
            net_para = Nets[f'model_{0}'].state_dict()
            for key in net_para:
                global_w[key] = net_para[key] * train_sample[0]/ num_sample
            # Averaging the local models' parameters to get global model
            for idx in range(1, args.numusers):
                net_para = Nets[f'model_{idx}'].state_dict()
                for key in net_para:
                    global_w[key] += net_para[key] * train_sample[idx] / num_sample
            global_model.load_state_dict(copy.deepcopy(global_w))

            global_centroids = get_global_centroids(args, Nets, data, device)
            pseudo_labels, acc, nmi = clustering_by_cosine_similarity(args, data, Nets, global_centroids, device)

        for idx in range(args.numusers):
            torch.save(Nets[f'model_{idx}'].state_dict(), path.join(trial_dir, f"model_{idx}.pt"))
            torch.save(Nets[f'model_private_{idx}'].state_dict(), path.join(trial_dir, f"model_private_{idx}.pt"))
            torch.save(Nets[f'model_decoder_{idx}'].state_dict(), path.join(trial_dir, f"model_decoder_{idx}.pt"))

        torch.save(global_model.state_dict(), save_path)

    checkpoint = torch.load(save_path, map_location=device)
    global_model.load_state_dict(checkpoint)  
    for idx in range(args.numusers):
        checkpoint_share = torch.load(path.join(trial_dir, f"model_{idx}.pt"), map_location=device)
        Nets[f'model_{idx}'] = Net(config['in_dim'], config['hiddens_dim'], args).to(device)
        Nets[f'model_{idx}'].load_state_dict(copy.deepcopy(checkpoint_share))

        checkpoint_private = torch.load(path.join(trial_dir, f"model_private_{idx}.pt"), map_location=device)
        Nets[f'model_private_{idx}'] = Net_Private(config['in_dim'], config['hiddens_dim']).to(device)
        Nets[f'model_private_{idx}'].load_state_dict(copy.deepcopy(checkpoint_private))

        Nets[f'optim_{idx}'] = torch.optim.Adam(Nets[f'model_{idx}'].parameters(), lr=args.lr)
        Nets[f'optim_private_{idx}'] = torch.optim.Adam(Nets[f'model_private_{idx}'].parameters(), lr=args.lr)

    global_centroids = get_global_centroids(args, Nets, data, device)
    pseudo_labels, acc, nmi = clustering_by_cosine_similarity(args, data, Nets, global_centroids, device)
    return Nets, global_model, pseudo_labels



def main(args, config, data, num_sample, trial_dir, device):
    Nets = locals()
    save_path = path.join(trial_dir, f"model_pretrain_mnist.pt")

    global_model = Net(config['in_dim'], config['hiddens_dim'], args).to(device)
    checkpoint = torch.load(save_path, map_location=device)
    global_model.load_state_dict(checkpoint) 

    loss_diff = DiffLoss().to(device)


    for idx in range(args.numusers):
        checkpoint_share = torch.load(path.join(trial_dir, f"model_{idx}.pt"), map_location=device)

        Nets[f'model_{idx}'] = Net(config['in_dim'], config['hiddens_dim'], args).to(device)
        Nets[f'model_{idx}'].load_state_dict(copy.deepcopy(checkpoint_share))

        checkpoint_private = torch.load(path.join(trial_dir, f"model_private_{idx}.pt"), map_location=device)
        Nets[f'model_private_{idx}'] = Net_Private(config['in_dim'], config['hiddens_dim']).to(device)
        Nets[f'model_private_{idx}'].load_state_dict(copy.deepcopy(checkpoint_private))

        checkpoint_decoder = torch.load(path.join(trial_dir, f"model_decoder_{idx}.pt"), map_location=device)
        Nets[f'model_decoder_{idx}'] = Net_Decoder(config['in_dim'], config['hiddens_dim']).to(device)
        Nets[f'model_decoder_{idx}'].load_state_dict(copy.deepcopy(checkpoint_decoder))

        Nets[f'optim_{idx}'] = torch.optim.Adam(Nets[f'model_{idx}'].parameters(), lr=args.lr)
        Nets[f'optim_private_{idx}'] = torch.optim.Adam(Nets[f'model_private_{idx}'].parameters(), lr=args.lr)
        Nets[f'optim_decoder_{idx}'] = torch.optim.Adam(Nets[f'model_decoder_{idx}'].parameters(), lr=args.lr)


    with torch.no_grad():
        local_centroids_ls = []
        for idx in range(args.numusers):
            id, train_data = read_user_mv_mnist(idx, data)
            features = [data[0] for data in train_data]
            labels = [data[1] for data in train_data]
            input_data = torch.stack(features).to(device)
            labels = torch.stack(labels).to(device)
            labels = labels.cpu().detach().numpy()

            private_result = Nets[f'model_private_{idx}'](input_data)
            private_code, class_label_private = private_result
            share_result = Nets[f'model_{idx}'](input_data)
            share_code, class_label_shared = share_result
            latent_z = torch.cat((private_code, share_code), dim=1)

            kmeans = KMeans(n_clusters=args.k)
            y_pred = kmeans.fit_predict(private_code.cpu().numpy())
            Nets[f'model_private_{idx}'].cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

            kmeans_share = KMeans(n_clusters=args.k)
            y_pred_share = kmeans_share.fit_predict(share_code.cpu().numpy())
            Nets[f'model_{idx}'].cluster_layer.data = torch.tensor(kmeans_share.cluster_centers_).to(device)
            #local_centroids_ls.append(kmeans_share.cluster_centers_)

            kmeans_s_p = KMeans(n_clusters=args.k)
            y_pred_s_p = kmeans_s_p.fit_predict(latent_z.cpu().numpy())
 
        global_centroids = get_global_centroids(args, Nets, data, device)
        pseudo_labels, acc, nmi = clustering_by_cosine_similarity(args, data, Nets, global_centroids, device)
        global_centroids = torch.tensor(global_centroids).to(device)

    print('---------begin training------------')
    for g_epoch in range(10):

        global_model.eval()
        global_w = global_model.state_dict()
        train_sample = []
        print(f'global epoch: {g_epoch}')

        local_centroids_all = []
        train_sample = []

        acc_g = []
        nmi_g = []
        fs_g = []
        purity_g = []


        for idx in range(args.numusers):
            id, train_data = read_user_mv_mnist(idx, data)
            train_sample.append(len(train_data))
            features = [data[0] for data in train_data]
            labels = [data[1] for data in train_data]
            input_data = torch.stack(features).to(device)
            labels = torch.stack(labels).to(device)
            labels = labels.cpu().detach().numpy()

            Nets[f'model_{idx}'].load_state_dict(copy.deepcopy(global_w))
            Nets[f'model_{idx}'].train()
            Nets[f'model_private_{idx}'].train()
            Nets[f'model_decoder_{idx}'].train()

            accs = []

            acc_ls = []
            nmi_ls = []
            fs_ls = []
            purity_ls = []

            for epoch in range(1, args.local_epochs + 1):
                Nets[f'optim_{idx}'].zero_grad()
                Nets[f'optim_private_{idx}'].zero_grad()
                Nets[f'optim_decoder_{idx}'].zero_grad()

                loss = 0

                private_result = Nets[f'model_private_{idx}'](input_data)
                private_code, class_label_private = private_result


                share_result = Nets[f'model_{idx}'](input_data)
                share_code, class_label_shared = share_result


                latent_z = torch.cat((private_code, share_code), dim=1)

                q = pseudo_label(latent_z, global_centroids)
                p = target_distribution(q.data)

                latent_private_share = torch.cat((private_code, share_code), dim=1)
                x_recon = Nets[f'model_decoder_{idx}'](latent_private_share)
                loss_recon = F.mse_loss(input_data, x_recon)
                loss += loss_recon

                diff = args.beta_weight * loss_diff(private_code, share_code) 
                loss += diff

                kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                kl_loss_private = F.kl_div(class_label_private.log(), p, reduction='batchmean')
                kl_loss_shared = F.kl_div(class_label_shared.log(), p, reduction='batchmean')
                loss += 0.1 * kl_loss + 0.01 * kl_loss_private + 0.01 * kl_loss_shared

                loss.backward()
                Nets[f'optim_{idx}'].step()
                Nets[f'optim_private_{idx}'].step()
                Nets[f'optim_decoder_{idx}'].step()

                kmeans = KMeans(n_clusters=args.k)
                y_pred = kmeans.fit_predict(latent_z.detach().cpu().numpy())
                y_p = class_label_private.cpu().detach().numpy().argmax(1)
                y_s = class_label_shared.cpu().detach().numpy().argmax(1)

                train_metrics = calculate_metrics(labels, y_pred)
                acc, nmi, f, purity = train_metrics
                print('>Train', METRIC_PRINT.format(*train_metrics))
                acc_ls.append(acc)
                nmi_ls.append(nmi)
                fs_ls.append(f)
                purity_ls.append(purity)

            acc = max(acc_ls)
            nmi = nmi_ls[np.where(acc_ls == np.max(acc_ls))[0][0]]
            f = fs_ls[np.where(acc_ls == np.max(acc_ls))[0][0]]
            purity = purity_ls[np.where(acc_ls == np.max(acc_ls))[0][0]]
            # print(f'num_user:{idx} | acc: {acc: .4f} | nmi: {nmi: .4f} | f: {f: .4f} | purity: {purity: .4f}')
            acc_g.append(acc)
            nmi_g.append(nmi)
            fs_g.append(f)
            purity_g.append(purity)

        print(f'global epoch: {g_epoch} | acc: {np.mean(acc_g): .4f} | nmi: {np.mean(nmi_g): .4f} | f: {np.mean(fs_g): .4f} | purity: {np.mean(purity_g): .4f}')

        #local_centroids_all = torch.cat(local_centroids_all, 0)
        #global_centroids = get_centroids(local_centroids_all.cpu().numpy(), args.k)
        global_centroids = get_global_centroids(args, Nets, data, device)
        global_centroids = torch.tensor(global_centroids).to(device)

        # Averaging the local models' parameters to get global model
        net_para = Nets[f'model_{0}'].state_dict()
        for key in net_para:
            global_w[key] = net_para[key] * train_sample[0] / num_sample
        # Averaging the local models' parameters to get global model
        for idx in range(1, args.numusers):
            net_para = Nets[f'model_{idx}'].state_dict()
            for key in net_para:
                global_w[key] += net_para[key] * train_sample[idx] / num_sample
        global_model.load_state_dict(copy.deepcopy(global_w))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="federated_train")
    parser.add_argument("--dataset", default="MNIST_USPS",
                        choices=["BDGP", "Scene", "wikipedia", "nuswide", "Fashion", "MNIST_USPS"])
    parser.add_argument("--n_sample", default=5000, choices=[2500, 4485, 1200, 2000, 10000])
    parser.add_argument("--view", default=2, choices=[2, 3, 5, 6, 6])
    parser.add_argument("--n_clients", type=int, default=2)
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--has_filename", default=True, type=bool)
    parser.add_argument("--modal", type=str, default="img", choices=["img", "txt"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="Local learning rate")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--active_domain_loss_step", type=int, default=150)
    parser.add_argument("--gamma_weight", type=float, default=0.4)
    parser.add_argument("--beta_weight", type=float, default=0.6)
    parser.add_argument("--arcface_weight", type=float, default=0.4)
    parser.add_argument("--spread", type=float, default=0.4)
    parser.add_argument("--step_decay_weight", type=float, default=0.95)
    parser.add_argument("--save_dir", default='data/MNIST_USPS/client4/v3')
    parser.add_argument("--data_dir", default='data/MNIST_USPS/')
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--numusers", type=int, default=4, help="Number of Users per round")



    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = dict()
    if args.dataset == 'nuswide':
        config['in_dim'] = 1000
        config['hiddens_dim'] = [1024, 1024, 128]
    elif args.dataset == 'wikipedia':
        config['in_dim'] = 2048
        config['hiddens_dim'] = [1024, 256, 128]
    elif args.dataset == 'MNIST_USPS':
        config['in_dim'] = 784
        config['hiddens_dim'] = [1024, 1024, 256]
    set_seed(args.seed)

    trial_dir = path.join(args.save_dir, 'outputs')
    if not path.exists(trial_dir):
        makedirs(trial_dir)

    data = read_mv_mnist(args.data_dir)
    import time

    start = time.time()

    train_ae(args, config, data, args.n_sample, trial_dir, device)

    main(args, config, data, args.n_sample, trial_dir, device)
    end = time.time()
    print(f'Running time: {end - start} Seconds')





