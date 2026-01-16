import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score
from sklearn import metrics
from sklearn.preprocessing import label_binarize
class KM(nn.Module):
    def __init__(self, input_dim):
        super(KM, self).__init__()
        self.f1 = nn.Sequential(
            nn.Linear(input_dim, input_dim,bias=True),
            nn.ReLU(),
            nn.LayerNorm([input_dim]),
        )
    def forward(self, x):
        return self.f1(x)
#vCLUB
class IDM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        self.logvar = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.Tanh()
        )

    def _loglik_per_sample(self, x, y):
        mu = self.mu(x)
        logvar = self.logvar(x)
        var = logvar.exp() + 1e-6
        ll = -((mu - y) ** 2) / var - logvar   # [B, D] (常数项省略)
        return ll.sum(dim=1)                   # [B]

    def upper_bound(self, x, y):
        ll_pos = self._loglik_per_sample(x, y)
        perm = torch.randperm(y.size(0), device=y.device)
        y_neg = y[perm]
        ll_neg = self._loglik_per_sample(x, y_neg)
        return (ll_pos - ll_neg).mean()

    def learning_loss(self, x, y):
        return (-self._loglik_per_sample(x, y)).mean()

    def forward(self, x, y):
        return self.upper_bound(x, y)

class clientRDA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.feature_dim = list(self.model.head.parameters())[0].shape[1]
        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.gamma=args.gamma
        self.tau=0.1
        #self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        
        self.KM = KM(input_dim=self.feature_dim).to(self.device)
        self.idm=IDM(embed_dim=self.feature_dim).to(self.device)
        self.opt_main = torch.optim.SGD(
            list(self.model.base.parameters()),
            lr=self.learning_rate
        )
        self.opt_idm = torch.optim.SGD(
            list(self.idm.parameters()),
            lr=self.learning_rate
        )
        self.train_samples_by_class = defaultdict(int)
        trainloader = self.load_train_data()
        for i, (x, y) in enumerate(trainloader):
            for i, yy in enumerate(y):
                y_c = yy.item()
                self.train_samples_by_class[y_c] += 1


    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.trainloader=trainloader
        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                rep_global = rep - self.KM(rep)
                rep_self = self.KM(rep)
                idm_loss = self.idm.learning_loss(rep_self.detach(), rep_global.detach())
                self.opt_idm.zero_grad()
                idm_loss.backward()
                self.opt_idm.step()
                output = self.model.head(rep_self)
                loss = self.loss(output, y)
                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if y_c in self.global_protos and self.global_protos[y_c].numel() > 0:
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep_global) * self.lamda
                if self.global_protos is not None:
                    proto_new2 = rep_global
                    feature_dim = self.global_protos[next(iter(self.global_protos))].shape[0]
                    global_protos_list = []
                    for label in range(self.num_classes):
                        if label in self.global_protos:
                            global_protos_list.append(self.global_protos[label])
                        else:
                            global_protos_list.append(torch.zeros(feature_dim).to(self.device))
                    global_protos_tensor = torch.stack(global_protos_list).to(self.device)

                    positive_sim = F.cosine_similarity(proto_new2.unsqueeze(1), global_protos_tensor.unsqueeze(0), dim=2)
                    positive_sim = positive_sim[torch.arange(positive_sim.size(0)), y]
                    negative_sim = positive_sim.unsqueeze(1) - positive_sim.unsqueeze(0)
                    mask = torch.eye(negative_sim.size(0), device=negative_sim.device).bool()
                    negative_sim = negative_sim.masked_fill(mask, float('-inf'))
                    positive_exp = torch.exp(positive_sim / self.tau)
                    negative_exp = torch.exp(negative_sim / self.tau).sum(dim=1)
                    proto_contrastive_loss = -torch.log(positive_exp / (positive_exp + negative_exp))
                    loss += proto_contrastive_loss.mean() * self.lamda
                mi_ub = self.idm(rep_self, rep_global)  
                loss = loss + (self.gamma * mi_ub)   
                self.opt_main.zero_grad()
                loss.backward()
                self.opt_main.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        self.global_protos = global_protos
    def set_parameters(self,model,R):
        for new_param, old_param in zip(model.base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()
    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                km=rep-self.KM(rep)
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(km[i, :].detach().data)

        self.protos = agg_func(protos)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                rep_self = self.KM(rep)  
                output = self.model.head(rep_self)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        return test_acc, test_num, auc
    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                km_out = self.KM(rep)
                output = self.model.head(km_out)
                loss = self.loss(output, y)        
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')
        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

