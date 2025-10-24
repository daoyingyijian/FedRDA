# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
        # 仅一层线性层，包含偏置项
        self.f1 = nn.Sequential(
            nn.Linear(input_dim, input_dim,bias=True),
            nn.ReLU(),
            nn.LayerNorm([input_dim]),
        )
    def forward(self, x):
        return self.f1(x)
class IDM(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(embed_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, embed_dim))

        self.logvar = nn.Sequential(nn.Linear(embed_dim, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, embed_dim),
                                    nn.Tanh())

    def forward(self, gfe, csfe):
        mu = self.mu(gfe)
        logvar = self.logvar(gfe)
        return -1.0 * (-(mu - csfe)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
class clientRDA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.feature_dim = list(self.model.head.parameters())[0].shape[1]
        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda
        self.gamma=args.lamda
        self.tau=0.1
        #self.optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
        
        self.KM = KM(input_dim=self.feature_dim).to(self.device)
        self.idm=IDM(embed_dim=self.feature_dim).to(self.device)
        self.optimizer = torch.optim.SGD(
            list(self.model.base.parameters()) + list(self.idm.parameters()),
            lr=self.learning_rate
        )
        #类别信息
        self.train_samples_by_class = defaultdict(int)
        trainloader = self.load_train_data()
        for i, (x, y) in enumerate(trainloader):
            for i, yy in enumerate(y):
                y_c = yy.item()
                self.train_samples_by_class[y_c] += 1 
        self.test_acc=0
    def get_intermediate_layer_output(self, data):
        """获取模型的中间层输出（例如用于 t-SNE）"""
        self.model.eval()
        with torch.no_grad():
            # 假设模型有一个中间层，我们获取它的输出
            #intermediate_output = self.model.base(data)
            rep = self.model.base(data)
            #全局表征
            rep_global = rep-self.KM(rep)  
            #个性化表征
            rep_self = self.KM(rep)
        return rep_global
    def send_updates(self):
        """将特征、标签和客户端 ID 发送到服务器"""
        features = []
        labels = []
        model_weights = []  # 新增：用于存储模型的权重
        # 打印模型的 state_dict()，查看其结构
        #print(self.model.state_dict())

        self.model.eval()
        with torch.no_grad():
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)

                # 获取中间层的特征
                intermediate_output = self.get_intermediate_layer_output(data)
                features.append(intermediate_output.cpu())

                # 收集对应的标签
                labels.append(target.cpu())

            # 获取当前客户端的模型权重（所有层的权重）
            model_weights.append(self.model.state_dict()['head.weight'].cpu().numpy())  # 假设您使用的是一个全连接层 `fc`

        # 拼接特征、标签和模型权重
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        model_weights = np.array(model_weights).flatten()  # 将权重展平为一维向量

        # 将特征、标签、权重和客户端 ID 发送给服务器
        self.client_data_up = {
            'features': features,
            'labels': labels,
            'model_weights': model_weights,  # 新增：客户端模型权重
            'client_id': self.id
        }

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
                #全局表征
                rep_global = rep-self.KM(rep)  
                #个性化表征
                rep_self = self.KM(rep)
                output = self.model.head(rep_self)
                loss = self.loss(output, y)
                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if y_c in self.global_protos and self.global_protos[y_c].numel() > 0:
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep_global) * self.lamda
                loss+= self.idm(rep_self, rep_global) *self.gamma
                # 原型对比损失
                if self.global_protos is not None:
                    proto_new2 =rep_global
                    feature_dim = self.global_protos[next(iter(self.global_protos))].shape[0]  # 获取特征维度
                    global_protos_list = []
                    for label in range(self.num_classes):
                        if label in self.global_protos:
                            global_protos_list.append(self.global_protos[label])
                        else:
                            # 用零向量或随机向量填充缺失的类
                            global_protos_list.append(torch.zeros(feature_dim).to(self.device))  # 或者 torch.randn(feature_dim)
                    global_protos_tensor = torch.stack(global_protos_list).to(self.device)
                    #global_protos_tensor = torch.stack([self.global_protos[label] for label in sorted(self.global_protos.keys())]).to(self.device)
                    positive_sim = F.cosine_similarity(proto_new2.unsqueeze(1), global_protos_tensor.unsqueeze(0), dim=2)
                    positive_sim = positive_sim[torch.arange(positive_sim.size(0)), y]
                    negative_sim = positive_sim.unsqueeze(1) - positive_sim.unsqueeze(0)
                    mask = torch.eye(negative_sim.size(0), device=negative_sim.device).bool()
                    negative_sim = negative_sim.masked_fill(mask, float('-inf'))
                    positive_exp = torch.exp(positive_sim / self.tau)
                    negative_exp = torch.exp(negative_sim / self.tau).sum(dim=1)
                    proto_contrastive_loss = -torch.log(positive_exp / (positive_exp + negative_exp))
                    loss+= proto_contrastive_loss.mean() * self.lamda
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
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
                #个性化表征
                rep_self = self.KM(rep)  # KM模块处理后的特征
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
        self.test_acc=max(self.test_acc,test_acc)
        return test_acc, test_num, auc
    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
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
                #km_out = self.KM(rep)+rep  # KM模块处理后的特征
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

def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)

