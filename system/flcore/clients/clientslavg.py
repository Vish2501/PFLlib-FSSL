import copy
import gc
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from pytorch_metric_learning import losses


class clientSLAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.alpha = 1.0
        self.beta = 1e-2
        self.fine_tuning_epochs = args.fine_tuning_epochs

        # Load labeled data using the existing method from Client base class
        self.labeled_loader = self.load_train_data()

        self.w_params = []
        self.r_params = []

        for name, param in self.model.named_parameters():
            if "classifier" in name:
                self.w_params.append(param)
            else:
                self.r_params.append(param)

        self.optimizer_W = torch.optim.Adam(
            self.w_params, lr=self.learning_rate)

        for param in self.model.classifier.parameters():
            param.requires_grad = False

    def set_learning_rate(self, learning_rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

    def train(self):
        self.model.train()
        start_time = time.time()

        labeled_loader = self.labeled_loader

        max_local_epochs = 5
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for batch_idx, (x_labeled, y_labeled) in enumerate(labeled_loader):
                x_labeled = x_labeled.to(self.device)
                y_labeled = y_labeled.to(self.device)

                embeddings_labeled = self.model(x_labeled, encoder_only=True)
                classifier_outputs = self.model(x_labeled, encoder_only=False)

                supervised_loss = self.loss(
                    classifier_outputs, y_labeled)

                # Compute KL divergence for Bayesian layers
                kl_loss = self.model.classifier.kl_loss()
                kl_loss = kl_loss * self.beta

                # Combine the losses with a weight for supervised loss and KL divergence
                total_loss = supervised_loss + kl_loss

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # Learning rate decay if applicable
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def fine_tune(self):
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(self.fine_tuning_epochs):
            for batch_idx, (x, y) in enumerate(self.labeled_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                outputs = self.model(x, encoder_only=False)

                ce_loss = self.loss(outputs, y)

                kl_loss = self.model.classifier.kl_loss()
                kl_loss = kl_loss * self.beta
                loss = ce_loss + kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        print("Local classifier training completed")

    def bnn_test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x, encoder_only=False)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(),
                                    classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def bnn_train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x, encoder_only=False)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
