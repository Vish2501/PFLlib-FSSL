import copy
import gc
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as FtaLoader
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
from flcore.clients.clientbase import Client
import torch.nn.functional as F

from sklearn.preprocessing import label_binarize
from sklearn import metrics
from pytorch_metric_learning import losses


class clientSSLAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.tau = 0.1
        self.alpha = 1.0
        self.beta = 1e-2
        self.fine_tuning_epochs = args.fine_tuning_epochs
        self.loss_func = losses.NTXentLoss(
            temperature=self.tau)  # Initialize the NTXentLoss

        self.labeled_loader, self.unlabeled_loader = self.load_labelled_unlabelled_data(
            visualise=False)
        # print(" model structure:")
        # print(self.model)

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

        #     # if global_round < 500:
        #     #   conduct semi-supervised learning to update the parameter of the encoder (set the encoder-only as True)
        #     # if global_round >500:
        #     # only conduct local training to update the personalized classifier(set the gradient for encoder as zero)

        #     # replace the current train function with semi-pretrain function to update the encoder
        #     # add another function to conduct local training to update the classifer.
        self.model.train()
        start_time = time.time()

        labeled_loader, unlabeled_loader = self.labeled_loader, self.unlabeled_loader

        max_local_epochs = 5
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for batch_idx, ((x_unlabeled_1, x_unlabeled_2), (x_labeled, y_labeled)) in enumerate(zip(unlabeled_loader, labeled_loader)):

                # Unlabeled data processing
                x_unlabeled_1 = x_unlabeled_1.to(self.device)
                x_unlabeled_2 = x_unlabeled_2.to(self.device)

                z_A_i = self.model(
                    x_unlabeled_1, encoder_only=True)
                z_B_i = self.model(
                    x_unlabeled_2, encoder_only=True)
                embeddings_unlabeled = torch.cat([z_A_i, z_B_i], dim=0)
                labels_unlabeled = torch.arange(
                    z_A_i.size(0)).repeat(2).to(self.device)

                unsupervised_loss = self.loss_func(
                    embeddings_unlabeled, labels_unlabeled)

                # Labeled data processing
                x_labeled = x_labeled.to(self.device)
                y_labeled = y_labeled.to(self.device)

                embeddings_labeled = self.model(x_labeled, encoder_only=True)
                classifier_outputs = self.model(x_labeled, encoder_only=False)

                supervised_loss = self.loss(
                    classifier_outputs, y_labeled)

                # Compute KL divergence for Bayesian layers
                kl_loss = self.model.classifier.kl_loss()
                # avg_kl = (kl_loss / len(self.labeled_loader.dataset))
                kl_loss = kl_loss * self.beta

                # Combine the losses with a weight for supervised loss and KL divergence
                total_loss = self.alpha * unsupervised_loss + \
                    supervised_loss + kl_loss

                # print(
                #     f"Epoch [{epoch+1}/{max_local_epochs}], Batch [{batch_idx+1}/{len(unlabeled_loader)}]")
                # print(f"Unsupervised Loss: {unsupervised_loss.item():.4f}")
                # print(f"Supervised Loss: {supervised_loss.item():.4f}")
                # print(f"KL Loss: {kl_loss.item():.4f}")
                # print(f"Total Loss: {total_loss.item():.4f}\n")

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
        # Load both labeled and unlabeled data
        # labeled_loader, _ = self.load_labeled_unlabeled_data(
        #     visualise=False)  # Only use labeled_loader

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    # Disable gradients for all other layers
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        self.model.train()  # Train the classifier
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(self.fine_tuning_epochs):
            for batch_idx, (x, y) in enumerate(self.labeled_loader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Forward pass with both encoder and classifier
                outputs = self.model(x, encoder_only=False)

                # print(
                #     f"Outputs: {outputs.size()}, IsLeaf: {outputs.is_leaf}, RequiresGrad: {outputs.requires_grad}")
                # loss = self.loss(outputs, y)

                # Compute the cross-entropy loss
                ce_loss = self.loss(outputs, y)
                # print(f"CE Loss: {ce_loss}")

                # Compute the KL divergence for the Bayesian layer
                # Assuming the classifier is the Bayesian Linear layer
                kl_loss = self.model.classifier.kl_loss()
                kl_loss = kl_loss * self.beta
                # print(f"KL Loss: {kl_loss}")
                # avg_kl = (kl_loss / len(self.labeled_loader.dataset))
                # print(f"Avg KL: {avg_kl}")
                loss = ce_loss + kl_loss

                # Combine the losses
                # loss = ce_loss + avg_kl
                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()
                # print(f"Loss: {loss.item()}, After step: {outputs.grad_fn}")

            # for param in self.model.parameters():
            #     param.data = param.data.detach().clone()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        print("Local classifier training completed")

    def normalise(self, image):
        mean = torch.tensor(
            [0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.tensor(
            [0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        image = (image - mean[None, :, None, None]) / std[None, :, None, None]
        return image

     # Since the data is already normalised to apply the augmentation
    # it is denormalised and then augmentation applided with normalissation to ensure image is not corrupted
    def get_augmentation(self):
        augmentation = transforms.Compose([
            # Normalize to [-1, 1] range
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)   # Random color jitter
            ], p=0.5),  # Adjusted probability
            # Convert to grayscale with 20% probability
            transforms.RandomGrayscale(p=0.2),

            # Normalize back to original range
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        return augmentation

    # this function is defined to denormalise the image for visualization purposes

    def denormalise(self, image):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        image = image * std + mean
        return image.clamp(0, 1)

    # this function is defined to visualise the augmented images

    def visualise_augmented_loader(self, original_images, augmented_images1, augmented_images2):

        num_images = min(10, len(original_images))
        fig, axes = plt.subplots(nrows=3, ncols=num_images, figsize=(20, 6))
        for i in range(num_images):
            # Display original images
            original_img = self.denormalise(original_images[i])  # Denormalise
            axes[0, i].imshow(original_img.permute(1, 2, 0))
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')

            # Display augmented images set 1
            augmented_img1 = self.denormalise(
                augmented_images1[i])  # Denormalise
            axes[1, i].imshow(augmented_img1.permute(1, 2, 0))
            axes[1, i].set_title("Augmented 1")
            axes[1, i].axis('off')

            # Display augmented images set 2
            augmented_img2 = self.denormalise(
                augmented_images2[i])  # Denormalise
            axes[2, i].imshow(augmented_img2.permute(1, 2, 0))
            axes[2, i].set_title("Augmented 2")
        axes[2, i].axis('off')

        plt.show()

    # this function is defined to apply the augmentation to the unlabelled dataset

    def apply_augmentation_to_dataset(self, dataset):
        augmented_images1, augmented_images2 = [], []
        augmentation = self.get_augmentation()

        for img in dataset:
            if isinstance(img, tuple):  # In case DataLoader returns (data, target) tuples
                img = img[0]

            # denorm_img = self.denormalise(img)

            augmented_img1 = augmentation(img)
            augmented_img2 = augmentation(img)

            # augmented_img1 = self.normalise(augmented_img1)
            # augmented_img2 = self.normalise(augmented_img2)

            augmented_images1.append(augmented_img1)
            augmented_images2.append(augmented_img2)

        return augmented_images1, augmented_images2

    # this function is defined to load the labelled and unlabelled data for training and is partitioned into 10% labelled and 90% unlabelled

    def load_labelled_unlabelled_data(self, batch_size=None, visualise=False):
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        print("batch_size: ", batch_size)

        length = len(train_data)
        labeled_data = train_data[:int(0.2 * length)]
        unlabeled_data = train_data[int(0.2 * length):]

        if visualise:
            # Store originals only for visualization
            original_images = [img[0] for img in unlabeled_data] if isinstance(
                unlabeled_data[0], tuple) else unlabeled_data

        # Apply augmentation to unlabeled data
        augmented_images1, augmented_images2 = self.apply_augmentation_to_dataset(
            unlabeled_data)

        unlabeled_loader = DataLoader(
            list(zip(augmented_images1, augmented_images2)), batch_size, drop_last=True, shuffle=True)

        labeled_loader = DataLoader(
            labeled_data, batch_size, drop_last=True, shuffle=True)

        # Visualise the original and augmented images if visuualize is True
        if visualise:
            self.visualise_augmented_loader(
                original_images, augmented_images1, augmented_images2)

        return labeled_loader, unlabeled_loader

    def bnn_test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def bnn_train_metrics(self):
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
                output = self.model(x, encoder_only=False)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
