import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
import numpy as np
import keras
import random
import argparse
# from torch.utils.tensorboard import SummaryWriter


class RadioMapDataset(Dataset):
    def __init__(self, full_radio_propagation_path, sampling_mask_path, transmitter_data_path, building_data_path):
        self.full_radio_propagation_path = full_radio_propagation_path
        self.sampling_mask_path = sampling_mask_path
        self.transmitter_data_path = transmitter_data_path
        self.building_data_path = building_data_path
        self.list_images = os.listdir(sampling_mask_path)

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        full_radio_propagation_image_path = self.full_radio_propagation_path + '/' + f"{'_'.join(self.list_images[idx].split('_')[0: 2])}.png"
        transmitter_target_image_path = self.transmitter_data_path + '/' + f"{'_'.join(self.list_images[idx].split('_')[0: 2])}.png"
        sampling_mask_path = self.sampling_mask_path + '/' + f"{self.list_images[idx]}"
        building_image_path = self.building_data_path + '/' + f"{self.list_images[idx].split('_')[0]}.png"

        full_radio_propagation_image = read_image(full_radio_propagation_image_path, ImageReadMode.GRAY).float() / 255.0
        sampling_mask = read_image(sampling_mask_path, ImageReadMode.GRAY).float()
        transmitter_target_image = read_image(transmitter_target_image_path, ImageReadMode.GRAY).float() / 255.0
        building_image = read_image(building_image_path, ImageReadMode.GRAY).float() / 255.0

        environment_with_known_sampling_pixels = (0.0 - building_image) + sampling_mask
        data = torch.cat([full_radio_propagation_image * sampling_mask, environment_with_known_sampling_pixels])
        return data, transmitter_target_image


class FocalLoss(nn.Module):
    def __init__(self, gamma=3):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = torch.nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pred = torch.exp(-1 * BCE) * targets + (1 - torch.exp(-1 * BCE)) * (1 - targets)
        z = (1 - pred) * targets
        z += (1 - targets) * pred
        z = z ** self.gamma
        alpha = targets * 0.75
        alpha += (1 - targets) * 0.25

        return torch.sum(z * BCE * alpha)


class Encoder(nn.Module):
    def __init__(self, enc_in, enc_out, n_dim, leaky_relu_alpha=0.3):
        super(Encoder, self).__init__()

        self.bn_0 = nn.GroupNorm(9, n_dim)
        self.bn_1 = nn.GroupNorm(9, n_dim)
        self.bn_2 = nn.GroupNorm(9, n_dim)
        self.bn_3 = nn.GroupNorm(9, n_dim)
        self.bn_4 = nn.GroupNorm(9, n_dim)
        self.bn_5 = nn.GroupNorm(9, n_dim)
        self.bn_6 = nn.GroupNorm(9, n_dim)
        self.bn_7 = nn.GroupNorm(9, n_dim)
        self.bn_8 = nn.GroupNorm(9, n_dim)
        self.bn_9 = nn.GroupNorm(2, enc_out)

        self.conv2d = nn.Conv2d(enc_in, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_1 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_2 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_3 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_4 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_5 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_6 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_7 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.conv2d_8 = nn.Conv2d(n_dim, n_dim, kernel_size=(3, 3), padding='same')
        self.mu = nn.Conv2d(n_dim, enc_out, kernel_size=(3, 3), padding='same')

        self.average_pooling2d = nn.AvgPool2d(kernel_size=(2, 2))
        self.average_pooling2d_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.average_pooling2d_2 = nn.AvgPool2d(kernel_size=(2, 2))

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.leaky_relu(self.bn_0(self.conv2d(x)))
        x = self.leaky_relu(self.bn_1(self.conv2d_1(x)))
        x = self.leaky_relu(self.bn_2(self.conv2d_2(x)))
        skip1 = x
        x = self.average_pooling2d(x)
        x = self.leaky_relu(self.bn_3(self.conv2d_3(x)))
        x = self.leaky_relu(self.bn_4(self.conv2d_4(x)))
        x = self.leaky_relu(self.bn_5(self.conv2d_5(x)))
        skip2 = x
        x = self.average_pooling2d_1(x)
        x = self.leaky_relu(self.bn_6(self.conv2d_6(x)))
        x = self.leaky_relu(self.bn_7(self.conv2d_7(x)))
        x = self.leaky_relu(self.bn_8(self.conv2d_8(x)))
        skip3 = x
        x = self.average_pooling2d_2(x)
        x = self.leaky_relu(self.bn_9(self.mu(x)))
        return x, skip1, skip2, skip3


class Decoder(nn.Module):
    def __init__(self, dec_in, dec_out, n_dim, leaky_relu_alpha=0.3):
        super(Decoder, self).__init__()

        self.bn_0 = nn.GroupNorm(2, dec_in)
        self.bn_1 = nn.GroupNorm(9, n_dim)
        self.bn_2 = nn.GroupNorm(9, n_dim)
        self.bn_3 = nn.GroupNorm(9, n_dim)
        self.bn_4 = nn.GroupNorm(9, n_dim)
        self.bn_5 = nn.GroupNorm(9, n_dim)
        self.bn_6 = nn.GroupNorm(9, n_dim)
        self.bn_7 = nn.GroupNorm(9, n_dim)
        self.bn_8 = nn.GroupNorm(9, n_dim)
        self.bn_9 = nn.GroupNorm(9, n_dim)

        self.conv2d_transpose = nn.ConvTranspose2d(dec_in, dec_in, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_1 = nn.ConvTranspose2d(dec_in + n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_2 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_3 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_4 = nn.ConvTranspose2d(2 * n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_5 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_6 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_7 = nn.ConvTranspose2d(2 * n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_8 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_transpose_9 = nn.ConvTranspose2d(n_dim, n_dim, kernel_size=(3, 3), stride=1, padding=1)

        self.up_sampling2d = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up_sampling2d_1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.up_sampling2d_2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.dropout = nn.Dropout()
        self.conv2d_output = nn.Conv2d(n_dim, dec_out, kernel_size=(1, 1))

        self.leaky_relu = torch.nn.LeakyReLU(negative_slope=leaky_relu_alpha)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, skip1, skip2, skip3):
        x = self.leaky_relu(self.bn_0(self.conv2d_transpose(x)))
        x = self.up_sampling2d(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.leaky_relu(self.bn_1(self.conv2d_transpose_1(x)))
        x = self.leaky_relu(self.bn_2(self.conv2d_transpose_2(x)))
        x = self.leaky_relu(self.bn_3(self.conv2d_transpose_3(x)))
        x = self.up_sampling2d_1(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.leaky_relu(self.bn_4(self.conv2d_transpose_4(x)))
        x = self.leaky_relu(self.bn_5(self.conv2d_transpose_5(x)))
        x = self.leaky_relu(self.bn_6(self.conv2d_transpose_6(x)))
        x = self.up_sampling2d_2(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.leaky_relu(self.bn_7(self.conv2d_transpose_7(x)))
        x = self.leaky_relu(self.bn_8(self.conv2d_transpose_8(x)))
        x = self.leaky_relu(self.bn_9(self.conv2d_transpose_9(x)))
        x = self.conv2d_output(x)
        return x


class LocNet(torch.nn.Module):
    def __init__(self, outpath, enc_in=2, enc_out=4, dec_out=1, n_dim=4, leaky_relu=0.3):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.outpath = outpath
        self.encoder = Encoder(enc_in, enc_out, n_dim, leaky_relu_alpha=leaky_relu)
        self.decoder = Decoder(enc_out, dec_out, n_dim, leaky_relu_alpha=leaky_relu)

    def forward(self, x):
        x, skip1, skip2, skip3 = self.encoder(x)
        x = self.decoder(x, skip1, skip2, skip3)
        return x

    def fit(self, num_epochs, train_loader, validation_loader, optimizer, loss_fn):
        self.to(self.device)
        save = np.inf
        # writer = SummaryWriter()
        for epoch in range(num_epochs):
            n_batches = len(train_loader)
            self.train()
            pBar = keras.utils.Progbar(target=n_batches, verbose=1)
            print(f"Starting epoch {epoch + 1}")
            running_loss = 0.0
            for t, data in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, targets = data
                inputs, targets = inputs.to(self.device), targets.squeeze().to(self.device)
                preds = self.forward(inputs)
                loss_ = loss_fn(preds.squeeze(), targets)
                loss_.backward()
                optimizer.step()
                pBar.update(
                    t,
                    values=[("loss", loss_.item())]
                )
                # running_loss += loss_.item()
            # writer.add_scalar("Loss/Train", running_loss / n_batches, epoch)

            with torch.no_grad():
                dist_loss = 0
                for inputs, targets in validation_loader:
                    self.eval()
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    preds = torch.nn.Sigmoid()(self.forward(inputs)).squeeze()
                    preds = torch.reshape(
                        preds,
                        (preds.shape[0], preds.shape[1] * preds.shape[2])
                    )

                    targets_ = targets.squeeze()
                    targets_ = torch.reshape(
                        targets_,
                        (targets_.shape[0], targets_.shape[1] * targets_.shape[2])
                    )

                    peak_points_indices = torch.argmax(preds, dim=1)
                    target_peak_points_indices = torch.argmax(targets_, dim=1)

                    dist = torch.sqrt(
                        ((peak_points_indices // 256) - (target_peak_points_indices // 256)) ** 2 +
                        ((peak_points_indices % 256) - (target_peak_points_indices % 256)) ** 2
                    ).detach().cpu()
                    dist_loss += torch.sum(dist)

            # writer.add_scalar("Val: Average Dist between prediction and transmitter", dist_loss / 40000, epoch)
            total_validation_images = len(validation_loader.dataset)
            pBar.update(
                n_batches,
                values=[("Val: Average Dist between prediction and transmitter", dist_loss / total_validation_images)]
            )
            if dist_loss < save:
                save = dist_loss
                # Name of the folder
                self.save_model(optimizer=optimizer, filename=f'LocNet_3_75_{epoch + 1}.pt')
        # writer.flush()
        # writer.close()

    def save_model(self, optimizer, filename):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
            self.outpath + '/' + filename)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-ftr",
        "--full_radio_propagation_path",
        dest="FULL_RADIO_PROPAGATION_PATH",
        required=True,
        help="Path to the dataset folder that contains all radio propagation images (include all: TRAIN + VAL + TEST)."
    )
    ap.add_argument(
        "-mtr",
        "--training_sampling_mask",
        dest="TRAINING_SAMPLING_MASK",
        required=True,
        help="Path to the training dataset folder that contains all binary sampling mask images."
    )
    ap.add_argument(
        "-t",
        "--transmitter_target",
        dest="TRANSMITTER_TARGET",
        required=True,
        help="Path to the dataset folder that contains all transmitter images. (include all: TRAIN + VAL + TEST)."
    )
    ap.add_argument(
        "-b",
        "--building_map",
        dest="BUILDING_MAP",
        required=True,
        help="Path to the training dataset folder that contains all building images. (include all: TRAIN + VAL + TEST)."
    )
    ap.add_argument(
        "-mval",
        "--validation_sampling_mask",
        dest="VALIDATION_SAMPLING_MASK",
        required=True,
        help="Path to the validation dataset folder that contains all binary sampling mask images."
    )
    ap.add_argument(
        "-o",
        "--path_2_saved_model",
        dest="PATH_2_SAVED_MODEL",
        required=True,
        help="Path to save the trained LocNet model."
    )
    args = vars(ap.parse_args())

    SEED = 0
    lr = 5e-4
    BATCH = 64
    EPOCHS = 100

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    Train = RadioMapDataset(
        full_radio_propagation_path=args['FULL_RADIO_PROPAGATION_PATH'],
        sampling_mask_path=args['TRAINING_SAMPLING_MASK'],
        transmitter_data_path=args['TRANSMITTER_TARGET'],
        building_data_path=args['BUILDING_MAP'],
    )
    Val = RadioMapDataset(
        full_radio_propagation_path=args['FULL_RADIO_PROPAGATION_PATH'],
        sampling_mask_path=args['VALIDATION_SAMPLING_MASK'],
        transmitter_data_path=args['TRANSMITTER_TARGET'],
        building_data_path=args['BUILDING_MAP'],
    )

    Train_Loader = DataLoader(Train, shuffle=True, batch_size=BATCH, drop_last=True)
    Val_Loader = DataLoader(Val, shuffle=True, batch_size=BATCH)
    model = LocNet(
        enc_in=2,
        enc_out=4,
        dec_out=1,
        n_dim=27,
        leaky_relu=0.3,
        outpath=args['PATH_2_SAVED_MODEL'],
    )
    criterion = torch.optim.AdamW(model.parameters(), lr=lr)
    model.fit(
        num_epochs=EPOCHS,
        train_loader=Train_Loader,
        validation_loader=Val_Loader,
        loss_fn=FocalLoss(gamma=3),
        optimizer=criterion
    )
