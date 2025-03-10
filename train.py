import os
import time
import numpy as np
import torch
import atexit
from argparse import ArgumentParser
from datetime import datetime

import dataloader as myloader  # 全尺寸
from Model.validation import NJD, dice_coef
from Model import losses
from Model.layers import SpatialTransformer_block

from Model import ESANet

def quit_operation(exp_folder):
    if not os.listdir(exp_folder):
        os.rmdir(exp_folder)

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)

def train(dataset,
          save_model_dir,
          load_model_dir,
          model_name,
          device,
          learning_rate,
          epochs,
          batch_size):

    if 'gpu' in device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device[-1]
        device = 'cuda'
        # torch.backends.cudnn.deterministic = True
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

    if dataset == 'OASIS':
        train_dir = '/Extra/yzy/Medical_Image_Registration/3D_brain_MRI/affine_img/'
        valid_dir = '/Extra/yzy/Medical_Image_Registration/3D_brain_MRI/affine_seg/'
        loader_train = myloader.torch_Dataloader_OASIS(train_dir, valid_dir, 'train', batch_size, (160, 192, 224))
        loader_validation = myloader.torch_Dataloader_OASIS(train_dir, valid_dir, 'test', batch_size, (160, 192, 224))
        data_train_value = 5000
        img_size = (160, 192, 224)
    elif dataset == 'IXI':
        train_dir = '/Extra/yzy/Medical_Image_Registration/3D_IXI_dataset/IXI_data/Train/'
        valid_dir = '/Extra/yzy/Medical_Image_Registration/3D_IXI_dataset/IXI_data/Test/'
        loader_train = myloader.torch_Dataloader_IXI(train_dir, valid_dir, 'train', batch_size)
        loader_validation = myloader.torch_Dataloader_IXI(train_dir, valid_dir, 'test', batch_size)
        data_train_value = 5000
        img_size = (160, 192, 224)
    elif dataset == 'Mind101':
        data_dir = '/Extra/yzy/Medical_Image_Registration/Mindboggle-101/data_used/'
        loader_train = myloader.torch_Dataloader_Mind101(data_dir, 'train', batch_size)
        loader_validation = myloader.torch_Dataloader_Mind101(data_dir, 'test', batch_size)
        img_size = (160, 192, 160)
        data_train_value = 800
        epochs = 30
    else:
        print('No such dataset:', dataset)
        exit(-1)

    data_train_sum = len(loader_train)
    data_value_sum = len(loader_validation)

    model = ESANet.ESANet()
    Losses = [losses.NCC(win=9).loss, losses.Grad().loss]
    Losses_weights = [1.0, 1.0]
    labels = [None, None]

    if load_model_dir != '':
        model.load_state_dict(torch.load(load_model_dir))
    model.to(device)

    stn = SpatialTransformer_block(mode='nearest')
    # stn = Local_SpatialTransformer_block(mode='nearest')
    stn.to(device)
    stn.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    timestamp = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    exp_folder = os.path.join(save_model_dir, model_name, dataset, timestamp)
    if os.path.exists(os.path.join(save_model_dir, model_name, dataset)):
        os.makedirs(exp_folder)
    else:
        os.makedirs(os.path.join(save_model_dir, model_name, dataset))
        os.makedirs(exp_folder)

    atexit.register(quit_operation, exp_folder)

    for epoch in range(epochs):
        model.train()
        step_t = 0
        adjust_learning_rate(optimizer, epoch, epochs, learning_rate)
        for pair_t, mi_t, fi_t, ml_t, fl_t in loader_train:
            step_t = step_t + 1
            mi_t = mi_t.unsqueeze(0).to(device)
            fi_t = fi_t.unsqueeze(0).to(device)
            labels[0] = fi_t
            input_t = [mi_t, fi_t]
            output_t = model(input_t)
            train_loss = 0
            loss_list = []
            for i, Loss in enumerate(Losses):
                if i is None:
                    continue
                else:
                    current_loss = Loss(labels[i], output_t[i]) * Losses_weights[i]
                train_loss += current_loss
                loss_list.append(current_loss.item())

            print('\ncurrent data:{} and {}, training {}/{}, training loss {:.6f} = {:.6f} + {:.6f}'.format(
                pair_t[0], pair_t[1], step_t, data_train_sum, train_loss, loss_list[0], loss_list[1]), end='')

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if step_t == data_train_sum or step_t % data_train_value == 0:
                model.eval()
                valid_warp_Dice = []
                valid_init_Dice = []
                valid_NJD = []
                valid_Time = []
                step_v = 0
                for pair_v, mi_v, fi_v, ml_v, fl_v in loader_validation:
                    step_v = step_v + 1
                    print('\rcurrent data:{} and {}, validation {}/{}'.format(pair_v[0], pair_v[1], step_v, data_value_sum), end='')
                    mi_v = mi_v.unsqueeze(0).to(device)
                    fi_v = fi_v.unsqueeze(0).to(device)
                    ml_v = ml_v.unsqueeze(0).to(device)
                    fl_v = fl_v.unsqueeze(0).to(device)

                    with torch.no_grad():
                        input_v = [mi_v, fi_v]
                        start_time = time.time()
                        output_v = model(input_v)
                        end_time = time.time()
                        warp_ml_v = stn(ml_v, output_v[1])
                    fl_v = fl_v.detach().cpu().numpy().squeeze()
                    ml_v = ml_v.detach().cpu().numpy().squeeze()
                    warp_ml_v = warp_ml_v.detach().cpu().numpy().squeeze()
                    flow_v = output_v[1].detach().cpu().permute(0, 2, 3, 4, 1).numpy().squeeze()


                    Dice_warp_val = dice_coef(warp_ml_v, fl_v, dataset)
                    valid_warp_Dice.append(Dice_warp_val)
                    Dice_init_val = dice_coef(ml_v, fl_v, dataset)
                    valid_init_Dice.append(Dice_init_val)
                    NJD_val = NJD(flow_v)
                    valid_NJD.append(NJD_val)
                    Time_val = end_time - start_time
                    valid_Time.append(Time_val)

                print('\nepoch{}: Time: {:.4f}, Dice_init: {:.6f}, Dice_warp: {:.6f}, NJD: {:.6f}'.format(epoch,
                    np.mean(valid_Time), np.mean(valid_init_Dice), np.mean(valid_warp_Dice), np.mean(valid_NJD)))
                torch.save(model.state_dict(), '{}/epoch{}_{}_Dice{:.4f}_NJD{:.4f}_Time{:.4f}.pt'.format(
                    os.path.join(save_model_dir, model_name, dataset, timestamp), epoch, step_t,
                    np.mean(valid_warp_Dice),
                    np.mean(valid_NJD),
                    np.mean(valid_Time)))
                # 继续训练
                model.train()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='Mind101')
    parser.add_argument("--save_model_dir", type=str, default='./saved_models')
    parser.add_argument("--load_model_dir", type=str, default='')
    parser.add_argument("--model_name", type=str, default='ESANet')
    parser.add_argument("--device", type=str, default='gpu0')
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    train(args.dataset,
          args.save_model_dir,
          args.load_model_dir,
          args.model_name,
          args.device,
          args.learning_rate,
          args.epochs,
          args.batch_size)
