import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import platform
from argparse import ArgumentParser
from skimage.measure import compare_ssim as ssim
import math


# ISTA-Net+ implementation is taken from "ISTA-Net-PyTorch" GitHub repo at https://github.com/jianzhangcs/ISTA-Net-PyTorch.


parser = ArgumentParser(description='ISTA-Net-plus')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net-plus')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=10, help='from {10, 20, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='our_model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='gdrive/My Drive/training_data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='our_log', help='log directory')

args = parser.parse_args()


start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


nrtrain = 700
nval = 100   # number of validation images
batch_size = 2


Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask_matrix']


mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = torch.unsqueeze(mask_matrix, 2)
mask = torch.cat([mask, mask], 2)
mask = mask.to(device)


Training_data_Name = 'Training_BrainImages_256x256_100.mat'
Training_data = sio.loadmat('/%s/%s' % (args.data_dir, Training_data_Name))

Training_labels = Training_data['labels'][0:700,:,:]
Validation_labels = Training_data['labels'][700:800,:,:]


class FFT_Mask_ForBack(torch.nn.Module):
    def __init__(self):
        super(FFT_Mask_ForBack, self).__init__()

    def forward(self, x, mask):
        x_dim_0 = x.shape[0]
        x_dim_1 = x.shape[1]
        x_dim_2 = x.shape[2]
        x_dim_3 = x.shape[3]
        x = x.view(-1, x_dim_2, x_dim_3, 1)
        y = torch.zeros_like(x)
        z = torch.cat([x, y], 3)
        fftz = torch.fft(z, 2)
        z_hat = torch.ifft(fftz * mask, 2)
        x = z_hat[:, :, :, 0:1]
        x = x.view(x_dim_0, x_dim_1, x_dim_2, x_dim_3)
        return x


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))


        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))
        self.conv1_add_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_add_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_add_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_add_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
        x_input = x

        #output of linear operator
        x_D = F.conv2d(x_input, self.conv_D, padding=1)
        #forward transform
        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        #soft-thresholding
        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        #backward transform
        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_G = F.conv2d(x_backward, self.conv_G, padding=1)
        x_pred = x_input + torch.mul(0.5,x_G) #update

        #backward transform without thresholding
        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D #symmetry loss


        #2nd transform
        x_2 = F.conv2d(x_D, self.conv1_add_forward, padding=1)
        x_2 = F.relu(x_2)
        x_forward2 = F.conv2d(x_2, self.conv2_add_forward, padding=1)

        #soft-thresholding
        x_2 = torch.mul(torch.sign(x_forward2), F.relu(torch.abs(x_forward2) - self.soft_thr))

        #backward transform
        x_2 = F.conv2d(x_2, self.conv1_add_backward, padding=1)
        x_2 = F.relu(x_2)
        x_2_backward = F.conv2d(x_2, self.conv2_add_backward, padding=1)
        x_G_2 = F.conv2d(x_2_backward, self.conv_G, padding=1)
        x_pred = x_pred + torch.mul(0.5,x_G_2) #update

        #backward transform without thresholding
        x_2 = F.conv2d(x_forward2, self.conv1_add_backward, padding=1)
        x_2 = F.relu(x_2)
        x_D_est_2 = F.conv2d(x_2, self.conv2_add_backward, padding=1)
        symloss2 = x_D_est_2 - x_D #symmetry loss

        #Asymmetry loss
        x_asy = F.conv2d(x_forward, self.conv1_add_backward, padding=1)
        x_asy = F.relu(x_asy)
        x_D_est_asy = F.conv2d(x_asy, self.conv2_add_backward, padding=1)
        asymloss = x_D_est_asy - x_D #symmetry loss

        #Asymmetry loss2
        x_asy2 = F.conv2d(x_forward2, self.conv1_backward, padding=1)
        x_asy2 = torch.relu(x_asy2)
        x_D_est_asy2 = F.conv2d(x_asy2, self.conv2_backward, padding=1)
        asymloss2 = x_D_est_asy2 - x_D #symmetry loss
        
        # uncomment for inner loss computation
        
        # c1 = torch.reshape(self.conv1_forward, [16,144])
        # c1_add = torch.reshape(self.conv1_add_forward, [16,144])
        # c1 = F.normalize(c1,p=2,dim=0)
        # c1_add = F.normalize(c1_add,p=2,dim=0)
        # sym_mul = torch.matmul(torch.transpose(c1,0,1),c1_add)
        
        # c2 = torch.reshape(self.conv2_forward, [16,144])
        # c2_add = torch.reshape(self.conv2_add_forward, [16,144])
        # c2 = F.normalize(c2,p=2,dim=0)
        # c2_add = F.normalize(c2_add,p=2,dim=0)
        # sym_mul2 = torch.matmul(torch.transpose(c2,0,1),c2_add)
        
        # sm1 = torch.sum(torch.abs(sym_mul)+torch.abs(sym_mul2),0)
        # sym_inner = torch.sum(sm1,0)
        # return [x_pred, symloss, symloss2 ,sym_inner]
        return [x_pred, symloss, symloss2 , asymloss, asymloss2]

# Define ISTA-Net-plus
class ISTANetplus(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANetplus, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.fft_forback = FFT_Mask_ForBack()

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, PhiTb, mask):

        x = PhiTb

        layers_sym = []   
        layers_sym_2 = []
        layers_asym = []
        layers_asym_2 = []
        #layers_sym_inn = []
        for i in range(self.LayerNo):
            [x, layer_sym, layer_sym2, asymloss, asymloss2] = self.fcs[i](x, self.fft_forback, PhiTb, mask)
            layers_sym.append(layer_sym)
            layers_sym_2.append(layer_sym2)
            layers_asym.append(asymloss)
            layers_asym_2.append(asymloss2)
            #layers_sym_inn.append(sym_inner)

        x_final = x
        # return [x_final, layers_sym, layers_sym_2, layers_sym_inn]
        return [x_final, layers_sym, layers_sym_2, layers_asym, layers_asym_2]

model = ISTANetplus(layer_num)
model = nn.DataParallel(model)
model = model.to(device)



class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length

    def __getitem__(self, index):
        return torch.Tensor(self.data[index, :]).float()

    def __len__(self):
        return self.len


#validation set is created by taking 1/8 of the training data which corresponds to 100 images. 
if (platform.system() =="Windows"):
    rand_loader_training = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=0,
                             shuffle=True)
    rand_loader_validation = DataLoader(dataset=RandomDataset(Validation_labels, nval), batch_size=1, num_workers=0,
                             shuffle=True)
else:
    rand_loader_training = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size, num_workers=4,
                             shuffle=True)
    rand_loader_validation = DataLoader(dataset=RandomDataset(Validation_labels, nval), batch_size=1, num_workers=4,
                             shuffle=True)
                             
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model_dir = "./%s/MRI_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d" % (args.model_dir, layer_num, group_num, cs_ratio)

log_file_name = "./%s/Log_MRI_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d.txt" % (args.log_dir, layer_num, group_num, cs_ratio)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)


if start_epoch > 0:
    pre_model_dir = model_dir
    model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (pre_model_dir, start_epoch)))


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))



for epoch_i in range(start_epoch+1, end_epoch+1):
    indd_1 = 1
    for data in rand_loader_training:

        batch_x = data
        batch_x = batch_x.to(device)
        batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])

        PhiTb = FFT_Mask_ForBack()(batch_x, mask)

        [x_output, loss_layers_sym, loss_layers_sym_2, layers_asym, layers_asym_2] = model(PhiTb, mask)
        
        
        loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))

        loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
        for k in range(layer_num-1):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k+1], 2))

        loss_constraint2 = torch.mean(torch.pow(loss_layers_sym_2[0], 2))
        for k in range(layer_num-1):
            loss_constraint2 += torch.mean(torch.pow(loss_layers_sym_2[k+1], 2))

        loss_constraint3 = torch.pow(torch.mean(torch.pow(layers_asym[0], 2)),-1)
        #torch.mean(torch.div(1,torch.pow(layers_asym[0], 2))) #+ torch.pow(10,-7)
        for k in range(layer_num-1):
            #print('Loss Constraint 3 for layer i, i = ' + str(k) + ':' + str(torch.exp(-torch.mean(torch.pow(layers_asym[k+1], 2)))))
            loss_constraint3 += torch.pow(torch.mean(torch.pow(layers_asym[k+1], 2)),-1)

        loss_constraint4 = torch.pow(torch.mean(torch.pow(layers_asym_2[0], 2)),-1)
        for k in range(layer_num-1):
            #print('Loss Constraint 3 for layer i, i = ' + str(k) + ':' + str(torch.exp(-torch.mean(torch.pow(layers_asym[k+1], 2)))))
            loss_constraint4 += torch.pow(torch.mean(torch.pow(layers_asym_2[k+1], 2)),-1)

        gamma = torch.Tensor([0.05]).to(device)
        gamma2 = torch.Tensor([0.05]).to(device)
        gamma3 = torch.Tensor([0.001]).to(device)
        gamma4 = torch.Tensor([0.001]).to(device)

        # loss_all = loss_discrepancy
        loss_all = loss_discrepancy + torch.mul(gamma, loss_constraint) + torch.mul(gamma2, loss_constraint2) + torch.mul(gamma3, loss_constraint3)  + torch.mul(gamma4, loss_constraint4)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        indd_1 = indd_1+1

        if indd_1 % 100 == 0:
          indd_1=0
          output_data = "[%02d/%02d] Total Loss: %.5f, Discrepancy Loss: %.5f,  Constraint Loss: %.5f\n, Constraint Loss2: %.5f\n, Asymmetry Loss: %.5f\n, Asymmetry Loss 2 : %.5f\n" % (epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item(), loss_constraint, loss_constraint2, loss_constraint3, loss_constraint4)
          print(output_data)
          output_file = open(log_file_name, 'a')
          output_file.write(output_data)
          output_file.close()
        del x_output
        del loss_layers_sym
        del loss_layers_sym_2
        del PhiTb
        del batch_x

    if epoch_i % 5 == 0:
        torch.save(model.state_dict(), "./%s/net_params_%d.pkl" % (model_dir, epoch_i))  # save only the parameters
        img_no = 0
        PSNR_All = np.zeros([1, nval], dtype=np.float32)
        SSIM_All = np.zeros([1, nval], dtype=np.float32)

        Init_PSNR_All = np.zeros([1, nval], dtype=np.float32)
        Init_SSIM_All = np.zeros([1, nval], dtype=np.float32)
        for data in rand_loader_validation:
            batch_x = data
            batch_x = batch_x.to(device)
            batch_x = batch_x.view(batch_x.shape[0], 1, batch_x.shape[1], batch_x.shape[2])

            PhiTb = FFT_Mask_ForBack()(batch_x, mask)

            [x_output, loss_layers_sym, loss_layers_sym_2, layers_asym, layers_asym_2] = model(PhiTb, mask)

            initial_result = PhiTb.cpu().data.numpy().reshape(256, 256)

            Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)

            X_init = np.clip(initial_result, 0, 1).astype(np.float64)
            X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)
            np_arr = batch_x.cpu().numpy().squeeze()

            init_PSNR = psnr(X_init * 255, np_arr*255)
            init_SSIM = ssim(X_init * 255, np_arr*255, data_range=255)

            rec_PSNR = psnr(X_rec*255., np_arr*255)
            rec_SSIM = ssim(X_rec*255., np_arr*255, data_range=255)

            #print("Initial  PSNR is %.2f, Initial  SSIM is %.4f" % ( init_PSNR, init_SSIM))
            #print("Proposed PSNR is %.2f, Proposed SSIM is %.4f" % ( rec_PSNR, rec_SSIM))
            PSNR_All[0, img_no] = rec_PSNR
            SSIM_All[0, img_no] = rec_SSIM

            Init_PSNR_All[0, img_no] = init_PSNR
            Init_SSIM_All[0, img_no] = init_SSIM
            img_no = img_no+1

        print('\n')
        init_data =   "CS ratio is %d, Avg Initial  PSNR/SSIM is %.2f/%.4f" % (cs_ratio,  np.mean(Init_PSNR_All), np.mean(Init_SSIM_All))
        output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio,  np.mean(PSNR_All), np.mean(SSIM_All), epoch_i)
        print(init_data)
        print(output_data)
        output_file_name = "./%s/PSNR_SSIM_Results_MRI_CS_ISTA_Net_plus_layer_%d_group_%d_ratio_%d.txt" % (args.log_dir, layer_num, group_num, cs_ratio)

        output_file = open(output_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
    
        print("MRI CS Reconstruction End")  
        del x_output
        del a
        del b
        del PhiTb
        del batch_x         
        