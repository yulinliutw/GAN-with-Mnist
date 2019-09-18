import torch
from model import Generator
import argparse
import matplotlib.pyplot as plt
import os
import pickle

'''init setting'''
parser = argparse.ArgumentParser(description='Gan_train: Inference Parameters')
parser.add_argument('--noise_dim',
                    type=int,
                    default=50,
                    help='Determine the input size of the generator, it must same with the training setting')
parser.add_argument('--load_weight_dir_gan',
                    default = './weight_pretrain_g/checkpoint_ep19_itir_1199.pkl', 
                    help    = 'Path to folder of saving weight')
parser.add_argument('--load_loss_figure_dir',
                    default = './loss_figure.pickle',
                    help    = 'Path to folder of saving loss figure')
parser.add_argument('--gpuid',
                    default = 0,
                    type    = int,
                    help    = 'GPU device ids (CUDA_VISIBLE_DEVICES)')

'''gobal setting'''
global args
args = parser.parse_args()
torch.manual_seed(0)


'''set the training gpu''' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)


'''init model'''
generator=Generator(args.noise_dim)
model_dict = generator.state_dict()
   
pretrained_dict = torch.load(args.load_weight_dir_gan) #load pre train model
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #load the layer only same with the target model
model_dict.update(pretrained_dict)
print('===================================')
print('load pre_train weight successfully')
print('===================================')

generator.load_state_dict(model_dict)
generator.cuda()
generator.eval()

print("         show the final training result")
plt.figure()
synthesis_img = torch.rand(25,args.noise_dim).cuda() #synthesis the image to test
synthesis_result = generator(synthesis_img)
for result_idx in range(25):
    plt.subplot(5,5,result_idx+1)
    plt.imshow(synthesis_result[result_idx,0,:,:].data.cpu().numpy(), cmap='gray')
    plt.axis('off')
plt.show() 

try:
    with open(args.load_loss_figure_dir, 'rb') as file:
        total_loss =pickle.load(file)
        '''draw the loss figure'''
        '''the prediction accuracy(%) of the discriminator'''  
        plt.plot(total_loss['fail_predict_real_his'], label='loss_real_image_predict') #loss_real_image_predict: how many % prediction is fail when input the real image
        plt.plot(total_loss['fail_predict_fake_his'], label='loss_fake_image_predict') #loss_fake_image_predict: how many % prediction is fail when input the fake image
        plt.title('real/fake predict loss(%)')
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')    
        plt.show()
        
        '''the loss curve from the loss function'''
        plt.plot(total_loss['loss_GAN_his'], label='loss_GAN')
        plt.plot(total_loss['loss_Dis_his'], label='loss_Dis')
        plt.title('GAN/Dis loss')
        plt.legend(loc='best')
        plt.xlabel('Steps')
        plt.ylabel('Loss')    
        plt.show()
except:  
    print("fail to open the loss figure")
