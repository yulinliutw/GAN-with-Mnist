import torch
from load_data import load_data
from model import Generator,Discriminator
from loss_func import Discriminator_loss,Generator_loss
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

'''init setting'''
parser = argparse.ArgumentParser(description='Gan_train: Inference Parameters')
parser.add_argument('--noise_dim',
                    type=int,
                    default=50,
                    help='Determine the input size of the generator')

parser.add_argument('--epoch',
                    type=int,
                    default=20,
                    help='training epoch setting')
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.0002,
                    help='learning rate setting')

parser.add_argument('--save_weight_dir_gan',
                    default = './weight_train_g/',
                    help    = 'Path to folder of saving weight')
parser.add_argument('--save_weight_dir_d',
                    default = './weight_train_d/',
                    help    = 'Path to folder of saving weight')

parser.add_argument('--load_weight_dir_gan',
                    default = './weight_pretrain_g/checkpoint_ep19_itir_1199.pkl', 
                    help    = 'Path to folder of saving weight')
parser.add_argument('--load_weight_dir_d',
                    default = './weight_pretrain_d/checkpoint_ep19_itir_1199.pkl',
                    help    = 'Path to folder of saving weight')
parser.add_argument('--save_loss_figure_dir',
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

'''init some training variable'''
total_step = 0
fail_predict_real = 0
fail_predict_fake = 0
loss_real = 0
loss_fake = 0
loss_GAN = 0
loss_Dis = 0
try:
    with open(args.save_loss_figure_dir, 'rb') as file:
        total_loss =pickle.load(file)
except:  
    total_loss = {'fail_predict_real_his':[],'fail_predict_fake_his':[],'loss_GAN_his':[],'loss_Dis_his':[]} 

'''set the training gpu''' 
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)



'''load_data'''
Load_data = load_data()
train_data = Load_data.train()
val_data,test_data = Load_data.test()

'''init model'''
generator=Generator(args.noise_dim)
model_dict = generator.state_dict()
try:    
    pretrained_dict = torch.load(args.load_weight_dir_gan) #load pre train model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #load the layer only same with the target model
    model_dict.update(pretrained_dict)
    print('===================================')
    print('load pre_train weight successfully')
    print('===================================')
except: 
    print('===================================')
    print('       random init the weight      ')
    print('===================================')
generator.load_state_dict(model_dict)
generator.cuda()
generator.train()

discriminator = Discriminator()
model_dict = discriminator.state_dict()
try:    
    pretrained_dict = torch.load(args.load_weight_dir_d) #load pre train model
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict} #load the layer only same with the target model
    model_dict.update(pretrained_dict)
    print('===================================')
    print('load pre_train weight successfully')
    print('===================================')
except: 
    print('===================================')
    print('       random init the weight      ')
    print('===================================')
discriminator.load_state_dict(model_dict)
discriminator.cuda()
discriminator.train()

'''opt setting'''
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate*0.1)   # optimize all cnn parameters
optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)   # optimize all cnn parameters

'''setting about sample the batch from the dataset to train the discriminator'''
total_iter = train_data.__len__()
sample_iter = 0

'''main training code'''
for epoch in range(args.epoch):  
    loss_real = 0
    loss_fake = 0
    loss_GAN = 0
    loss_Dis = 0
    for step, (x, b_label) in enumerate(train_data):       
        generator.train()
        
        synthesis_batch = torch.rand(x.size()[0],args.noise_dim).cuda() #synthesis image has the same size with the batch image
        
        synthesis_img = generator(synthesis_batch)
        
        x_in = torch.tensor(x).cuda()
        prob_real = discriminator(x_in)  
        prob_fake = discriminator(synthesis_img)        
        if((step % 3) == 0):
            D_loss = Discriminator_loss(prob_real,prob_fake)
            optimizer_d.zero_grad()           # clear gradients for this training step
            D_loss.backward(retain_graph = True)# backpropagation, compute gradients
            optimizer_d.step()                # apply gradients            
       
        G_loss = Generator_loss(prob_fake)        
        optimizer_g.zero_grad()           # clear gradients for this training step
        G_loss.backward()               # backpropagation, compute gradients
        optimizer_g.step()                # apply gradients
        
        '''caculate the fail predict rate''' 
        fail_predict_real = (sum(prob_real<0.5).float()) / x_in.size()[0]
        fail_predict_fake = (sum(prob_fake>0.5).float()) / x_in.size()[0]
        
        loss_real = loss_real + fail_predict_real.cpu().numpy()
        loss_fake = loss_fake + fail_predict_fake.cpu().numpy()
        
        '''caculate the gan and dis loss'''
        loss_GAN = loss_GAN + G_loss.data.cpu().numpy()
        loss_Dis = loss_Dis + D_loss.data.cpu().numpy()
        
        
        '''save weight'''
        if(((step+1)%200) == 0):
            filename = 'checkpoint_ep'+str(epoch)+'_itir_'+str(step)+'.pkl'            
            filename = os.path.join(args.save_weight_dir_gan, filename)  
            torch.save(generator.state_dict(), filename) 
            
            filename = 'checkpoint_ep'+str(epoch)+'_itir_'+str(step)+'.pkl'
            filename = os.path.join(args.save_weight_dir_d, filename)  
            torch.save(discriminator.state_dict(), filename) 

    '''visualize current gan output result'''  
    generator.eval()      
    plt.title('current result from GAN')
    synthesis_img = torch.rand(2,args.noise_dim).cuda() #synthesis the image to test
    synthesis_result = generator(synthesis_img)
    plt.imshow(synthesis_result[0,0,:,:].data.cpu().numpy(), cmap='gray')
    plt.show()        
    '''caculate one epoch loss'''    
    loss_real = (loss_real / (step+1))*100
    loss_fake = (loss_fake / (step+1))*100
    total_loss['fail_predict_real_his'].append(loss_real)
    total_loss['fail_predict_fake_his'].append(loss_fake)
    
    loss_GAN = loss_GAN / (step+1)
    loss_Dis = loss_Dis / (step+1)
    total_loss['loss_GAN_his'].append(loss_GAN)
    total_loss['loss_Dis_his'].append(loss_Dis)
    
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
    
'''saving the loss data'''   
file = open(args.save_loss_figure_dir, 'wb')
pickle.dump(total_loss, file)
file.close()
        
