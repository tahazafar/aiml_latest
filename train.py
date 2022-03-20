import argparse
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss
#import torch.optim as optim
import torch.cuda.amp as amp
import torch.backends.cudnn as cudnn
from dataset.cityscapes import cityscapesDataSet
import json
from dataset.gta5 import GTA5DataSet
from torch.utils import data
from model.discriminator import FCDiscriminator
from model.discriminator import LightFCDiscriminator
import torch.nn.functional as F
from torch.autograd import Variable
from utils import source_to_target_np
import pandas as pd
from torchsummary import summary
from flops_counter import get_model_complexity_info




IMG_MEAN = (104.00698793, 116.66876762, 122.67891434)
input_size = (1024,512) #'1280,720'



               
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_epochs, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
        
def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_epochs, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def val(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            data = data.cuda()
            label = label.long().cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            if args.loss == 'dice':
                label = reverse_one_hot(label)
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)
        
        precision = np.mean(precision_record)
        # miou = np.mean(per_class_iu(hist))
        miou_list = per_class_iu(hist)
        #IOU PER CLASSE
        
        
        # miou_dict, miou = cal_miou(miou_list, csv_path)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        # miou_str = ''
        # for key in miou_dict:
        #     miou_str += '{}:{},\n'.format(key, miou_dict[key])
        # print('mIoU for each class:')
        # print(miou_str)
        return precision, miou, miou_list

#make changes in train part and make changes to it for domain adaptation
def train_adv(args, model, optimizer, source_dataset, target_dataset, dataloader_val, model_D, optimizer_D, FDA=False, step=0):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    #writerD = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    
    model = model.cuda()
    model_D = model_D.cuda()
    cudnn.benchmark 

    scaler = amp.GradScaler()
    
    #if len(source_dataset)<len(target_dataset):
    #    source_dataloader_iterator = iter(source_dataset)
    
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    max_miou = 0
    #step = 0
    
    source_losses_D = []
    target_losses_D = []
    source_losses = []
    target_adv_losses = []
    dataframe=pd.DataFrame()
    
    targetloader_iter = iter(target_dataset)
    sourceloader_iter = iter(source_dataset)
    
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        lrD = poly_lr_scheduler(optimizer_D, args.learning_rate_D, iter=epoch, max_iter=args.num_epochs)

        
       
        #ricordarci di controllare la cardinalità di source e target. Se solo uguali va bene 
        #for i, (data, label) in enumerate(source_dataset) o indifferentemente for i, (data, label) in enumerate(target_dataset)
        #se le cardinalità sono diverse iterare uno dei due (quello minore), si può mettere il controllo con l'if
        iterations=max(len(source_dataset), len(target_dataset))
        
        tq = tqdm(total=iterations*args.batch_size)
        tq.set_description('epoch %d, lr %f, lr D %f' % (epoch, lr, lrD))
        #aggiungere loss
        model.train()
        model_D.train()
        optimizer.zero_grad()
        optimizer_D.zero_grad()
        adjust_learning_rate(args, optimizer, epoch)
        adjust_learning_rate_D(args, optimizer_D, epoch)
        
        
        class_sources = 0
        class_target = 1
        
        
        for i in range(iterations):

            try:
               sour_images, sour_labels = next(sourceloader_iter)
            except:
               source_dataloader_iterator = iter(source_dataset)
               sour_images, sour_labels = next(source_dataloader_iterator)

            try:
               target_images, _ = next(targetloader_iter)
            except:
               target_dataloader_iterator = iter(target_dataset)
               target_images, _ = next(target_dataloader_iterator)
            
                
            if FDA:
                B, C, H, W = sour_images.shape
                
                mean_img = torch.reshape(torch.from_numpy(np.array(IMG_MEAN, dtype=np.float32)), (1,3,1,1)).repeat(B,1,H,W)
                
              

                sour_target = source_to_target_np(sour_images, target_images)   
        
#------->  #inserire qui la media o la normalizzazione!!!
              
                sour_images = sour_target - mean_img     #capire questa cosa della media
                target_images = target_images - mean_img    #non normalizziamo mai le immagini?

        # source classification task (train G)
         
            for param in model_D.parameters():
                param.requires_grad = False    
            
            sour_images = Variable(sour_images).cuda()
            sour_labels = Variable(sour_labels).cuda()
            with amp.autocast():
                pred1, pred1_sup1, pred1_sup2 = model(sour_images)
                #pred1 = interp(pred1)  CAPIRE SE INSERIRLO O MENO, andrebbe inserito per la questione del size
                #pred1 = interp(pred1)
                source_loss = loss_func(pred1, sour_labels)+loss_func(pred1_sup1, sour_labels)+loss_func(pred1_sup2, sour_labels)
                source_loss=source_loss/iterations
            
            scaler.scale(source_loss).backward()
            source_losses.append(source_loss.item())
            
         # train G on target
            with amp.autocast():
                target_images = Variable(target_images).cuda()
                pred2, _, _=model(target_images)
                #pred_target1 = interp_target(pred2)  CAPIRE SE INSERIRLO O MENO
                #pred_target2 = interp_target(pred2)
                out1 = model_D(F.softmax(pred2, dim=1))
                #out2= model_D(F.softmax(pred2_sup1))
                #out3= model_D(F.softmax(pred2_sup2))
                target_adv_loss=bce_loss(out1, Variable(torch.FloatTensor(out1.data.size()).fill_(class_sources)).cuda())#+bce_loss(out2, Variable(torch.FloatTensor(out1.data.size()).fill_(class_sources)).cuda())+bce_loss(out3, Variable(torch.FloatTensor(out1.data.size()).fill_(class_sources)).cuda())
                loss = args.lambda_adv * target_adv_loss
                loss=loss/ifterations
                
            scaler.scale(loss).backward()
            target_adv_losses.append(target_adv_loss.item())
    
            
            for param in model_D.parameters():
                param.requires_grad = True
                
         # source discrimination task (domain '0') (train D)
            pred1=pred1.detach()
            with amp.autocast():
                pred1 = model_D(F.softmax(pred1, dim=1))
                source_loss_D = bce_loss(pred1, Variable(torch.FloatTensor(out1.data.size()).fill_(class_sources)).cuda())/2
                source_loss_D=source_loss_D/iterations
            scaler.scale(source_loss_D).backward()
            source_losses_D.append(source_loss_D.item()) 
            
         # target discrimination task (domain '1') (train D)
            pred2=pred2.detach()
            with amp.autocast():
                pred2 = model_D(F.softmax(pred2, dim=1))
                target_loss_D = bce_loss(pred2, Variable(torch.FloatTensor(out1.data.size()).fill_(class_target)).cuda())/2
                target_loss_D=target_loss_D/iterations
            scaler.scale(target_loss_D).backward() 
            target_losses_D.append(target_loss_D.item())         
            
            total_of_loss=target_loss_D+source_loss_D
            tq.update(args.batch_size)
            tq.set_postfix(loss_seg='%.6f' % source_loss, loss_adv='%.6f' % target_adv_loss, loss_D='%.6f' % total_of_loss )
            #tq.set_postfix(loss_adv='%.6f' % target_adv_loss)
            #tq.set_postfix(loss_D='%.6f' % total_of_loss)
            
            scaler.step(optimizer)
            scaler.step(optimizer_D)
            scaler.update()
 
        tq.close()
        #print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d},       loss_seg = {2:.3f}, loss_adv = {3:.3f}, loss_D = {4:.3f}'.format(
                epoch, args.num_epochs, np.mean(source_losses), np.mean(target_adv_losses), (np.mean(source_losses_D)+np.mean(target_losses_D))))
        
        writer.add_scalar('epoch/loss_seg_train', float(np.mean(source_losses)), epoch)
            
        if epoch >= args.num_epochs - 1:
            print('save model ...')
            if not os.path.isdir(args.save_model_path):
                torch.save(model.state_dict(),  os.path.join(args.save_model_path, 'last_model_step_'+str(step)+'.pth'))
            break 
        if (epoch+1) % args.validation_step == 0 or epoch == 0:
            precision, miou, miou_list = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                #max_miou_list = miou_list
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'best_mIoU_model_'+str(step)+'.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
            
            d=(pd.DataFrame(miou_list, description_label)).T
            d['mIoU']=miou
            d['accuracy']=precision
            dataframe=dataframe.append(d)
            
    
    
    #print((pd.DataFrame(max_miou_list, description_label).T).rename(index={0:'max_mIoU'}))
    dataframe.to_excel(os.path.join(args.save_model_path, 'table_'+str(step)+'.xlsx'))
    summary(model, (3, 1024,512))
    #flops = FlopCountAnalysis(model, torch.zeros((1, 3, 512, 1024)))
    #print('Total Flops='+flops.total())
    #sum_flops = flopth(model, in_size=[[3], [512],[1024]])
    flops, params = get_model_complexity_info(model, (3,1024,512))
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: (3,1024,512)\n'
      f'Flops: {flops}\nParams: {params}\n{split_line}')

    flops, params = get_model_complexity_info(model_D, (19,1024,512))
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: (19,1024,512)\n'
      f'Flops: {flops}\nParams: {params}\n{split_line}')

    #print(sum_flops)
    #model = model.eval().cuda()
    # print(summary(model, ( 3, 512, 1024)))
    #print(summary(model_D,( 19, 512, 1024)))
    #model = net()
    # Flops&params
     



    writer.close()    
        
def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=10, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="Cityscapes", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=1024, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')
    parser.add_argument('--source_data_root_path', type=str, default='/content/drive/MyDrive/BiSeNet/data/GTA5', help='source path of training data')
    parser.add_argument('--target_data_root_path', type=str, default='/content/drive/MyDrive/BiSeNet/data/cityscapes', help='target path of training data')
    parser.add_argument("--random_mirror", action="store_true", help="Whether to randomly mirror the inputs during the training")
    parser.add_argument('--use_d2', action='store_true', help='use second discriminator')
    parser.add_argument('--use_mclight', action='store_true', help='use multi class discriminator')
    parser.add_argument('--use_light', action='store_true', help='use multi class discriminator')
    parser.add_argument('--use_lightlight', action='store_true', help='use multi class discriminator')
    parser.add_argument('--use_weights', action='store_true', help='use weights for discriminator')
    parser.add_argument("--gan", type=str, default='Vanilla',help="choose the GAN objective")


    # for segmentation
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay optimizer')
    parser.add_argument('--seg_lr', type=float, default=2.5e-4, help='learning rate for scene parser')
    parser.add_argument('--power', type=float, default=0.9, help='poly learning rate decay power')
    parser.add_argument('--seg_norm', type=str, default=None, help='(instance_norm|None) if None batch_norm as default')
    parser.add_argument('--seg_loss', type=str, default=None, help='(focal|dice|None) if None it uses cross entropy')

    # for discriminators
    parser.add_argument('--lambda_adv', type=float, default=0.001, help="lambda adv for discriminator 0.001")
    parser.add_argument('--learning_rate_D', type=float, default=1e-4, help='learning rate for discriminators')
    parser.add_argument('--beta1_d', type=float, default=0.9, help='momentum term for adam')
    parser.add_argument('--beta2_d', type=float, default=0.99, help='momentum term for adam')
    parser.add_argument('--mean', type=int, default=(104.00698793, 116.66876762, 122.67891434), help='dataset mean')
    parser.add_argument('--name', type=str, default='workaround', help='name of the experiment. It decides where to store samples and models')    
    parser.add_argument('--ignore_index', default=255, type=int, help='ignore index')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_18_sgd', help='models are saved here')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    args = parser.parse_args(params)
    print(args.data)
        

    # Define here your dataloaders
    #dataset_train = CityscapesDataset(args.data, train=True, scale=(args.crop_height, args.crop_width))
    #dataset_val = CityscapesDataset(args.data, train=False, scale=False, mirror=False)

  
    #dataloader_train = DataLoader(
     #   dataset_train,
      #  batch_size=args.batch_size,
      #  shuffle=True,
      #  num_workers=args.num_workers,
      #  drop_last=True
   # )

    #dataloader_val = DataLoader(
     #   dataset_val,
      #  batch_size=1,
      #  shuffle=False,
     #   num_workers=args.num_workers,
     #   drop_last=False
    #)
    # Path, make train and val according to text files present 
    
    source_data_root_path = os.path.join(args.data, "GTA5") # /content/data/GTA5
    source_train_path = os.path.join(source_data_root_path, "train.txt") # /content/data/GTA5/train.txt

    target_data_root_path = os.path.join(args.data, "Cityscapes") # /content/data/Cityscapes
    target_root_path = os.path.join(target_data_root_path,  "train.txt")   # /content/data/Cityscapes/train.txt

    target_root_path_val = os.path.join(target_data_root_path,  "val.txt")   # /content/data/Cityscapes/val.txt

    info_path = os.path.join(source_data_root_path,  "info.json") # /content/data/GTA/info.json 
    info_json = json.load(open(info_path))
    global description_label
    description_label=info_json['label']
    img_mean = np.array(IMG_MEAN, dtype=np.float32) # stanard image mean
    

    # Datasets you are testing on target not on val(Define Dataloaders)

    source_dataset = GTA5DataSet(source_data_root_path,
                                 source_train_path,
                                 info_json,
                                 crop_size = input_size,
                                 scale = True,
                                 mirror = args.random_mirror,
                                 mean = img_mean)

    target_dataset = cityscapesDataSet(target_data_root_path,
                                       target_root_path,
                                       info_json,
                                       crop_size = input_size,
                                       scale=False,
                                       mirror=False,
                                       mean=img_mean)

    target_dataset_val = cityscapesDataSet(target_data_root_path,
                                           target_root_path_val,
                                           info_json,
                                           crop_size = input_size,
                                           scale = False,
                                           mirror = False,
                                           mean = img_mean)

    print("GTA: ", len(source_dataset))
    print("Cityscapes: ", len(source_dataset))
    img,label = source_dataset[0]
    print ("GTA image", img.shape )
    print ("GTA label", label.shape )
    img, _ = target_dataset[0]
    print ("Cityscapes image", img.shape )

    # Itersize
    assert len(source_dataset) == len(target_dataset)
    iter_size = len(source_dataset) // args.batch_size # the source and the target have the same len
    print("Iter_Size = ", iter_size)

    # Create DataLoaders
    trainloader = data.DataLoader(source_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers = args.num_workers,
                                  pin_memory=True)

    #trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(target_dataset,
                                   batch_size = args.batch_size,
                                   shuffle=True,
                                   num_workers = args.num_workers,
                                   pin_memory=True)

    #targetloader_iter = enumerate(targetloader)
    
    targetloaderVal = data.DataLoader(target_dataset_val,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers = args.num_workers,
                                      pin_memory = True)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None
    

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # load discriminator model    
    model_D = FCDiscriminator(args.num_classes)
    
    #built optimizer for discriminator
  
     
    optimizer_adv = torch.optim.Adam(model_D.parameters(), args.learning_rate_D)
   
        
        

    # train the adv models 
    train_adv(args, model, optimizer, trainloader, targetloader, targetloaderVal, model_D, optimizer_adv)
    
    model_D_Light = LightFCDiscriminator(args.num_classes)

    #built optimizer for discriminator (Light)
 
    optimizer_advL = torch.optim.Adam(model_D_Light.parameters(), args.learning_rate_D)

    
    #train the adv models with lightweight depthwise-separable convolutions
    #train_adv(args, model, optimizer, trainloader, targetloader, targetloaderVal, model_D_Light, optimizer_advL, step=1)
    
    #train the adv models with FDA
    #train_adv(args, model, optimizer, trainloader, targetloader, targetloaderVal, model_D, optimizer_adv, FDA=True, step=2)

if __name__ == '__main__':
    params = [
        '--num_epochs', '50',
        '--learning_rate', '2.5e-2',
        '--data', '/content/drive/MyDrive/BiSeNet/data', #Here is the images
        '--num_workers', '8',
        '--num_classes', '19',
        '--cuda', '0',
        '--batch_size', '4',
        '--save_model_path', './checkpoints_18_sgd',
        '--context_path', 'resnet18',  # set resnet18 or resnet101, only support resnet18 and resnet101
        '--optimizer', 'sgd',
        '--source_data_root_path', '/content/drive/MyDrive/BiSeNet/data/GTA5',
        '--target_data_root_path', '/content/drive/MyDrive/BiSeNet/data/cityscapes'

    ]
    main(params)



























