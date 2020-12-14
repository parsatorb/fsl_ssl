# This code is modified from https://github.com/jakesnell/prototypical-networks

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
from model_resnet import *
from itertools import cycle
from utils import Memory

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, jigsaw=False, lbda=0.0, rotation=False, tracking=False, use_bn=True, pretrain=False, image_loader=None, len_dataset=None):
        super(ProtoNet, self).__init__(model_func,  n_way, n_support, use_bn, pretrain)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.len_dataset = len_dataset
        self.cuda()
        self.memory = Memory(size=len_dataset, weight=0.5, device='cuda')
        self.memory.initialize(self.feature, image_loader)
        
        self.jigsaw = jigsaw
        self.rotation = rotation
        self.lbda = lbda
        self.global_count = 0
        
        self.indx = 0
       

        if self.jigsaw:
            
            
            
            self.projection_transformed_features = nn.Linear(512*9, 512) ### Self-supervision branch 

            #self.fc6 = nn.Sequential()
            #self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            #self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            #self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            #self.fc7 = nn.Sequential()
            #self.fc7.add_module('fc7',nn.Linear(9*512,4096))#for resnet
            #self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            #self.fc7.add_module('drop7',nn.Dropout(p=0.5))
            
            #self.classifier = nn.Sequential()
            #self.classifier.add_module('fc8',nn.Linear(4096, 35))
            
        if self.rotation:
            self.fc6 = nn.Sequential()
            self.fc6.add_module('fc6_s1',nn.Linear(512, 512))#for resnet
            self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
            self.fc6.add_module('drop6_s1',nn.Dropout(p=0.5))

            self.fc7 = nn.Sequential()
            self.fc7.add_module('fc7',nn.Linear(512,128))#for resnet
            self.fc7.add_module('relu7',nn.ReLU(inplace=True))
            self.fc7.add_module('drop7',nn.Dropout(p=0.5))

            self.classifier_rotation = nn.Sequential()
            self.classifier_rotation.add_module('fc8',nn.Linear(128, 4))
        

    def train_loop(self, epoch, train_loader, optimizer, writer, base_loader_u=None):
        
        print_freq = 10
        avg_loss= 0
        avg_loss_proto= 0
        avg_loss_jigsaw= 0
        avg_loss_rotation= 0
         
        if base_loader_u is not None:
            
            for i,inputs in enumerate(zip(train_loader,cycle(base_loader_u))):
                self.global_count += 1
                x = inputs[0][0]
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way  = x.size(0)
                optimizer.zero_grad()
                loss_proto, acc = self.set_forward_loss(x)
                if self.jigsaw:
                    #loss_jigsaw, acc_jigsaw = self.set_forward_loss_unlabel(inputs[1][2], inputs[1][3],x)# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                    loss_jigsaw = self.set_forward_loss_unlabel(inputs[1][2], inputs[1][3],x)# torch.Size([5, 21, 9, 3, 64, 64]), torch.Size([5, 21])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_jigsaw', float(loss_jigsaw.data.item()), self.global_count)
                elif self.rotation:
                    loss_rotation, acc_rotation = self.set_forward_loss_unlabel(inputs[1][2], inputs[1][3],x)# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_rotation', float(loss_rotation.data.item()), self.global_count)
                else:
                    loss = loss_proto
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss+loss.data
                writer.add_scalar('train/loss', float(loss.data.item()), self.global_count)

                if self.jigsaw:
                    avg_loss_proto += loss_proto.data
                    avg_loss_jigsaw += loss_jigsaw.data
                    writer.add_scalar('train/acc_proto', acc, self.global_count)
                    writer.add_scalar('train/acc_jigsaw', acc_jigsaw, self.global_count)
                elif self.rotation:
                    avg_loss_proto += loss_proto.data
                    avg_loss_rotation += loss_rotation.data
                    writer.add_scalar('train/acc_proto', acc, self.global_count)
                    writer.add_scalar('train/acc_rotation', acc_rotation, self.global_count)

                if (i+1) % print_freq==0:
                    if self.jigsaw:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Jigsaw {:f}'.\
                            format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_jigsaw/float(i+1)))
                    elif self.rotation:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Rotation {:f}'.\
                            format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_rotation/float(i+1)))
                    else:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(train_loader), avg_loss/float(i+1)))
        else:
            #### This branch is used 
            self.memory.update_weighted_count()
            self.indx = 0
            for i, inputs in enumerate(train_loader):
                
                self.global_count += 1
                x = inputs[0] ### [5,21,3,224,224]
                self.n_query = x.size(1) - self.n_support
                if self.change_way:
                    self.n_way  = x.size(0)
                optimizer.zero_grad()
                loss_proto, acc = self.set_forward_loss(x)
                if self.jigsaw:
                    #  print(x.size(), inputs[2].size(), inputs[3].size())
                    loss_jigsaw  = self.set_forward_loss_unlabel(x, inputs[2], inputs[3])# torch.Size([5, 21, 9, 3, 64, 64]), torch.Size([5, 21])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_jigsaw
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_jigsaw', float(loss_jigsaw.data.item()), self.global_count)
                elif self.rotation:
                    loss_rotation, acc_rotation = self.set_forward_loss_unlabel(inputs[2], inputs[3],x)# torch.Size([5, 21, 9, 3, 75, 75]), torch.Size([5, 21])
                    loss = (1.0-self.lbda) * loss_proto + self.lbda * loss_rotation
                    writer.add_scalar('train/loss_proto', float(loss_proto.data.item()), self.global_count)
                    writer.add_scalar('train/loss_rotation', float(loss_rotation.data.item()), self.global_count)
                else:
                    loss = loss_proto
                loss.backward()
                optimizer.step()
                avg_loss = avg_loss+loss.item()
                writer.add_scalar('train/loss', float(loss.data.item()), self.global_count)

                if self.jigsaw:
                    avg_loss_proto += loss_proto.data
                    avg_loss_jigsaw += loss_jigsaw.data
                    writer.add_scalar('train/acc_proto', acc, self.global_count)
                    # writer.add_scalar('train/acc_jigsaw', acc_jigsaw, self.global_count)
                elif self.rotation:
                    avg_loss_proto += loss_proto.data
                    avg_loss_rotation += loss_rotation.data
                    writer.add_scalar('train/acc_proto', acc, self.global_count)
                    writer.add_scalar('train/acc_rotation', acc_rotation, self.global_count)

                if (i+1) % print_freq==0:
                    #print(optimizer.state_dict()['param_groups'][0]['lr'])
                    if self.jigsaw:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Jigsaw {:f}'.\
                            format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_jigsaw/float(i+1)))
                    elif self.rotation:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Loss Proto {:f} | Loss Rotation {:f}'.\
                            format(epoch, i+1, len(train_loader), avg_loss/float(i+1), avg_loss_proto/float(i+1), avg_loss_rotation/float(i+1)))
                    else:
                        print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(train_loader), avg_loss/float(i+1)))
                self.indx += 105
                

    def test_loop(self, test_loader, record = None):
        # breakpoint()
        correct =0
        count = 0
        acc_all = []
        acc_all_jigsaw = []
        acc_all_rotation = []

        iter_num = len(test_loader)
        for i, inputs in enumerate(test_loader):
            x = inputs[0]
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)

            if self.jigsaw:
                # correct_this, correct_this_jigsaw, count_this, count_this_jigsaw = self.correct(x, inputs[2], inputs[3])
                correct_this,  count_this = self.correct(x)
            elif self.rotation:
                correct_this, correct_this_rotation, count_this, count_this_rotation = self.correct(x, inputs[2], inputs[3])
            else:
                correct_this, count_this = self.correct(x)
            acc_all.append(correct_this/ count_this*100)
            # if self.jigsaw:
            #    acc_all_jigsaw.append(correct_this_jigsaw/ count_this_jigsaw*100)
            # elif self.rotation:
            #    acc_all_rotation.append(correct_this_rotation/ count_this_rotation*100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Protonet Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if self.jigsaw:
            # acc_all_jigsaw  = np.asarray(acc_all_jigsaw)
            # acc_mean_jigsaw = np.mean(acc_all_jigsaw)
            # acc_std_jigsaw  = np.std(acc_all_jigsaw)
            # print('%d Test Jigsaw Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_jigsaw, 1.96* acc_std_jigsaw/np.sqrt(iter_num)))
            #return acc_mean, acc_mean_jigsaw
            return acc_mean
        elif self.rotation:
            acc_all_rotation  = np.asarray(acc_all_rotation)
            acc_mean_rotation = np.mean(acc_all_rotation)
            acc_std_rotation  = np.std(acc_all_rotation)
            print('%d Test Rotation Acc = %4.2f%% +- %4.2f%%' %(iter_num,  acc_mean_rotation, 1.96* acc_std_rotation/np.sqrt(iter_num)))
            return acc_mean, acc_mean_rotation
        else:
            return acc_mean

    def correct(self, x, patches=None, patches_label=None):
        
        scores = self.set_forward(x)
        #if self.jigsaw:
        #    x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
        #elif self.rotation:
        #    x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)

        return float(top1_correct), len(y_query)

        #if self.jigsaw:
        #    pred = torch.max(x_,1)
        #    top1_correct_jigsaw = torch.sum(pred[1] == y_)
        #    return float(top1_correct), float(top1_correct_jigsaw), len(y_query), len(y_)
        #elif self.rotation:
        #    pred = torch.max(x_,1)
        #    top1_correct_rotation = torch.sum(pred[1] == y_)
        #    return float(top1_correct), float(top1_correct_rotation), len(y_query), len(y_)
        #else:
        #    return float(top1_correct), len(y_query)

    def set_forward(self,x,is_feature = False):
        
        z_support, z_query  = self.parse_feature(x,is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1 ).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto)
        scores = -dists
        return scores

    def set_forward_unlabel(self, patches=None, patches_label=None):
        
        # print(patches.size())
        if len(patches.size()) == 6:
            patches_support = patches[:, :self.n_support] ###support pathces
            Way,S,T,C,H,W = patches_support.size()#torch.Size([5, 5, 9, 3, 64, 64]) ###new 
            B = Way*S
        elif len(patches.size()) == 5:
            B,T,C,H,W = patches.size()#torch.Size([5, 15, 9, 3, 75, 75])
        if self.jigsaw:
            patches_support = patches_support.reshape(B*T,C,H,W).cuda()#torch.Size([225, 3, 64, 64]) ###new
            if self.dual_cbam:
                patch_feat = self.feature(patches_support, jigsaw=True)#torch.Size([225, 512])
            else:
                patch_feat = self.feature(patches_support)#torch.Size([225, 512])

            x_ = patch_feat.view(B,T,-1)### [25,9,512]
            x_ = x_[:,torch.randperm(x_.size()[1])]
            x_=x_.view(B,-1) #[25,4608] ###new
            v_t=self.projection_transformed_features(x_) ### [25,512]
            v_t=v_t.view(self.n_way,self.n_way,-1) ### [5,5,512]
            
            #x_ = x_.transpose(0,1)#torch.Size([9, 75, 512])

            #x_list = []
            #for i in range(9):
            #    z = self.fc6(x_[i])#torch.Size([75, 512])
            #    z = z.view([B,1,-1])#torch.Size([75, 1, 512])
            #    x_list.append(z)

            #x_ = torch.cat(x_list,1)#torch.Size([75, 9, 512])
            #x_ = (x_.view(B,-1))#torch.Size([105, 9*512])
            #x_=  self.projection_transformed_features(x_) # [105,512]
            #x_ = self.classifier(x_)

            #y_ = patches_label.view(-1).cuda()

            return v_t
        elif self.rotation:
            patches = patches.view(B*T,C,H,W).cuda()
            x_ = self.feature(patches)#torch.Size([64, 512, 1, 1])
            x_ = x_.squeeze()
            x_ = self.fc6(x_)
            x_ = self.fc7(x_)#64,128
            x_ = self.classifier_rotation(x_)#64,4
            pred = torch.max(x_,1)
            y_ = patches_label.view(-1).cuda()
            return x_, y_


    def set_forward_loss(self, x):
        
        y_query = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))
        
        scores = self.set_forward(x)
        
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        acc = np.sum(topk_ind[:,0] == y_query.numpy())/len(y_query.numpy())
        y_query = Variable(y_query.cuda())
        

        return self.loss_fn(scores, y_query), acc



    def contrastive_loss(self, original_features, patch_features, negative_nb, index): ###new
        loss=0
        # rng = np.random.default_rng()
        # print(z_support.size())
        
        #negatives = torch.empty(5,20,512)
        #negatives[0] = torch.cat((z_support[1], z_support[2], z_support[3], z_support[4]))
        #negatives[1] = torch.cat((z_support[0], z_support[2], z_support[3], z_support[4]))
        #negatives[2] = torch.cat((z_support[0], z_support[1], z_support[3], z_support[4]))
        #negatives[3] = torch.cat((z_support[0], z_support[1], z_support[2], z_support[4]))
        #negatives[4] = torch.cat((z_support[0], z_support[1], z_support[2], z_support[3]))
        
        for i in range(original_features.shape[0]):

            temp = 0.07
            cos = torch.nn.CosineSimilarity()
            criterion = torch.nn.CrossEntropyLoss()

            ### Obtaining negative images N=20     
            
            # Index=np.array(range(0,original_features.shape[0])) ### [,25]
            # Index=np.delete(Index,i) ### [,24]
            # numbers = rng.choice(24, size=negative_nb, replace=False) # [1,20]
            
            #for j in range(negative_nb):    
               # if(j==1):
                  #  negative=z_support[Index[numbers[j]]] 
               # else:
                  #  negative=torch.cat((negative,z_support[Index[numbers[j]]])) 

	        ### Negative should have a size of [20,512]

            # negative = negatives[i//5]
            
            
            negative = self.memory.return_random(size=negative_nb, index=[index[i]])
            negative = torch.Tensor(negative).to('cuda').detach()

            image_to_modification_similarity = cos(original_features[None, i, :], patch_features[None, i, :])/temp ### [,1]
            matrix_of_similarity = cos(patch_features[None, i, :], negative) / temp ### [,20]

            similarities = torch.cat((image_to_modification_similarity, matrix_of_similarity))
            loss += criterion(similarities[None, :], torch.tensor([0]).to('cuda'))
            
        
        return loss / original_features.shape[0]




    def set_forward_loss_unlabel(self, x, patches=None, patches_label=None): ###new
        
        if self.jigsaw:

            #x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            #pred = torch.max(x_,1)
            #acc_jigsaw = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)
            #x = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])

            v_t=self.set_forward_unlabel(patches=patches,patches_label=patches_label) ###new [5,5,512]
            v_t=v_t.view(25,-1) ###new [25,512]
 
            z_support, z_query  = self.parse_feature(x,is_feature=False)  ###new
            v= z_support  ###new [5,5,512]
            # print(v[0][0])
            v = v.reshape(-1, 512)
            # print(v[0])
            # print(v.size())
            # v=v.view(25,-1) ###new [25,512]
            
            indxs = [i + self.indx for i in [0, 1, 2, 3, 4, 21, 22, 23, 24, 25, 42, 43, 44, 45, 46, 63, 64, 65, 66, 67, 84, 85, 86, 87, 88]]
            representations = self.memory.return_representations(indxs).to('cuda').detach()
            
            negative_nb=2000
            loss_weight=0.5
            loss_1 = self.contrastive_loss(representations, v_t, negative_nb, indxs)
            loss_2 = self.contrastive_loss(representations, v, negative_nb, indxs)
            loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2
            
            self.memory.update(indxs, v.detach().cpu().numpy())

        elif self.rotation:
            x_, y_ = self.set_forward_unlabel(patches=patches,patches_label=patches_label)
            pred = torch.max(x_,1)
            acc_rotation = torch.sum(pred[1] == y_).cpu().numpy()*1.0/len(y_)

        if self.jigsaw:
            return loss
            
        elif self.rotation:
            return self.loss_fn(x_,y_), acc_rotation


    def parse_feature(self,x,is_feature):
        
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all       = self.feature(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query



def euclidean_dist( x, y):
    
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
