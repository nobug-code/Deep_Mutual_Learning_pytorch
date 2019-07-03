import torch
import torch.nn as nn
import torchvision.models as models

class MUTUAL(object):

    def __init__(self, args):

        self.total_model_dic = {}
        #criterion means 기준
        #KL Loss
        self.kl_criterion = nn.KLDivLoss().cuda()
        self.total_loss_dic = {}
        self.kl_loss_dic = {}
        self.cross_loss_dic = {}
        self.model_output_dic = {}
        #CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.model_name = args.model_name
        self.lr = args.lr
        self.epochs = args.epochs
        self.scope = args.model_number
        if(self.model_name == 'mobilenet'):
            for i in range(self.scope):
                self.total_model_dic["{0}".format(i)] = models.MobileNetV2()
                self.cross_loss_dic["{0}".format(i)] = 0
                self.model_output_dic["{0}".format(i)] = 0
                self.kl_loss_dic["{0}".format(i)] = 0
                self.total_loss_dic["{0}".format(i)] = 0

        print(self.total_model_dic.keys())
        self.optimizer = torch.optim.Adam(self.total_model_dic[str(0)].parameters(), lr= self.lr, betas=(0.5, 0.999))

    def train(self, train_loader):
        #Declaration Model
        for model in self.total_model_dic:
            self.total_model_dic[model].train()

        for epoch in range(self.epochs):

            for k, (input, target) in enumerate(train_loader):
                target = target.cuda()
                input = input.cuda()
                #Calculate each model
                for model_num, _model in enumerate(self.total_model_dic):
                    model = self.total_model_dic[_model]
                    output = model(input)
                    loss = self.criterion(output, target)
                    self.cross_loss_dic[str(model_num)] = loss
                    self.model_output_dic[str(model_num)] = output

                #Calculate KL loss each model
                for i, model_first in enumerate(self.model_output_dic):
                    first_output = self.model_output_dic[model_first]
                    for j, model_second in enumerate(self.model_output_list):
                        #Not Calculate self model
                        if(i == j):
                            continue
                        second_output = self.model_output_dic[model_second]
                        self.kl_loss_dic[str(i)] += self.kl_loss_compute(first_output, second_output)
                #total loss
                for i, key in enumerate(self.total_loss_dic):
                    self.total_loss_dic[str(i)] = self.cross_loss_dic[str(i)] + self.kl_loss_dic[str(i)]/self.scope

                for model in self.total_model_dic:
                    self.optimizer.zero_grad()
                    self.total_loss_dic[str(model)].backward()
                    self.optimizer.step()


    def kl_loss_compute(self, logits1, logits2):
        kl_loss = self.kl_criterion(logits1, logits2)
        return kl_loss











