import torch
import os
import re

def load_bert_weights(model, pretrained_model):

    # load model weights from the pretrained model
    # specific to BERT model
    for model_name, model_param in model.named_parameters():

        # The linear layer of BLO linear unit that gets a gradient
        if 'linear_layer' in model_name:  # load the pretrained dense layer weights
            num_list = re.findall(r'\d+', model_name)
            if len(num_list) > 0:
                if 'weight' in model_name:
                    pretrained_name = 'pretrained_model.' + (model_name.split(str(num_list[0]))[0]).strip(
                        '.') + '[' + str(num_list[0]) + ']' + ((model_name.split(str(num_list[0]))[1])).replace(
                        'linear_layer.weight', 'weight')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')
                if 'bias' in model_name:
                    pretrained_name = 'pretrained_model.' + (model_name.split(str(num_list[0]))[0]).strip(
                        '.') + '[' + str(num_list[0]) + ']' + ((model_name.split(str(num_list[0]))[1])).replace(
                        'linear_layer.bias', 'bias')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')
            else:
                if 'weight' in model_name:
                    pretrained_name = 'pretrained_model.' + model_name.replace('linear_layer.weight', 'weight')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')
                if 'bias' in model_name:
                    pretrained_name = 'pretrained_model.' + model_name.replace('linear_layer.bias', 'bias')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')

        # the pretrained layer of the BLO linear unit that does not get a gradient
        elif 'pretrained_layer' in model_name:  # load the pretrained dense layer weights
            num_list = re.findall(r'\d+', model_name)
            if len(num_list) > 0:
                if 'weight' in model_name:
                    pretrained_name = 'pretrained_model.' + (model_name.split(str(num_list[0]))[0]).strip(
                        '.') + '[' + str(num_list[0]) + ']' + ((model_name.split(str(num_list[0]))[1])).replace(
                        'pretrained_layer.weight', 'weight')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')
                if 'bias' in model_name:
                    pretrained_name = 'pretrained_model.' + (model_name.split(str(num_list[0]))[0]).strip(
                        '.') + '[' + str(num_list[0]) + ']' + ((model_name.split(str(num_list[0]))[1])).replace(
                        'pretrained_layer.bias', 'bias')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')
            else:
                if 'weight' in model_name:
                    pretrained_name = 'pretrained_model.' + model_name.replace('pretrained_layer.weight', 'weight')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')
                if 'bias' in model_name:
                    pretrained_name = 'pretrained_model.' + model_name.replace('pretrained_layer.bias', 'bias')
                    model_param.data = eval(pretrained_name + '.data.detach().clone()')

        elif 'alpha' in model_name:  # alphas do not have a pretrained weight
            model_param.data = torch.normal(1.0, 0.005, size=model_param.shape)
            
        else:  # outside of the BLO linear function layer
            pretrained_name = 'pretrained_model.' + model_name
            num_list = re.findall(r'\d+', model_name)
            if len(num_list) > 0:
                pretrained_name = 'pretrained_model.' + (model_name.split(str(num_list[0]))[0]).strip(
                    '.') + '[' + str(
                    num_list[0]) + ']' + (model_name.split(str(num_list[0]))[1])
                model_param.data = eval(pretrained_name + '.data.detach().clone()')
            else:
                pretrained_name = 'pretrained_model.' + model_name
                model_param.data = eval(pretrained_name + '.data.detach().clone()')

    return model