import torch
import torch.nn as nn
from pytorch_nndct.apis import torch_quantizer # You should active the Vitis-AI enviroment on ubuntu first.
from model import model_class # Change here to your model.py (You should import all of your model subclass.)

if  __name__=='__main__':
	'''
	In the begining, you should put weight.pth and model.py in 
	workspace, and their path should be same to the path you torch.save.
	After this work, you should get a 
	.xmodel file in quantize_result folder.
	Then you should keep going to compile the .xmodel file.
	'''
	quant_mode = 'calib'
	model = torch.load('./resnet_with_squeezenet_and_unet.pth')
	batch_size = 1
	input = torch.randn([batch_size, 3, 400, 400])
	quantizer = torch_quantizer(quant_mode, model, (input))
	quant_model = quantizer.quant_model
	
	output = quant_model(input)

	quantizer.export_quant_config()
	
	quant_mode = 'test'
	quantizer = torch_quantizer(quant_mode, model, (input))
	quant_model = quantizer.quant_model
	
	output = quant_model(input)
	quantizer.export_xmodel(deploy_check=False)