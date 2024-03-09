import os
import logging
import numpy as np
import pandas as pd

# Env
from utils import *
from model_GAT import *
from options import parse_args
from test_model import test
from model_GAT import *


### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:0')
# Changed this from 1 to 15
num_splits = 1
results = []

### 2. Sets-Up Main Loop
for k in range(1, num_splits+1):
	print("*******************************************")
	print("************** SPLIT (%d/%d) **************" % (k, num_splits))
	print("*******************************************")

	train_features, train_labels, test_features, test_labels, adj_matrix = load_csv_data(k, opt)
	# Define pre-trained model path.
	load_path = opt.model_dir + '/split' + str(k) + '_' + opt.task + '_' + str(
				opt.lin_input_dim) + 'd_all_' + str(opt.num_epochs) + 'epochs.pt'
	# Loads a saved model checkpoint into memory, ready to use in your program.
	model_ckpt = torch.load(load_path, map_location=device)

	#### Loading Env
	# Extracts the model state dictionary from the model checkpoint. This can be loaded into a model using the load_state_dict() method.
	model_state_dict = model_ckpt['model_state_dict']
	# hasattr(target, attr) 用于判断对象中是否含有某个属性，有则返回true.
	if hasattr(model_state_dict, '_metadata'):
		del model_state_dict._metadata

	# Creates an instance of the GAT model. cuda() is used to move the model to the GPU.
	model = GAT(opt=opt, input_dim=opt.input_dim, omic_dim=opt.omic_dim, label_dim=opt.label_dim,
				dropout=opt.dropout, alpha=opt.alpha).cuda()

	### multiple GPU
	# model = torch.nn.DataParallel(model)
	# torch.backends.cudnn.benchmark = True

 	# For parallel GPU utilization: 
	if isinstance(model, torch.nn.DataParallel): model = model.module

	print('Loading the model from %s' % load_path)
	model.load_state_dict(model_state_dict)


	### 3.2 Test the model.
	loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test, test_features, te_fc_features = test(
		opt, model, test_features, test_labels, adj_matrix)
	GAT_te_features_labels = np.concatenate((test_features, te_fc_features, test_labels), axis=1)

	# print("model preds:", list(np.argmax(pred_test[3], axis=1)))
	# print("ground truth:", pred_test[4])
	# print(test_labels[:, 2])
	if not os.path.exists("./results/"+opt.task+"/GAT_features_"+str(opt.lin_input_dim)+"d_model/"):
		os.makedirs("./results/"+opt.task+"/GAT_features_"+str(opt.lin_input_dim)+"d_model/")
    

	pd.DataFrame(GAT_te_features_labels).to_csv(
	    "./results/"+opt.task+"/GAT_features_"+str(opt.lin_input_dim)+"d_model/split"+str(k)+"_"+ opt.which_layer+"_GAT_te_features.csv")

	if opt.task == 'surv':
		print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
		results.append(cindex_test)
	elif opt.task == 'grad':
		print("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		logging.info("[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f" % (loss_test, grad_acc_test))
		results.append(grad_acc_test)

	test_preds_labels = np.concatenate((pred_test[3], np.expand_dims(pred_test[4], axis=1)), axis=1)
	print(test_preds_labels.shape)

	if not os.path.exists("./results/" + opt.task + "/preds/"):
		os.makedirs("./results/" + opt.task + "/preds/")
	
	pd.DataFrame(test_preds_labels, columns=["class1", "class2", "class3", "pred_class"]).to_csv(
		"./results/" + opt.task + "/preds/split" + str(k) + "_" + opt.which_layer + "_test_preds_labels.csv")
	# pickle.dump(pred_test, open(os.path.join(opt.results_dir, opt.task,
	# 		'preds/split%d_pred_test_%dd_%s_%depochs.pkl' % (k, opt.lin_input_dim, opt.which_layer, opt.num_epochs)), 'wb'))

print('Split Results:', results)
print("Average:", np.array(results).mean())
