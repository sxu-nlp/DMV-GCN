"""
Liu, F., Liao, J., Zheng, J. et al.
GCN recommendation model based on the fusion of dynamic multiple-view latent interest topics.
Int. J. Mach. Learn. & Cyber. (2022).
"""
import os
import torch
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()


config = {}
all_dataset = ['lastfm', 'ml-1m','ml-100k']
all_models  = ['mf', 'vs_lgn']
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['A_split'] = False
config['bigdata'] = False
config['alpha']= args.alpha

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

id=args.graphID
dataset = args.dataset
model_name = args.model
save = args.save

if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")


TRAIN_epochs = args.epochs
stop = args.stop
topks = eval(args.topks)
t = eval(args.t)
beta = eval(args.beta)
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

