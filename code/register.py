"""
Liu, F., Liao, J., Zheng, J. et al.
GCN recommendation model based on the fusion of dynamic multiple-view latent interest topics.
Int. J. Mach. Learn. & Cyber. (2022).
"""
import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['ml-100k','ml-1m']:
    dataset = dataloader.Movielens(world.id,path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM(world.id)

print('===========config================')
pprint(world.config)
print("model id:",world.id)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'vs_lgn': model.VS_LightGCN
}

