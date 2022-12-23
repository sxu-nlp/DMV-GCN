"""
Liu, F., Liao, J., Zheng, J. et al.
GCN recommendation model based on the fusion of dynamic multiple-view latent interest topics.
Int. J. Mach. Learn. & Cyber. (2022).
"""
import world
import utils
from world import cprint
import time
import Procedure
import os

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

if __name__ == '__main__':
    id = world.id
    Recmodel = register.MODELS[world.model_name](world.config, dataset, id)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    filepath = "../result/" + str(world.dataset) +'/'+ str(Recmodel.n_layers) + "layers"+ '/model'+str(id)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filename = filepath + "/VS_LightGCN" + str(Recmodel.n_layers)+".txt"

    save = world.save
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, filename,epoch, world.config['multicore'])

        if save == 1 and epoch == world.stop:
            Recmodel.saveEmbedding('../data/' + str(world.dataset) + '/model' + str(id)+"/layer"+str(Recmodel.n_layers))
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr)
        print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
