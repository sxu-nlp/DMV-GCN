"""
Liu, F., Liao, J., Zheng, J. et al.
GCN recommendation model based on the fusion of dynamic multiple-view latent interest topics.
Int. J. Mach. Learn. & Cyber. (2022).
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F
import pickle
from world import cprint


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class PureMF(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(PureMF, self).__init__()
        self.num_users = dataset.n_users
        self.num_items = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")

    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)

    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb = self.embedding_item(pos.long())
        neg_emb = self.embedding_item(neg.long())
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1 / 2) * (users_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / float(len(users))
        return loss, reg_loss

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb * items_emb, dim=1)
        return self.f(scores)


class VS_LightGCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset, i):
        super(VS_LightGCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.id = i
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.alpha = self.config['alpha']

        # load the svd embedding
        user_file = "../data/" + world.dataset + "/model" + str(self.id) + "/user_embedding.pkl"
        item_file = "../data/" + world.dataset + "/model" + str(self.id) + "/item_embedding.pkl"
        try:
            user_embdding, item_embedding = self.readEmbedding(user_file, item_file)
            self.user_emb0 = torch.tensor(user_embdding).to(world.device)
            self.item_emb0 = torch.tensor(item_embedding).to(world.device)
            self.embedding_user.weight.data = torch.tensor(user_embdding).to(world.device)
            self.embedding_item.weight.data = torch.tensor(item_embedding).to(world.device)
            world.cprint("load the svd embedding")
        except:
            world.cprint('please generate the svd embedding in create_graph.py')
            exit(0)

        self.f = nn.Sigmoid()
        self.Graph, self.adj_mat = self.dataset.getSparseGraph()
        print(f"vs_lgn is already to go(dropout:{self.config['dropout']})")


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def readEmbedding(self, user_file, item_file):
        '''
        load the pretrain svd embedding
        '''
        with open(user_file, 'rb') as file1:
            user_embedding = pickle.load(file1)
        with open(item_file, 'rb') as file2:
            item_embedding = pickle.load(file2)
        return user_embedding, item_embedding

    def computer(self):
        """
        propagate methods for VS_LightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        all_emb0 = torch.cat([self.user_emb0, self.item_emb0])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb + self.alpha * all_emb0
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb) + self.alpha * all_emb0
            embs.append((self.n_layers - layer) * all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def saveEmbedding(self, filepath):
        '''
        save the embedding for DMV_GCN
        '''
        users, items = self.computer()
        users = users.detach().cpu()
        items = items.detach().cpu()
        user_file = filepath + '/model_user_embedding.pkl'
        item_file = filepath + '/model_item_embedding.pkl'
        with open(user_file, 'wb') as file:
            pickle.dump(users, file)
        with open(item_file, 'wb') as file:
            pickle.dump(items, file)
        print('write over!')

    def getCosin(self):
        # norm = torch.norm(embedding,p=2,dim=1,keepdim=True)
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        embedding = torch.cat([users_emb, items_emb])
        adj = self.Graph.to_dense()
        cos_1 = embedding / (torch.norm(embedding, 2, -1, keepdim=True).expand_as(embedding) + 1e-12)
        cos = torch.matmul(cos_1,cos_1.T)
        cos_mask = cos*adj
        cos_mask = torch.sum(cos_mask)/(torch.count_nonzero(adj))
        return cos_mask.cpu().item()


class DMV_GCN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(DMV_GCN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.f = nn.Sigmoid()
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.n_layers = self.config['n_layers']
        try:
            # import the trained users embedding and items embedding
            self.user1_embedding, self.item1_embedding = self.getEmbedding('../data/' + str(world.dataset) + '/model1/layer' + str(self.n_layers))
            self.user2_embedding, self.item2_embedding = self.getEmbedding('../data/' + str(world.dataset) + '/model2/layer' + str(self.n_layers))
            self.user3_embedding, self.item3_embedding = self.getEmbedding('../data/' + str(world.dataset) + '/model3/layer' + str(self.n_layers))
            self.rating1 = torch.matmul(self.user1_embedding, self.item1_embedding.T).flatten().reshape(-1, 1)
            self.rating2 = torch.matmul(self.user2_embedding, self.item2_embedding.T).flatten().reshape(-1, 1)
            self.rating3 = torch.matmul(self.user3_embedding, self.item3_embedding.T).flatten().reshape(-1, 1)
            self.ratings = torch.cat([self.rating1, self.rating2, self.rating3], dim=1)
            self.dim = len(self.user1_embedding[0])
        except:
            cprint("please use VS_lightgcn to generate the user and item embeddings in main.py")
            exit(0)
        # get the indicative vectors from each view
        self.indicate_vector1 = self.getIndicate_Vector(self.user1_embedding, self.item1_embedding)
        self.indicate_vector2 = self.getIndicate_Vector(self.user2_embedding, self.item2_embedding)
        self.indicate_vector3 = self.getIndicate_Vector(self.user3_embedding, self.item3_embedding)

        self.h = nn.Parameter(torch.randn(2 * self.dim).to(world.device))
        self.Graph, self.adj_mat = self.dataset.getSparseGraph()

    def getEmbedding(self,filepath):
        '''
        read the pretrain embedding from VS_LightGCN
        '''
        user_file = filepath+'/model_user_embedding.pkl'
        item_file = filepath+'/model_item_embedding.pkl'

        with open(user_file,'rb') as file:
            user_embedding = pickle.load(file).to(world.device)
        with open(item_file,'rb') as file:
            item_embedding = pickle.load(file)
        return user_embedding,item_embedding.to(world.device)

    def getIndicate_Vector(self,user_embedding,item_embedding):
        '''
        generate the indicative vector
        '''
        user_embedding = user_embedding.expand(self.num_items,self.num_users,self.dim)
        user_embedding = user_embedding.permute(1,0,2)
        item_embedding = item_embedding.expand(self.num_users,self.num_items,self.dim)
        indicate_vector = torch.cat([user_embedding,item_embedding],dim=2)
        return indicate_vector

    def computer(self):
        '''
        attention fusion
        '''
        attention1 = torch.matmul(self.indicate_vector1,self.h)
        attention2 = torch.matmul(self.indicate_vector2,self.h)
        attention3 = torch.matmul(self.indicate_vector3,self.h)

        attention1 = attention1.flatten().reshape(-1,1)
        attention2 = attention2.flatten().reshape(-1,1)
        attention3 = attention3.flatten().reshape(-1,1)

        attention = torch.cat([attention1, attention2, attention3], dim=1)
        weight = F.softmax(attention, dim=1)
        rating = self.f(torch.sum(self.ratings * weight, dim=1))
        return rating

    def computer1(self):
        weight = F.softmax(self.weight,dim=1)
        weight_reg = torch.norm(weight,2)
        # a = weight*self.ratings
        ratings = self.f(torch.sum(weight*self.ratings,dim=1))
        return ratings,weight_reg

    def computer2(self,a):
        rating = self.ratings[:, 0]
        pos_index = torch.where(rating>a)
        pos_rating = torch.max(self.ratings[pos_index],dim=1).values
        rating[pos_index] = pos_rating
        self.rating = rating

    def bpr_loss(self,user,pos,neg):
        ratings, weight_reg = self.computer1()
        ratings = ratings.view(self.num_users,self.num_items)
        pos_scores = ratings[user,pos]
        neg_scores = ratings[user,neg]
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss,weight_reg

    def loss(self,user,item):
        '''
        caculate the loss value
        '''
        ratings,weight_reg = self.computer1()
        ratings = ratings.view(self.num_users,self.num_items)
        rating = ratings[user,item]
        loss = torch.sum(-torch.log(rating))
        return loss,weight_reg

    def getUsersRating(self, users):
        rating,_ = self.computer1()
        # rating = self.rating
        item = torch.tensor([i for i in range(self.num_items)],dtype=int).to(world.device)
        ratings = []
        for user in users:
            index = torch.ones(size=[self.num_items],dtype=int).to(world.device) * user * self.num_items + item
            ratings.append(rating[index.tolist()].tolist())
        ratings = torch.tensor(ratings).view(len(users), self.num_items)
        ratings = self.f(ratings)
        return ratings

