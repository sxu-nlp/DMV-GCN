"""
Liu, F., Liao, J., Zheng, J. et al.
GCN recommendation model based on the fusion of dynamic multiple-view latent interest topics.
Int. J. Mach. Learn. & Cyber. (2022).
"""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go DMV_GCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of DMV_GCN/VS_LightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of DMV_GCN/VS_LightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-3,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like ml-1m")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='lastfm',
                        help="available datasets: [lastfm,mk-100k,ml-1m]")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--comment', type=str,default="vs_lgn")
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--model', type=str, default='vs_lgn', help='rec-model, support [mf, vs_lgn]')
    parser.add_argument('--graphID', type=str, default="1",help="graph1 graph2 graph3,support [1,2,3]")
    parser.add_argument('--alpha', type=float, default=0.4, help="the Personalized represent weight,supoort [0-1]")
    parser.add_argument('--save',type=int,default=0,help='1 : save the embedding')
    parser.add_argument('--stop',type=int,default=990,
                        help='Select the epoch with the highest performance in VS_LightGCN for DMV_GCN')
    parser.add_argument('--t',nargs='?',default="[64,52]",
                        help='t concepts for constructing multiple views(view1,view2),t support [1-64]')
    parser.add_argument('--beta',nargs='?',default="[1e-6,1e-5]",
                        help='beta for filtering the similarity of the lower edge. ')
    return parser.parse_args()
