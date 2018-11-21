#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# k-means centers
dim_dict = {5 : (1,5), 10 : (2,5), 20 : (4,5)}


for k in [5,10,20]:
    for pp in [True,False]:
        fig = plt.figure(figsize=tuple(reversed(dim_dict[k])))
        gs = fig.add_gridspec(*dim_dict[k])
        gs.update(wspace=0.,hspace=0.)
        
        pp_str = 'pp' if pp else ''

        data = np.load(f'data/{k}-clusters{pp_str}.npz')
        objs, mu = data['objs']/70000,data['mu']
        
        for i in range(k):
            ind = np.unravel_index(i, dim_dict[k])
            ax = fig.add_subplot(gs[ind])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.imshow(mu[i].reshape((28,28)),aspect='equal',cmap='binary')

    plt.savefig(f'../figures/{k}{pp_str}-centers.pdf',bbox_inches='tight')
    plt.clf()
    
    data = np.load(f'data/{k}-clusters.npz')
    datapp = np.load(f'data/{k}-clusterspp.npz')

    objs,objspp = data['objs']/70000, datapp['objs']/70000

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(objs,'o',ms=4,label="uniform init")
    ax.plot(objspp,'o',ms=4,label="k-means++ init")
    ax.set_xlabel('Iteration', size=18)
    ax.set_ylabel('Objective/N', size=18)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=16)
    plt.savefig(f'../figures/{k}-objective.pdf',bbox_inches='tight')
    plt.clf()
