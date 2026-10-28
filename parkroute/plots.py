"""All analysis figures use the same output paths in the notebook and report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .environment import Environment,SCENARIOS
from .experiment import rollout
from .agents import Heuristic

COLORS={'Frozen route':'#8994a5','Nearest':'#c38230','Adaptive greedy':'#237e6d',
        'Q-learning':'#9a5db0','DQN':'#2f65b0','DQN stale input':'#b44a51'}
SCENE_LABELS=['Calm','Peak','Disruption','Shifted']
def style():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'figure.dpi':125,'savefig.dpi':180,
                         'font.family':'DejaVu Sans'})
def save(fig,root,name):
    fig.savefig(Path(root)/'outputs/figures'/name,bbox_inches='tight')
    return fig

def map_base(ax,park):
    for i,j in park.edges:
        ax.plot(park.xy[[i,j],0],park.xy[[i,j],1],color='#d6dce4',lw=1,zorder=0)
    ax.scatter(*park.xy[:park.n].T,s=65,c=park.base,cmap='viridis',vmin=.4,vmax=1,zorder=3)
    for i,(x,y) in enumerate(park.xy[:park.n]): ax.annotate(str(i+1),(x,y),xytext=(4,4),textcoords='offset points',fontsize=9)
    for node,symbol,color,label in [(park.entry,'s','#252c35','Entry'),(park.exit,'X','#b44a51','Exit'),(park.rest,'^','#59844a','Rest')]:
        x,y=park.xy[node]; ax.scatter(x,y,marker=symbol,c=color,s=90,zorder=4)
        ax.annotate(label,(x,y),xytext=(5,6),textcoords='offset points',fontsize=8)
    ax.set(xlabel='East displacement (m)',ylabel='North displacement (m)'); ax.set_aspect('equal')

def data_figure(source,park,root):
    style(); fig=plt.figure(figsize=(10.5,4.1)); grid=fig.add_gridspec(1,3,width_ratios=[1.6,1,1])
    ax=fig.add_subplot(grid[0]); map_base(ax,park); ax.set_title('Ten-attraction road graph')
    ax=fig.add_subplot(grid[1]); ax.scatter(source.n_reviews,source.rating,c='#237e6d',s=30)
    ax.set(xscale='log',xlabel='Review count (log scale)',ylabel='Rating',title='Public attraction records')
    ax=fig.add_subplot(grid[2]); ax.hist(source.duration,bins=[0,5,10,15,20,25,30],color='#2f65b0',edgecolor='white')
    ax.set(xlabel='Duration field',ylabel='Number of attractions',title='Visit durations')
    fig.tight_layout(); return save(fig,root,'01_data_map.png')

def dynamics_figure(park,cfg,root):
    fig,axes=plt.subplots(1,2,figsize=(10,3.1)); times=np.arange(0,181,2)
    for scene,color in zip(SCENARIOS,['#237e6d','#c38230','#2f65b0','#b44a51']):
        env=Environment(park,cfg); env.reset(9000007,scene)
        queues=[env.queues(t)[park.hot].mean() for t in times]
        travel=[env.roads(t)[0][park.entry,park.hot[-1]] for t in times]
        axes[0].plot(times,queues,color=color,label=scene.capitalize())
        axes[1].plot(times,travel,color=color)
    axes[0].set(xlabel='Elapsed time (min)',ylabel='Queue time (min)',title='Three highest-value attractions')
    axes[1].set(xlabel='Elapsed time (min)',ylabel='Travel time (min)',title='Entry to the highest-value attraction')
    for ax in axes:
        ax.axvline(60,c='#2f65b0',ls=':',lw=1); ax.axvline(40,c='#b44a51',ls=':',lw=1)
    axes[0].legend(ncol=2,fontsize=8); fig.tight_layout()
    return save(fig,root,'02_environment_changes.png')

def learning_figure(checks,root):
    fig,ax=plt.subplots(figsize=(9,3.1))
    for policy in ['Q-learning','DQN']:
        x=checks[checks.policy==policy].groupby('episode').reward.agg(['mean','min','max'])
        ax.plot(x.index,x['mean'],label=policy,color=COLORS[policy])
        ax.fill_between(x.index,x['min'],x['max'],color=COLORS[policy],alpha=.16)
    ax.set(xlabel='Training episodes',ylabel='Mean validation reward',title='Greedy validation policies on 24 fixed cases')
    ax.legend(); fig.tight_layout(); return save(fig,root,'03_learning_curves.png')

def performance_figure(df,root):
    fig,axes=plt.subplots(2,2,figsize=(10,5.2))
    for ax,metric,label in zip(axes.ravel(),['satisfaction','coverage','wait','completion'],
                               ['Preference satisfaction (%)','Attraction coverage (%)','Queue time (min)','On-time completion (%)']):
        for i,policy in enumerate(COLORS):
            if policy=='DQN stale input': continue
            vals=df[df.policy==policy].groupby('scenario')[metric].mean().reindex(SCENARIOS)
            ax.plot(range(4),vals,'o-',label=policy,color=COLORS[policy],ms=4)
        ax.set_xticks(range(4),SCENE_LABELS); ax.set_ylabel(label)
    axes[0,0].legend(fontsize=8,ncol=2); fig.tight_layout()
    return save(fig,root,'04_policy_comparison.png')

def paired_figure(paired,root):
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),sharey=True)
    for ax,other in zip(axes,['Adaptive greedy','Q-learning','DQN stale input']):
        x=paired[paired.comparison==other].set_index('scenario').loc[SCENARIOS]
        ax.errorbar(x['mean'],range(4),xerr=np.vstack([x['mean']-x.low,x.high-x['mean']]),fmt='o',color=COLORS['DQN'],capsize=3)
        ax.axvline(0,c='#555',ls='--',lw=1); ax.set_title('DQN minus '+other,fontsize=10)
        ax.set_xlabel('Paired reward difference'); ax.set_yticks(range(4),SCENE_LABELS); ax.invert_yaxis()
    fig.tight_layout(); return save(fig,root,'05_paired_differences.png')

def budget_figure(df,root):
    fig,axes=plt.subplots(1,2,figsize=(10,2.9))
    for policy in ['Nearest','Adaptive greedy','Q-learning','DQN']:
        x=df[df.policy==policy].groupby('budget')[['satisfaction','completion']].mean()
        for ax,metric,label in zip(axes,['satisfaction','completion'],['Preference satisfaction (%)','On-time completion (%)']):
            ax.plot(x.index,x[metric],'o-',label=policy,color=COLORS[policy]); ax.set(xlabel='Time budget (min)',ylabel=label)
            ax.set_xticks([120,150,180])
    axes[0].legend(fontsize=8,ncol=2); fig.tight_layout()
    return save(fig,root,'06_budget_sensitivity.png')

def route_figure(park,cfg,agents,root):
    fig=plt.figure(figsize=(10,5.3)); grid=fig.add_gridspec(2,2,height_ratios=[3,1]); rows=[]; histories={}
    for j,policy in enumerate(['Adaptive greedy','DQN']):
        env=Environment(park,cfg); env.reset(9000007,'disruption')
        agent=Heuristic(policy) if policy=='Adaptive greedy' else agents[cfg.train_seeds[0]]['DQN']
        metrics=rollout(env,agent); rows.append(dict(policy=policy,**metrics)); histories[policy]=env.history
        ax=fig.add_subplot(grid[0,j]); map_base(ax,park)
        for h in env.history:
            xy=park.xy[h['path']]; ax.plot(xy[:,0],xy[:,1],c=COLORS[policy],alpha=.75,lw=2)
        i,k=park.blocked_edge; ax.plot(park.xy[[i,k],0],park.xy[[i,k],1],'--',c='#b44a51',lw=2)
        order=[str(h['action']+1) for h in env.history if h['action']<park.n and h['utility']>0]
        ax.set_title(policy+': '+', '.join(order),fontsize=10)
    ax=fig.add_subplot(grid[1,:]); comp=[('walk','#9ca7b8','Walking'),('wait','#c38230','Queueing'),('service','#237e6d','Visiting'),('rest','#9a5db0','Rest')]
    for j,policy in enumerate(['Adaptive greedy','DQN']):
        left=0
        for key,color,label in comp:
            value=sum(h[key] for h in histories[policy]); ax.barh(j,value,left=left,color=color,label=label if j==0 else None)
            left+=value
    ax.axvline(cfg.budget,ls='--',c='#333',lw=1); ax.set(xlabel='Total time by component (min)',yticks=[0,1],yticklabels=['Adaptive greedy','DQN'])
    ax.legend(ncol=4,fontsize=8,loc='upper right'); fig.tight_layout()
    pd.DataFrame(rows).to_csv(Path(root)/'outputs/tables/example_routes.csv',index=False)
    (Path(root)/'outputs/tables/example_trace.json').write_text(json.dumps(histories,indent=2))
    return save(fig,root,'07_example_routes.png'),pd.DataFrame(rows)
