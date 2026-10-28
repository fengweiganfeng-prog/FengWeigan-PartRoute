"""Training, paired testing, budget sensitivity and confidence intervals."""
from pathlib import Path
import json, pickle, time
import numpy as np
import pandas as pd
from .environment import Environment,Config,SCENARIOS
from .agents import Heuristic,QLearning,DQN

POLICIES=['Frozen route','Nearest','Adaptive greedy','Q-learning','DQN','DQN stale input']

def rollout(env,agent,stale=False):
    env.stale_observation=env.observe().copy()
    if isinstance(agent,Heuristic): agent.reset(env)
    for _ in range(30):
        if env.done: break
        a=agent.act(env,stale=stale) if isinstance(agent,DQN) else agent.act(env)
        env.step(a)
    if not env.done: raise RuntimeError('Episode exceeded decision limit')
    return env.metrics()

def train(park,cfg,root):
    root=Path(root); models=root/'outputs/models'; models.mkdir(parents=True,exist_ok=True)
    histories=[]; checks=[]; agents={}; start=time.perf_counter()
    for seed in cfg.train_seeds:
        env=Environment(park,cfg); obs=env.reset(1000000+seed*10000)
        q=QLearning(park.actions,seed,cfg.gamma); dqn=DQN(len(obs),park.actions,seed,cfg.gamma)
        for name,agent in [('Q-learning',q),('DQN',dqn)]:
            chooser=np.random.default_rng(seed+7)
            for episode in range(cfg.episodes):
                scene=SCENARIOS[int(chooser.integers(3))]
                x=env.reset(1000000+seed*10000+episode,scene)
                epsilon=max(.06,1-.94*episode/(.8*cfg.episodes))
                losses=[]
                for _ in range(30):
                    key=env.tabular_state(); a=agent.act(env,epsilon)
                    nx,r,done=env.step(a); mask=env.mask()
                    if name=='Q-learning': agent.update(key,a,r,env.tabular_state(),mask,done)
                    else:
                        loss=agent.remember(x,a,r,nx,mask,done)
                        if loss is not None: losses.append(loss)
                    x=nx
                    if done: break
                histories.append(dict(policy=name,train_seed=seed,episode=episode+1,scenario=scene,
                                      epsilon=epsilon,loss=np.mean(losses) if losses else np.nan,**env.metrics()))
                if (episode+1)%250==0:
                    rewards=[]
                    for case in range(24):
                        val=Environment(park,cfg); val.reset(5000000+case,SCENARIOS[case%3])
                        rewards.append(rollout(val,agent)['reward'])
                    checks.append(dict(policy=name,train_seed=seed,episode=episode+1,reward=np.mean(rewards)))
            print(f'{name}: training seed {seed}, {cfg.episodes} episodes',flush=True)
        with open(models/f'q_{seed}.pkl','wb') as f: pickle.dump(q,f)
        dqn.save(models/f'dqn_{seed}.npz'); agents[seed]={'Q-learning':q,'DQN':dqn}
    pd.DataFrame(histories).to_csv(root/'outputs/tables/training.csv',index=False)
    pd.DataFrame(checks).to_csv(root/'outputs/tables/learning_curve.csv',index=False)
    (root/'outputs/tables/training_metadata.json').write_text(json.dumps(dict(
        elapsed_seconds=time.perf_counter()-start,episodes_per_agent=cfg.episodes,
        train_seeds=list(cfg.train_seeds),validation_cases=24,
        validation_seeds=[5000000,5000023],test_seed_start=9000000,
        architecture=[len(obs),48,48,park.actions],gamma=cfg.gamma,
        replay_capacity=12000,batch_size=64,optimizer='Adam',learning_rate=.001,
        target_update_gradient_steps=200,train_every_transitions=2),indent=2))
    return agents,pd.DataFrame(histories),pd.DataFrame(checks)

def evaluate(park,cfg,agents,root):
    rows=[]
    for scene in SCENARIOS:
        for case in range(cfg.test_cases):
            for seed in cfg.train_seeds:
                for policy in POLICIES:
                    env=Environment(park,cfg); env.reset(9000000+case,scene)
                    agent=Heuristic(policy) if policy in POLICIES[:3] else agents[seed]['DQN' if policy=='DQN stale input' else policy]
                    metrics=rollout(env,agent,stale=policy=='DQN stale input')
                    rows.append(dict(scenario=scene,case=case,train_seed=seed,policy=policy,**metrics))
    df=pd.DataFrame(rows); df.to_csv(Path(root)/'outputs/tables/evaluation.csv',index=False)
    summary=df.groupby(['scenario','policy'])[['reward','coverage','satisfaction','wait','walk','distance','completion','visits']].mean().reset_index()
    summary.to_csv(Path(root)/'outputs/tables/summary.csv',index=False)
    return df,summary

def paired_intervals(df,root):
    # Average learned policies over training seeds before resampling CASES.
    case=df.groupby(['scenario','case','policy']).reward.mean().unstack('policy')
    rng=np.random.default_rng(601); rows=[]
    for scene in SCENARIOS:
        subset=case.loc[scene]
        for other in ['Frozen route','Nearest','Adaptive greedy','Q-learning','DQN stale input']:
            diff=(subset.DQN-subset[other]).to_numpy()
            boot=diff[rng.integers(len(diff),size=(2000,len(diff)))].mean(1)
            lo,hi=np.quantile(boot,[.025,.975])
            rows.append(dict(scenario=scene,comparison=other,mean=diff.mean(),low=lo,high=hi))
    out=pd.DataFrame(rows); out.to_csv(Path(root)/'outputs/tables/paired_intervals.csv',index=False)
    return out

def budget_experiment(park,cfg,agents,root):
    rows=[]
    for budget in [120,150,180]:
        for case in range(40):
            for seed in cfg.train_seeds:
                for policy in ['Nearest','Adaptive greedy','Q-learning','DQN']:
                    env=Environment(park,cfg); env.reset(9100000+case,'disruption',budget)
                    agent=Heuristic(policy) if policy in ['Nearest','Adaptive greedy'] else agents[seed][policy]
                    rows.append(dict(budget=budget,case=case,train_seed=seed,policy=policy,**rollout(env,agent)))
    out=pd.DataFrame(rows); out.to_csv(Path(root)/'outputs/tables/budget.csv',index=False)
    return out
