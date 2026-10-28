"""Transparent route baselines, tabular Q-learning and a NumPy DQN."""
from collections import defaultdict
import numpy as np

class Heuristic:
    def __init__(self,kind): self.kind=kind
    def reset(self,env):
        travel,_,_=env.snapshot(); remaining=list(range(env.park.n)); pos=env.pos; self.order=[]
        while remaining:
            a=min(remaining,key=lambda j:travel[pos,j]); self.order.append(a); remaining.remove(a); pos=a
    def act(self,env):
        mask=env.mask(); candidates=np.flatnonzero(mask[:env.park.n])
        if not len(candidates): return env.park.n
        travel,waits,_=env.snapshot()
        if self.kind=='Frozen route':
            return next(a for a in self.order if a in candidates)
        if self.kind=='Nearest': return int(candidates[np.argmin(travel[env.pos,candidates])])
        score=env.values[candidates]/(travel[env.pos,candidates]+waits[candidates]+env.park.duration[candidates])
        return int(candidates[np.argmax(score)])

class QLearning:
    def __init__(self,actions,seed,gamma=.98):
        self.actions=actions; self.rng=np.random.default_rng(seed); self.gamma=gamma
        self.q={}; self.count=defaultdict(int)
    def values(self,key): return self.q.get(key,np.zeros(self.actions))
    def act(self,env,epsilon=0):
        allowed=np.flatnonzero(env.mask())
        if self.rng.random()<epsilon: return int(self.rng.choice(allowed))
        values=self.values(env.tabular_state())[allowed]
        # Seeded tie breaking also applies to completely unseen states.
        best=allowed[np.isclose(values,values.max(),rtol=0,atol=1e-10)]
        return int(self.rng.choice(best))
    def update(self,key,a,r,nextkey,nextmask,done):
        if key not in self.q: self.q[key]=np.zeros(self.actions)
        target=r if done else r+self.gamma*self.values(nextkey)[nextmask].max()
        self.count[(key,a)]+=1
        alpha=max(.08,.7/(1+.02*self.count[(key,a)]))
        self.q[key][a]+=alpha*(target-self.q[key][a])

class Network:
    def __init__(self,inputs,outputs,rng,hidden=48):
        dims=[inputs,hidden,hidden,outputs]
        self.params=[]
        for a,b in zip(dims[:-1],dims[1:]):
            self.params.extend([rng.normal(0,np.sqrt(2/a),(a,b)).astype(np.float32),np.zeros(b,np.float32)])
        self.m=[np.zeros_like(p) for p in self.params]; self.v=[np.zeros_like(p) for p in self.params]; self.steps=0
    def forward(self,x,cache=False):
        w1,b1,w2,b2,w3,b3=self.params
        h1=np.maximum(x@w1+b1,0); h2=np.maximum(h1@w2+b2,0); y=h2@w3+b3
        return (y,(x,h1,h2)) if cache else y
    def loss_grad(self,x,actions,targets):
        y,(x,h1,h2)=self.forward(x,True); idx=np.arange(len(x)); diff=y[idx,actions]-targets
        loss=np.where(abs(diff)<=1,.5*diff**2,abs(diff)-.5).mean()
        dy=np.zeros_like(y); dy[idx,actions]=np.clip(diff,-1,1)/len(x)
        w1,b1,w2,b2,w3,b3=self.params
        dw3=h2.T@dy; db3=dy.sum(0); dh2=(dy@w3.T)*(h2>0)
        dw2=h1.T@dh2; db2=dh2.sum(0); dh1=(dh2@w2.T)*(h1>0)
        return float(loss),[x.T@dh1,dh1.sum(0),dw2,db2,dw3,db3]
    def learn(self,x,actions,targets,lr=.001):
        loss,grads=self.loss_grad(x,actions,targets)
        norm=np.sqrt(sum(float((g*g).sum()) for g in grads)); scale=min(1,10/(norm+1e-8))
        self.steps+=1
        for i,(p,g) in enumerate(zip(self.params,grads)):
            g=g*scale; self.m[i]=.9*self.m[i]+.1*g; self.v[i]=.999*self.v[i]+.001*g*g
            p-=lr*(self.m[i]/(1-.9**self.steps))/(np.sqrt(self.v[i]/(1-.999**self.steps))+1e-8)
        return loss
    def copy_from(self,other): self.params=[p.copy() for p in other.params]

class DQN:
    def __init__(self,inputs,actions,seed,gamma=.98,capacity=12000):
        self.rng=np.random.default_rng(seed); self.gamma=gamma; self.actions=actions
        self.net=Network(inputs,actions,self.rng); self.target=Network(inputs,actions,self.rng)
        self.target.copy_from(self.net); self.capacity=capacity; self.size=0; self.cursor=0; self.steps=0
        self.x=np.empty((capacity,inputs),np.float32); self.nx=np.empty_like(self.x)
        self.a=np.empty(capacity,int); self.r=np.empty(capacity,np.float32); self.done=np.empty(capacity,bool)
        self.masks=np.empty((capacity,actions),bool)
    def act(self,env,epsilon=0,stale=False):
        allowed=np.flatnonzero(env.mask())
        if self.rng.random()<epsilon: return int(self.rng.choice(allowed))
        x=env.observe()
        if stale:
            # Freeze only queue and travel features at their departure-time values.
            start=env.park.size+env.park.n+2+3+env.park.n
            base=env.stale_observation
            x[start:start+3*env.park.n]=base[start:start+3*env.park.n]
        values=self.net.forward(x[None,:])[0]
        return int(allowed[np.argmax(values[allowed])])
    def remember(self,x,a,r,nx,mask,done):
        i=self.cursor; self.x[i]=x; self.a[i]=a; self.r[i]=r; self.nx[i]=nx; self.masks[i]=mask; self.done[i]=done
        self.cursor=(i+1)%self.capacity; self.size=min(self.size+1,self.capacity); self.steps+=1
        if self.size<256 or self.steps%2: return None
        batch=self.rng.integers(self.size,size=64)
        values=self.target.forward(self.nx[batch]); values=np.where(self.masks[batch],values,-1e9)
        bootstrap=np.where(self.done[batch],0,values.max(1))
        targets=self.r[batch]+self.gamma*bootstrap
        loss=self.net.learn(self.x[batch],self.a[batch],targets)
        if self.net.steps%200==0: self.target.copy_from(self.net)
        return loss
    def save(self,path): np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.net.params)})
    def load(self,path):
        z=np.load(path); self.net.params=[z[f'p{i}'].copy() for i in range(6)]; self.target.copy_from(self.net)
