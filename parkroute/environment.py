"""A single-visitor, time-varying attraction-selection environment."""
from dataclasses import dataclass
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path

SCENARIOS = ['calm','peak','disruption','shifted']
@dataclass
class Config:
    budget: float = 150.0
    walk_speed: float = 70.0  # metres/minute
    road_interval: float = 15.0
    gamma: float = .98
    episodes: int = 2500
    train_seeds: tuple = (11,22,33)
    test_cases: int = 80

class Park:
    def __init__(self,df):
        self.df = df.copy(); self.n = len(df)
        self.xy = df[['x_m','y_m']].to_numpy()
        self.xy = np.vstack([self.xy, [self.xy[:,0].mean(),self.xy[:,1].min()-120],
                            [self.xy[:,0].mean()+80,self.xy[:,1].min()-120],
                            self.xy.mean(axis=0)])
        self.entry,self.exit,self.rest = self.n,self.n+1,self.n+2
        self.size = self.n+3; self.actions = self.n+2
        delta = self.xy[:,None,:]-self.xy[None,:,:]
        dist = np.sqrt((delta**2).sum(axis=2))
        graph = np.zeros_like(dist)
        for i in range(self.size):
            js = np.argsort(dist[i])[1:4]
            graph[i,js] = dist[i,js]
        tree = minimum_spanning_tree(dist).toarray()
        graph = np.maximum(graph,graph.T)
        graph = np.maximum(graph,np.maximum(tree,tree.T))
        self.graph = graph; self.edges = np.array(np.where(np.triu(graph)>0)).T
        self.duration = df.duration.to_numpy(dtype=float)
        self.base = df.base_value.to_numpy(dtype=float)
        rng = np.random.default_rng(2026)
        self.profiles = self.base[None,:]*rng.uniform(.7,1.3,(3,self.n))
        self.queue_base = 4+13*df.popularity.to_numpy()
        self.phase = rng.uniform(-.5,.5,self.n)
        self.open_at = np.zeros(self.n); self.open_at[[2,6]] = [20,35]
        self.close_at = np.full(self.n,220.); self.close_at[[3,8]] = [115,125]
        self.hot = np.argsort(self.base)[-3:]
        # A deterministic interior edge receives the closure event.
        candidates = [tuple(e) for e in self.edges if max(e)<self.n]
        # Select a connected closure with high shortest-path use.
        _, predecessors = shortest_path(graph, directed=False, return_predecessors=True)
        counts = {e: 0 for e in candidates}
        for origin in range(self.size):
            for target in range(origin + 1, self.size):
                v = target
                while v != origin:
                    u = int(predecessors[origin, v]); edge = tuple(sorted((u, v)))
                    if edge in counts: counts[edge] += 1
                    v = u
        eligible = []
        for edge in candidates:
            trial = graph.copy(); i, j = edge; trial[i, j] = trial[j, i] = 0
            if np.isfinite(shortest_path(trial, directed=False)).all(): eligible.append(edge)
        self.blocked_edge = max(eligible, key=lambda e: (counts[e], graph[e]))

class Environment:
    def __init__(self,park,config=None):
        self.park = park; self.cfg = config or Config()
    def reset(self,seed,scenario='calm',budget=None):
        if scenario not in SCENARIOS: raise ValueError('Unknown scenario')
        self.seed,self.scenario = int(seed),scenario
        self.budget = self.cfg.budget if budget is None else float(budget)
        rng = np.random.default_rng(seed)
        self.profile = int(rng.integers(3)); self.values = self.park.profiles[self.profile].copy()
        self.noise = rng.normal(0,1.8,(40,self.park.n))
        self.road_noise = rng.uniform(.95,1.7,(40,len(self.park.edges)))
        self.pos = self.park.entry; self.time = 0.; self.visited = np.zeros(self.park.n,bool)
        self.rested = False; self.done = False; self.total_reward=0.
        self.walk=self.wait=self.service=self.rest_time=0.; self.distance=0.
        self.history=[]; self.cache={}
        return self.observe()
    @property
    def event_time(self): return 40. if self.scenario=='shifted' else 60.
    def queues(self,t=None):
        t = self.time if t is None else t
        k = min(int(t//10),39)
        scale = 1.65 if self.scenario=='peak' else 1.
        q = self.park.queue_base*scale*(.8+.35*np.sin(np.pi*t/180+self.park.phase)) + self.noise[k]
        if self.scenario in ['disruption','shifted'] and self.event_time <= t < self.event_time+55:
            q[self.park.hot] += 22 if self.scenario=='disruption' else 32
        return np.maximum(q,0.)
    def roads(self,t=None):
        t = self.time if t is None else t
        k = min(int(t//self.cfg.road_interval),39)
        event = self.scenario in ['disruption','shifted'] and self.event_time<=t<self.event_time+55
        key = (k,event)
        if key not in self.cache:
            g = self.park.graph.copy()
            mult = self.road_noise[k].copy()
            if self.scenario=='peak': mult *= 1.3
            if event: mult *= 1.35
            for (i,j),m in zip(self.park.edges,mult): g[i,j]=g[j,i]=g[i,j]*m
            if event:
                i,j = self.park.blocked_edge; g[i,j]=g[j,i]=0
            d,p = shortest_path(g,directed=False,return_predecessors=True)
            self.cache[key] = (d/self.cfg.walk_speed,p,g)
        return self.cache[key]
    def snapshot(self):
        travel,_,_ = self.roads()
        waits = self.queues()
        opened = (self.time>=self.park.open_at)&(self.time<self.park.close_at)
        return travel,waits,opened
    def mask(self):
        p = self.park; travel,waits,opened = self.snapshot()
        # Everyone receives the same reserve check based on CURRENT conditions.
        arrival = self.time+travel[self.pos,:p.n]
        end = arrival+waits+p.duration
        allowed = (~self.visited)&opened&(end<=p.close_at)&(
            end+travel[:p.n,p.exit] <= self.budget)
        mask = np.r_[allowed, True, (not self.rested) and
                     self.time+travel[self.pos,p.rest]+8+travel[p.rest,p.exit]<=self.budget]
        if self.done: mask[:] = False
        return mask.astype(bool)
    def observe(self):
        p = self.park; travel,waits,opened = self.snapshot()
        loc = np.eye(p.size)[self.pos]
        return np.r_[loc,self.visited.astype(float), (self.budget-self.time)/self.budget,
            self.time/220, np.eye(3)[self.profile],self.values,
            np.minimum(waits,80)/80,np.minimum(travel[self.pos,:p.n],30)/30,
            np.minimum(travel[:p.n,p.exit],30)/30,opened.astype(float),float(self.rested)].astype(np.float32)
    def tabular_state(self):
        # Deliberately coarse aggregation makes the table small enough to inspect.
        crowd = int(np.mean(self.queues())//12)
        visited = sum((1<<i) for i,v in enumerate(self.visited) if v)
        return (self.pos,visited,int(self.time//20),self.profile,min(crowd,5),self.rested,
                int(self.scenario in ['disruption','shifted'] and self.event_time<=self.time<self.event_time+55),
                int(self.budget//30))
    def step(self,action):
        if self.done: raise RuntimeError('Episode already terminated')
        if not 0<=action<self.park.actions or not self.mask()[action]:
            raise ValueError('Masked or out-of-range action')
        p = self.park; start=self.time; origin=self.pos
        target = action if action<p.n else (p.exit if action==p.n else p.rest)
        travel,pred,_ = self.roads()
        walk = float(travel[self.pos,target]); path=[target]; v=target
        while v!=self.pos:
            v=int(pred[self.pos,v])
            if v<0: raise RuntimeError('Disconnected path')
            path.append(v)
        path.reverse()
        # Road costs remain at departure values for this macro-action.
        self.time += walk; self.walk += walk
        dist = sum(p.graph[i,j] for i,j in zip(path[:-1],path[1:])); self.distance+=dist
        self.pos=target; queue=duration=rest=utility=0.
        reward=-.08*walk
        if action<p.n:
            queue=float(self.queues(self.time)[action]); duration=float(p.duration[action])
            self.time += queue+duration; self.wait+=queue; self.service+=duration
            # Completion after the attraction's closing time provides no utility.
            if self.time<=p.close_at[action]:
                self.visited[action]=True; utility=float(self.values[action])
            reward += 10*utility-.12*queue-.025*duration
        elif action==p.n:
            self.done=True; reward += 5 if self.time<=self.budget else -25
        else:
            rest=8.; self.time+=rest; self.rest_time+=rest; self.rested=True
            reward-=.025*rest
        if self.time>self.budget and not self.done:
            self.done=True; reward-=25
        self.total_reward+=reward
        self.history.append(dict(action=int(action),origin=int(origin),target=int(target),start=start,end=self.time,
                                 walk=walk,wait=queue,service=duration,rest=rest,utility=utility,
                                 path=path,reward=reward))
        return self.observe(),float(reward),self.done
    def metrics(self):
        success=self.done and self.pos==self.park.exit and self.time<=self.budget
        return dict(reward=self.total_reward,coverage=100*float(self.visited.mean()),
                    satisfaction=100*float(self.values[self.visited].sum()/self.values.sum()),
                    wait=self.wait,walk=self.walk,distance=self.distance,time=self.time,
                    completion=100*float(success),visits=int(self.visited.sum()),profile=self.profile)
