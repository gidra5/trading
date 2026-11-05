"""Dense account cells, quotiented by permutations of resting orders.

Every cell is allocated, including cells whose representative is infeasible.
The numeric order block is (passive distance x/(1+x), signed notional/equity).
Wealth is a separate conditioning axis because quote-size bounds are absolute.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import numpy as np


@dataclass(frozen=True)
class AccountGrid:
    max_orders: int = 2
    price_bins: int = 2
    size_bins: int = 4
    exposure_points: tuple = (-2.5,-2.,-1.,0.,1.,2.,2.5)
    wealth_points: tuple = (1000.,5000.,10000.,20000.,100000.)
    price_scale: float = .0001

    def __post_init__(self):
        if self.max_orders < 0 or self.max_orders > 3 or self.price_bins < 1 or self.size_bins < 2 or self.size_bins%2:
            raise ValueError('Require up to three orders and an even signed-size partition')
        if (len(self.exposure_points)<2 or len(self.wealth_points)<2 or self.price_scale<=0 or not np.all(np.diff(self.exposure_points)>0)
                or not np.all(np.diff(self.wealth_points)>0) or self.wealth_points[0]<=0
                or 0. not in self.exposure_points):
            raise ValueError('Invalid axes, or missing exact cash exposure')

    def arrays(self):
        e=np.asarray(self.exposure_points,dtype=float);w=np.asarray(self.wealth_points,dtype=float)
        size_edges=np.linspace(-2.,2.,self.size_bins+1)
        size_points=(size_edges[:-1]+size_edges[1:])/2
        # Equal-width cells cover the entire transformed [0,1) price axis.
        types=[((j+.5)/self.price_bins,size_points[k])
            for j in range(self.price_bins) for k in range(self.size_bins)]
        T=len(types);offsets=np.cumsum([0]+[T**k for k in range(self.max_orders+1)])
        lookup=np.full(offsets[-1],-1,dtype=np.int64);books=[];counts=[]
        for k in range(self.max_orders+1):
            for ids in combinations_with_replacement(range(T),k):
                lookup[offsets[k]+sum(v*T**j for j,v in enumerate(ids))]=len(books)
                row=np.zeros((self.max_orders,2))
                for j,ix in enumerate(ids):row[j]=types[ix]
                books.append(row);counts.append(k)
        coordinates=np.array([(x,*book.ravel()) for book in books for x in e])
        return dict(exposures=e,wealth=w,size_edges=size_edges,books=np.asarray(books),
            order_counts=np.repeat(np.asarray(counts,dtype=np.int64),len(e)),
            book_counts=np.asarray(counts,dtype=np.int64),code_offsets=offsets.astype(np.int64),
            book_lookup=lookup,coordinates=coordinates,price_bins=self.price_bins,size_bins=self.size_bins,
            price_scale=self.price_scale,exposure_edges=np.r_[-3.,(e[:-1]+e[1:])/2,3.])

    def canonical_cell(self, exposure, orders):
        """orders are unordered (transformed distance, signed size) pairs."""
        a=self.arrays();T=self.price_bins*self.size_bins;k=len(orders)
        if k>self.max_orders or not -3<=exposure<=3:raise ValueError('Outside account domain')
        ids=[]
        for z,size in orders:
            if not 0<=z<1 or not -2<=size<=2 or size==0:raise ValueError('Invalid order coordinates')
            ids.append(min(int(z*self.price_bins),self.price_bins-1)*self.size_bins+
                min(max(np.searchsorted(a['size_edges'],size,side='right')-1,0),self.size_bins-1))
        ids.sort();code=a['code_offsets'][k]+sum(ix*T**j for j,ix in enumerate(ids))
        book=a['book_lookup'][code]
        ei=min(max(np.searchsorted(a['exposure_edges'],exposure,side='right')-1,0),len(a['exposures'])-1)
        return int(book*len(a['exposures'])+ei)

    def sample_cell(self, cell, rng):
        a=self.arrays();E=len(a['exposures']);book,ei=divmod(cell,E);result=[]
        for z,n in a['books'][book,:a['book_counts'][book]]:
            pi=min(int(z*self.price_bins),self.price_bins-1)
            si=np.searchsorted(a['size_edges'],n,side='right')-1
            result.append((rng.uniform(pi/self.price_bins,(pi+1)/self.price_bins),
                rng.uniform(a['size_edges'][si],a['size_edges'][si+1])))
        return rng.uniform(a['exposure_edges'][ei],a['exposure_edges'][ei+1]),tuple(sorted(result))


@dataclass
class CellErrors:
    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def empty(cls,cells):
        return cls(np.zeros(cells,dtype=np.int64),np.zeros(cells),np.zeros(cells))

    def observe(self,cell,residual):
        if not np.isfinite(residual):return
        self.count[cell]+=1;n=self.count[cell];delta=residual-self.mean[cell]
        self.mean[cell]+=delta/n;self.m2[cell]+=delta*(residual-self.mean[cell])

    def allocate(self,value,variance,total_branches,budget,seed):
        """Adaptive compute allocation; NEVER modifies market probabilities.

        Keep at least two outcomes per cell and random exploration everywhere.
        Historical residuals affect priority only; estimates use fresh targets.
        """
        if total_branches<2 or budget<2:raise ValueError('Need at least two outcomes')
        C=len(self.count);n=np.full(C,min(2,total_branches),dtype=np.int64)
        v=np.nan_to_num(value,nan=-1.,neginf=-1.,posinf=1.)+self.mean
        scale=max(float(np.std(v)),1e-5)
        p=.2/C+.8*np.exp(np.clip((v-v.max())/scale,-30,0))
        p+=np.sqrt(np.maximum(np.nan_to_num(variance,nan=0.,posinf=0.),0))+abs(self.mean)
        p+=np.sqrt(self.m2/np.maximum(self.count-1,1))
        p/=p.sum();rng=np.random.default_rng(seed)
        remaining=int(min(C*total_branches,C*budget)-n.sum())
        capacity=total_branches-2
        if remaining>0 and capacity>0:
            # Weighted sampling of available compute slots, without a Python
            # loop per extra draw. A cell can win at most its remaining slots.
            keys=rng.exponential(size=(C,capacity))/p[:,None]
            selected=np.argpartition(keys.ravel(),remaining-1)[:remaining]
            n+=np.bincount(selected//capacity,minlength=C)
        return n
