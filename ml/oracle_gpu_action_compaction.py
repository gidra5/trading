"""Exact candidate-state compaction for the dense GPU planner.

Standalone primitive, not yet integrated into the running Bellman operator.
Hashes only group candidates; full coordinate equality decides duplication.
Hash collisions can leave extra work but must never remove a distinct state.
"""
import triton
import triton.language as tl


@triton.jit
def compact_indices(states, feasible, AC:tl.constexpr,
                    HASH_MASK:tl.constexpr=0x7fffffffffffffff):
    tl.static_assert(AC>0 and (AC&(AC-1))==0,'Action width must be a positive power of two')
    lane=tl.arange(0,AC)
    hashed=tl.full((AC,),0,tl.uint64)
    multiplier=tl.full((),0x9e3779b97f4a7c15,tl.uint64)
    maximum=tl.full((),0xffffffffffffffff,tl.uint64)
    coordinates=()
    for j in tl.static_range(len(states)):
        coordinate=states[j]+tl.full((AC,),0.,tl.float64)
        coordinates+=(coordinate,)
        hashed=(hashed^coordinate.to(tl.uint64,bitcast=True))*multiplier
    key=tl.where(feasible,(hashed&(HASH_MASK&~(AC-1)))|lane.to(tl.uint64),maximum)
    key=tl.sort(key,descending=False)
    order=(key&(AC-1)).to(tl.int32);live=key!=maximum
    previous=tl.maximum(lane-1,0)
    previous_order=tl.gather(order,previous,0)
    duplicate=live&tl.gather(live,previous,0)&(lane>0)
    for j in tl.static_range(len(coordinates)):
        duplicate&=tl.gather(coordinates[j],order,0)==tl.gather(coordinates[j],previous_order,0)
    unique=live&~duplicate
    indices=tl.sort(tl.where(unique,order,AC),descending=False)
    return indices,tl.sum(unique.to(tl.int32),0)


@triton.jit
def _prefix_max(states,node,post_args,post_constants:tl.constexpr,indices,count,AC:tl.constexpr,LANES:tl.constexpr,POST:tl.constexpr):
    lane=tl.arange(0,LANES)
    index=tl.minimum(tl.gather(indices,lane,0),AC-1)
    packed=()
    for j in tl.static_range(len(states)):
        packed+=(tl.gather(states[j]+tl.full((AC,),0.,tl.float64),index,0),)
    value=POST(packed,node,post_args,post_constants)
    return tl.max(tl.where(lane<count,value,-float('inf')),0)


@triton.jit
def compact_max(states,feasible,node,post_args,post_constants:tl.constexpr,AC:tl.constexpr,POST:tl.constexpr):
    """Score the smallest supported prefix containing all retained candidates.

    Each uniform runtime branch has a compile-time vector size. Masking an
    unchanged full-size vector would still execute the expensive unused work.
    The supplied post-state function must depend only on the complete state,
    market node and shared arguments, never on the original action's label.
    """
    indices,count=compact_indices(states,feasible,AC)
    result=tl.full((),-float('inf'),tl.float64)
    if count>0:
        if AC<=8:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
        elif count<=8:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,8,POST)
        elif AC<=16:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
        elif count<=16:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,16,POST)
        elif AC<=32:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
        elif count<=32:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,32,POST)
        elif AC<=64:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
        elif count<=64:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,64,POST)
        elif AC<=128:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
        elif count<=128:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,128,POST)
        elif AC<=256:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
        elif count<=256:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,256,POST)
        elif AC<=512:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
        elif count<=512:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,512,POST)
        else:result=_prefix_max(states,node,post_args,post_constants,indices,count,AC,AC,POST)
    return result
