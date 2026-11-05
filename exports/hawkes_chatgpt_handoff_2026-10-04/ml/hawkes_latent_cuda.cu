// Float64 persistent, chunked particle filtering. One CUDA block per model.
typedef unsigned int U32;
typedef unsigned long long U64;

__device__ double uniform53(U64 seed, U64 offset) {
    U32 c0=(U32)offset, c1=(U32)(offset>>32), c2=0, c3=0;
    U32 k0=(U32)seed, k1=(U32)(seed>>32);
    for(int r=0;r<10;r++) {
        U64 p0=U64(0xD2511F53u)*c0, p1=U64(0xCD9E8D57u)*c2;
        U32 n0=U32(p1>>32)^c1^k0, n1=U32(p1), n2=U32(p0>>32)^c3^k1, n3=U32(p0);
        c0=n0;c1=n1;c2=n2;c3=n3;k0+=0x9E3779B9u;k1+=0xBB67AE85u;
    }
    U64 bits=(U64(c0)<<21)|(U64(c1)>>11);
    return fmax(double(bits)/9007199254740992.,1./18014398509481984.);
}
__device__ double normal53(U64 seed,U64 offset) {
    return sqrt(-2.*log(uniform53(seed,offset*2)))*cos(6.283185307179586*uniform53(seed,offset*2+1));
}
__device__ double logweight(int n,double prior,double b0,double b1,double s0,double s1,double y0,double y1,const double* factorial) {
    double f=n<1024?factorial[n]:lgamma(double(n)+1.);
    double r0=b0+s0*n,r1=b1+s1*n;
    return n*log(prior)-prior-f+y0*log(r0)-r0+y1*log(r1)-r1;
}
__device__ double logratio(int n,double prior,double b0,double b1,double s0,double s1,double y0,double y1) {
    return log(prior)-log(double(n)+1.)-s0-s1+y0*log1p(s0/(b0+s0*n))+y1*log1p(s1/(b1+s1*n));
}
struct Integral {double ll,mean,active;int lo,hi;};
__device__ Integral integrate(double prior,double b0,double b1,double s0,double s1,double y0,double y1,const double* factorial,bool force=false) {
    Integral out;
    out.lo=out.hi=-3;out.ll=-1./0.;out.mean=out.active=0.;
    if(!isfinite(prior)||prior>1.e10||!isfinite(b0)||!isfinite(b1)) return out;
    if(prior<=0.) {
        out.ll=y0*log(b0)-b0+y1*log(b1)-b1;out.lo=out.hi=-1;return out;
    }
    if(!force&&((s0+s1)==0.||(y0+y1)==0.)) {
        out.mean=(y0+y1)==0.?prior*exp(-s0-s1):prior;
        out.ll=(y0+y1)==0.?-b0-b1+prior*expm1(-s0-s1):y0*log(b0)-b0+y1*log(b1)-b1;
        out.active=-expm1(-out.mean);out.lo=out.hi=-2;return out;
    }
    int low=0,high=max(1,int(fmin(prior,10000001.)));
    while(logratio(high,prior,b0,b1,s0,s1,y0,y1)>0.) {
        high*=2;if(high>10000000)return out;
    }
    while(low<high) {
        int mid=(low+high)/2;
        if(logratio(mid,prior,b0,b1,s0,s1,y0,y1)>0.)low=mid+1;else high=mid;
    }
    int lo=low,hi=low;
    double peak=logweight(low,prior,b0,b1,s0,s1,y0,y1,factorial);
    while(lo>0) {
        double r=logratio(lo-1,prior,b0,b1,s0,s1,y0,y1);
        if(r>0.&&logweight(lo-1,prior,b0,b1,s0,s1,y0,y1,factorial)-peak-log(-expm1(-r)) < -27.631021115928547)break;
        --lo;
    }
    while(true) {
        double r=logratio(hi+1,prior,b0,b1,s0,s1,y0,y1);
        if(r<0.&&logweight(hi+1,prior,b0,b1,s0,s1,y0,y1,factorial)-peak-log(-expm1(r)) < -27.631021115928547)break;
        ++hi;if(hi-lo>100000)return out;
    }
    double total=0.,first=0.,active=0.;
    for(int n=lo;n<=hi;n++) {
        double w=exp(logweight(n,prior,b0,b1,s0,s1,y0,y1,factorial)-peak);
        total+=w;first+=n*w;if(n>0)active+=w;
    }
    out.ll=peak+log(total);out.lo=lo;out.hi=hi;out.mean=first/total;out.active=active/total;
    return out;
}
__device__ int sample_count(double prior,double b0,double b1,double s0,double s1,double y0,double y1,Integral integral,double u,const double* factorial) {
    if(integral.lo==-1)return 0;
    if(integral.lo==-3)return -3;
    if(integral.lo==-2) {
        double mean=integral.mean;
        if(mean<600.) {
            double probability=exp(-mean),cumulative=probability;int count=0;
            int maximum=max(128,int(mean+15.*sqrt(mean)+100.));
            while(cumulative<u&&count<maximum){++count;probability*=mean/count;cumulative+=probability;}
            return count;
        }
        // Same 1e-12 geometric support error bound used by the adapted count law.
        integral=integrate(mean,1.,1.,0.,0.,0.,0.,factorial,true);
        if(integral.lo<0)return -3;
        prior=mean;b0=b1=1.;s0=s1=y0=y1=0.;
    }
    double cumulative=0.;
    for(int n=integral.lo;n<=integral.hi;n++) {
        cumulative+=exp(logweight(n,prior,b0,b1,s0,s1,y0,y1,factorial)-integral.ll);
        if(cumulative>=u)return n;
    }
    return integral.hi;
}

extern "C" __global__ void filter_chunk(
    const double* y,const bool* valid,const double* bg,const int* target,const int* source,
    const double* mass,const double* decay,const int* asource,const double* adecay,
    const double* beta,const double* initial,const int* saturation,const double* zdecay,
    const double* zload,const double* variance,const U64* seeds,const int* counts,const int* segments,
    double* fast,double* slope,double* slow,double* sslope,double* pending,double* zstate,
    double* nextfast,double* nextslope,double* nextslow,double* nextsslope,double* nextpending,double* nextz,
    const double* initial_fast,const double* initial_slow,const double* factorial,
    double* temporary,double* output,long long* failed,const signed char* loss_weights,
    int n,int particles,int K,int A,int Q,int start,int stop,int reduce_loss) {
    const int b=blockIdx.x,p=threadIdx.x;
    const int k=counts[b*3],a=counts[b*3+1],qcount=counts[b*3+2];
    const U64 seed=seeds[b];
    __shared__ double weights[512],pred0[512],pred1[512],means[512],probabilities[512],resample_u;
    __shared__ int bad;
    const int begin=segments[b*2],length=segments[b*2+1]-begin;
    bool reset=start==0||!valid[begin+start-1];
    for(int t=start;t<stop;t++) {
        if(t>=length)break;
        if(failed[b]<n)break;
        const int data_t=begin+t;
        if(!valid[data_t]){reset=true;continue;}
        if(p==0)bad=0;
        U64 offset=U64(t)*particles*64+U64(p)*64;
        if(p<particles&&reset) {
            for(int j=0;j<k;j++) {
                fast[(b*particles+p)*K+j]=t==0?initial_fast[b*K+j]:initial[b*3+source[b*K+j]];
                slope[(b*particles+p)*K+j]=0.;
            }
            for(int j=0;j<a;j++) {
                slow[(b*particles+p)*A+j]=t==0?initial_slow[b*A+j]:initial[b*3+asource[b*A+j]];
                sslope[(b*particles+p)*A+j]=0.;
            }
            pending[b*particles+p]=0.;
            for(int q=0;q<qcount;q++)zstate[(b*particles+p)*Q+q]=normal53(seed,offset+q);
        }
        reset=false;
        __syncthreads();
        double y0=y[data_t*2],y1=y[data_t*2+1];
        if(p<particles) {
            double b0=0.,b1=0.,s0=0.,s1=0.;
            for(int j=0;j<k;j++) {
                int side=target[b*K+j];double g=mass[b*K+j];
                if(side==0){b0+=g*fast[(b*particles+p)*K+j];s0+=g*slope[(b*particles+p)*K+j];}
                if(side==1){b1+=g*fast[(b*particles+p)*K+j];s1+=g*slope[(b*particles+p)*K+j];}
            }
            double prior=pending[b*particles+p],bg0=bg[b*3],bg1=bg[b*3+1],bg2=bg[b*3+2],correction=0.;
            pred0[p]=bg0+b0+s0*prior;pred1[p]=bg1+b1+s1*prior;
            if(qcount) {
                double pm[4],sd[4],center[4]={0.,0.,0.,0.},precision[4]={1.,1.,1.,1.};
                double ex0=0.,ex1=0.;
                for(int q=0;q<qcount;q++) {
                    double d=zdecay[b*Q+q],c0=zload[(b*Q+q)*3],c1=zload[(b*Q+q)*3+1];
                    pm[q]=d*zstate[(b*particles+p)*Q+q];sd[q]=sqrt(1.-d*d);
                    ex0+=c0*pm[q]-.5*(c0*d)*(c0*d);ex1+=c1*pm[q]-.5*(c1*d)*(c1*d);
                }
                pred0[p]=bg0*exp(ex0)+b0+s0*prior;pred1[p]=bg1*exp(ex1)+b1+s1*prior;
                for(int it=0;it<9;it++) {
                    double gradient[4];
                    for(int q=0;q<qcount;q++){gradient[q]=-center[q];precision[q]=1.;}
                    for(int side=0;side<2;side++) {
                        double exponent=-.5*variance[b*3+side];
                        for(int q=0;q<qcount;q++)exponent+=zload[(b*Q+q)*3+side]*(pm[q]+sd[q]*center[q]);
                        double immigration=bg[b*3+side]*exp(exponent);
                        double rate=fmax((side==0?b0+s0*prior:b1+s1*prior)+immigration,1.e-300);
                        for(int q=0;q<qcount;q++) {
                            double c=zload[(b*Q+q)*3+side]*sd[q];
                            gradient[q]+=c*immigration*((side==0?y0:y1)/rate-1.);
                            precision[q]+=c*c*immigration*immigration/rate;
                        }
                    }
                    if(it==8)break;
                    double maximum=0.;
                    for(int q=0;q<qcount;q++) {
                        double delta=fmax(-2.,fmin(2.,gradient[q]/precision[q]));
                        if(qcount>1)delta*=.5;
                        center[q]+=delta;maximum=fmax(maximum,fabs(delta));
                    }
                    if(maximum<1.e-4)break;
                }
                bool prior_component=uniform53(seed,offset+24)<.25;
                double lp=0.,lg=0.,exponents[3]={-.5*variance[b*3],-.5*variance[b*3+1],-.5*variance[b*3+2]};
                for(int q=0;q<qcount;q++) {
                    double psd=1./sqrt(precision[q]),noise=normal53(seed,offset+8+q);
                    double innovation=prior_component?noise:center[q]+psd*noise;
                    double z=pm[q]+sd[q]*innovation,standard=(innovation-center[q])/psd;
                    zstate[(b*particles+p)*Q+q]=z;lp-=.5*innovation*innovation;lg-=.5*standard*standard+log(psd);
                    for(int side=0;side<3;side++)exponents[side]+=zload[(b*Q+q)*3+side]*z;
                }
                double left=log(.25)+lp,right=log(.75)+lg,peak=fmax(left,right);
                correction=lp-peak-log(exp(left-peak)+exp(right-peak));
                bg0*=exp(exponents[0]);bg1*=exp(exponents[1]);bg2*=exp(exponents[2]);
            }
            b0+=bg0;b1+=bg1;
            Integral integral=integrate(prior,b0,b1,s0,s1,y0,y1,factorial);
            if(integral.lo==-3||!isfinite(integral.ll+correction))atomicExch(&bad,1);
            weights[p]=integral.ll+correction;means[p]=integral.mean;probabilities[p]=integral.active;
            double* tmp=temporary+(b*particles+p)*10;
            tmp[0]=b0;tmp[1]=b1;tmp[2]=s0;tmp[3]=s1;tmp[4]=integral.ll;
            tmp[5]=integral.lo;tmp[6]=integral.hi;tmp[7]=integral.mean;tmp[8]=integral.active;tmp[9]=bg2;
        }
        __syncthreads();
        if(bad){if(p==0)failed[b]=t;break;}
        if(p==0) {
            double peak=-1./0.,sum=0.,pr0=0.,pr1=0.;
            for(int i=0;i<particles;i++)peak=fmax(peak,weights[i]);
            for(int i=0;i<particles;i++){weights[i]=exp(weights[i]-peak);sum+=weights[i];pr0+=pred0[i];pr1+=pred1[i];}
            double sumsq=0.,previous=0.,active=0.,cumulative=0.;
            for(int i=0;i<particles;i++) {
                double w=weights[i]/sum;sumsq+=w*w;previous+=w*means[i];active+=w*probabilities[i];
                cumulative+=w;weights[i]=cumulative;
            }
            double loss=-(peak+log(sum/particles));
            if(reduce_loss) {
                int weight=loss_weights[U64(b)*n+t];
                if(weight)output[b]+=weight*loss;
            } else {
                double* out=output+(U64(b)*n+t)*7;
                out[0]=pr0/particles;out[1]=pr1/particles;out[2]=loss;
                out[3]=1./sumsq;out[5]=previous;out[6]=active;
            }
            resample_u=uniform53(seed,U64(t)*particles*64+63)/particles;
        }
        __syncthreads();
        if(p<particles) {
            double threshold=resample_u+double(p)/particles;int lo=0,hi=particles-1;
            while(lo<hi){int mid=(lo+hi)/2;if(weights[mid]<threshold)lo=mid+1;else hi=mid;}
            int ancestor=lo;double* tmp=temporary+(b*particles+ancestor)*10;
            Integral integral={tmp[4],tmp[7],tmp[8],int(tmp[5]),int(tmp[6])};
            int count=sample_count(pending[b*particles+ancestor],tmp[0],tmp[1],tmp[2],tmp[3],y0,y1,integral,uniform53(seed,offset+25),factorial);
            if(count<0)atomicExch(&bad,1);
            double rate=tmp[9];
            for(int j=0;j<k;j++)if(target[b*K+j]==2)rate+=mass[b*K+j]*(fast[(b*particles+ancestor)*K+j]+slope[(b*particles+ancestor)*K+j]*count);
            nextpending[b*particles+p]=rate;
            for(int j=0;j<k;j++) {
                double factor=1.;
                for(int h=0;h<a;h++) {
                    double r=(slow[(b*particles+ancestor)*A+h]+sslope[(b*particles+ancestor)*A+h]*count)/fmax(initial[b*3+asource[b*A+h]],1.e-8);
                    factor+=beta[(b*K+j)*A+h]*(saturation[b]?tanh(r):r);
                }
                double d=decay[b*K+j],state=fast[(b*particles+ancestor)*K+j]+slope[(b*particles+ancestor)*K+j]*count;
                int s=source[b*K+j];
                nextfast[(b*particles+p)*K+j]=d*state+(s==2?0.:(1.-d)*factor*y[data_t*2+s]);
                nextslope[(b*particles+p)*K+j]=s==2?(1.-d)*factor:0.;
            }
            for(int h=0;h<a;h++) {
                int s=asource[b*A+h];double d=adecay[b*A+h];
                double state=slow[(b*particles+ancestor)*A+h]+sslope[(b*particles+ancestor)*A+h]*count;
                nextslow[(b*particles+p)*A+h]=d*state+(s==2?0.:(1.-d)*y[data_t*2+s]);
                nextsslope[(b*particles+p)*A+h]=s==2?1.-d:0.;
            }
            for(int q=0;q<qcount;q++)nextz[(b*particles+p)*Q+q]=zstate[(b*particles+ancestor)*Q+q];
        }
        __syncthreads();
        if(bad){if(p==0)failed[b]=t;break;}
        if(p<particles) {
            for(int j=0;j<k;j++){fast[(b*particles+p)*K+j]=nextfast[(b*particles+p)*K+j];slope[(b*particles+p)*K+j]=nextslope[(b*particles+p)*K+j];}
            for(int h=0;h<a;h++){slow[(b*particles+p)*A+h]=nextslow[(b*particles+p)*A+h];sslope[(b*particles+p)*A+h]=nextsslope[(b*particles+p)*A+h];}
            pending[b*particles+p]=nextpending[b*particles+p];means[p]=pending[b*particles+p];
            for(int q=0;q<qcount;q++)zstate[(b*particles+p)*Q+q]=nextz[(b*particles+p)*Q+q];
        }
        __syncthreads();
        if(p==0&&!reduce_loss){double rate=0.;for(int i=0;i<particles;i++)rate+=means[i];output[(U64(b)*n+t)*7+4]=rate/particles;}
        __syncthreads();
    }
}
