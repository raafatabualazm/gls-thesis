// Minimal per-partition timing microbenchmark (2-value, cycle-level timed eval) for the thesis partitioner configs.
// One CUDA block per partition; each block keeps a private copy of all net values/arrivals in shared memory,
// updates flops+PIs each cycle, then evaluates ITS partition's gates level by level (__syncthreads between levels).
// Per-block time via clock64() and %globaltimer. Values are checked against a host reference of the same eval.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
typedef long long ll;
static std::vector<ll> rd(FILE*f){ ll n; if(fread(&n,8,1,f)!=1){fprintf(stderr,"bad file\n");exit(1);} std::vector<ll> v(n); if(n) fread(v.data(),8,n,f); return v; }
__device__ __forceinline__ unsigned long long gtimer(){ unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

__global__ void kern(int N,int nflop,const int*flops,int npi,const int*pis,const int*vec,int ncyc,int loops,
  const int*nfan,const int*fan,const unsigned*tt,const int*dr,const int*df,const int*dpos,
  const int*poff,const int*gates,const int*loff,const int*lcnt_off,
  long long*clk,unsigned long long*ns,int*events,unsigned char*finalv){
  extern __shared__ unsigned char sm[];
  unsigned char*val=sm; int*arr=(int*)(sm+((N+15)/16)*16); unsigned char*tmp=(unsigned char*)(arr+N);
  int b=blockIdx.x;
  for(int i=threadIdx.x;i<N;i+=blockDim.x){val[i]=0;arr[i]=-1;}
  __shared__ int ev; if(threadIdx.x==0) ev=0;
  __syncthreads();
  long long c0=clock64(); unsigned long long t0=gtimer();
  int lb=lcnt_off[b], le=lcnt_off[b+1];
  for(int L=0;L<loops;L++) for(int c=0;c<ncyc;c++){
    for(int k=threadIdx.x;k<nflop;k+=blockDim.x){int i=flops[k]; int s=fan[i*6+dpos[i]]; tmp[k]= s>=0?val[s]:(s==-3);}
    __syncthreads();
    for(int i=threadIdx.x;i<N;i+=blockDim.x) arr[i]=-1;
    __syncthreads();
    for(int k=threadIdx.x;k<nflop;k+=blockDim.x){int i=flops[k]; unsigned char nv=tmp[k]; if(nv!=val[i]){val[i]=nv;arr[i]=nv?dr[i]:df[i];atomicAdd(&ev,1);} }
    for(int q=threadIdx.x;q<npi;q+=blockDim.x){int p=pis[q]; unsigned char nv=(unsigned char)vec[c*npi+q]; if(nv!=val[p]){val[p]=nv;arr[p]=0;atomicAdd(&ev,1);} }
    __syncthreads();
    for(int l=lb;l<le;l++){
      for(int g=loff[l]+threadIdx.x; g<loff[l+1]; g+=blockDim.x){
        int i=gates[g]; unsigned m=0; int t=-1;
        for(int j=0;j<nfan[i];j++){int s=fan[i*6+j]; unsigned bv= s>=0?val[s]:(s==-3); m|=bv<<j; if(s>=0&&arr[s]>t) t=arr[s];}
        unsigned char nv=(tt[i]>>m)&1;
        if(nv!=val[i]){val[i]=nv; arr[i]=t+(nv?dr[i]:df[i]); atomicAdd(&ev,1);}
      }
      __syncthreads();
    }
  }
  __syncthreads();
  long long c1=clock64(); unsigned long long t1=gtimer();
  if(threadIdx.x==0){clk[b]=c1-c0; ns[b]=t1-t0; events[b]=ev;}
  for(int g=poff[b]+threadIdx.x; g<poff[b+1]; g+=blockDim.x) finalv[g]=val[gates[g]];
}

int main(int argc,char**argv){
  if(argc<2){fprintf(stderr,"usage: partbench file.bin [loops] [reps] [threads]\n");return 1;}
  int loops=argc>2?atoi(argv[2]):1, reps=argc>3?atoi(argv[3]):21, nth=argc>4?atoi(argv[4]):128;
  FILE*f=fopen(argv[1],"rb"); auto H=rd(f); int N=H[0],nc=H[1],ncyc=H[2],npi=H[3],ncfg=H[4];
  auto kind=rd(f),nfan=rd(f),fan=rd(f),tt=rd(f),dr=rd(f),df=rd(f),dpos=rd(f),lvl=rd(f),pis=rd(f),vec=rd(f);
  std::vector<int> I(N),F(N*6),T(N),DR(N),DF(N),DP(N),PI(npi),V(vec.size()),FL; std::vector<unsigned> TT(N);
  for(int i=0;i<N;i++){I[i]=nfan[i];T[i]=kind[i];TT[i]=(unsigned)tt[i];DR[i]=dr[i];DF[i]=df[i];DP[i]=dpos[i]; if(kind[i]==1)FL.push_back(i);}
  for(int i=0;i<N*6;i++)F[i]=fan[i]; for(int i=0;i<npi;i++)PI[i]=pis[i]; for(size_t i=0;i<vec.size();i++)V[i]=vec[i];
  // host reference (global, unpartitioned) of the same eval
  std::vector<int> order; for(int i=0;i<N;i++) if(kind[i]==0||kind[i]==3) order.push_back(i);
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){return lvl[a]<lvl[b];});
  std::vector<unsigned char> val(N,0),old; std::vector<int> arr(N); long long refev=0;
  auto fv=[&](int s)->unsigned{return s>=0?val[s]:(s==-3);};
  for(int L=0;L<loops;L++) for(int c=0;c<ncyc;c++){ old=val; std::fill(arr.begin(),arr.end(),-1);
    for(int i:FL){int s=F[i*6+DP[i]]; unsigned char nv= s>=0?old[s]:(s==-3); if(nv!=val[i]){val[i]=nv;arr[i]=nv?DR[i]:DF[i];refev++;}}
    for(int q=0;q<npi;q++){int p=PI[q]; unsigned char nv=V[c*npi+q]; if(nv!=val[p]){val[p]=nv;arr[p]=0;refev++;}}
    for(int i:order){unsigned m=0;int t=-1; for(int j=0;j<I[i];j++){int s=F[i*6+j]; m|=fv(s)<<j; if(s>=0&&arr[s]>t)t=arr[s];}
      unsigned char nv=(TT[i]>>m)&1; if(nv!=val[i]){val[i]=nv;arr[i]=t+(nv?DR[i]:DF[i]);refev++;}}
  }
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr,0));
  int clkkhz=0; cudaDeviceGetAttribute(&clkkhz,cudaDevAttrClockRate,0);
  printf("device %s SMs=%d clockRate=%d kHz | %s N=%d cells=%d cycles=%d loops=%d reps=%d threads/block=%d host_ref_events=%lld\n",
         pr.name,pr.multiProcessorCount,clkkhz,argv[1],N,nc,ncyc,loops,reps,nth,refev);
  auto up=[&](const void*h,size_t n){void*d;CK(cudaMalloc(&d,n));CK(cudaMemcpy(d,h,n,cudaMemcpyHostToDevice));return d;};
  int*dI=(int*)up(I.data(),N*4),*dF=(int*)up(F.data(),N*24),*dDR=(int*)up(DR.data(),N*4),*dDF=(int*)up(DF.data(),N*4),*dDP=(int*)up(DP.data(),N*4);
  unsigned*dTT=(unsigned*)up(TT.data(),N*4); int*dFL=(int*)up(FL.data(),FL.size()*4+4),*dPI=(int*)up(PI.data(),npi*4+4),*dV=(int*)up(V.data(),V.size()*4+4);
  size_t shm=((N+15)/16)*16+N*4+FL.size()+16;
  for(int cf=0;cf<ncfg;cf++){
    auto sizes=rd(f),gl=rd(f); int k=sizes.size();
    std::vector<int> poff(k+1,0),G(gl.begin(),gl.end()),loff,lco(k+1,0);
    for(int b=0;b<k;b++)poff[b+1]=poff[b]+sizes[b];
    for(int b=0;b<k;b++){ int s=poff[b]; while(s<poff[b+1]){ loff.push_back(s); int lv=lvl[G[s]]; while(s<poff[b+1]&&lvl[G[s]]==lv) s++; } lco[b+1]=loff.size(); }
    loff.push_back(poff[k]);
    // drop flops/PIs from partition gate lists (they are updated by every block); keep them in finalv check
    int*dG=(int*)up(G.data(),G.size()*4+4),*dPO=(int*)up(poff.data(),(k+1)*4),*dLO=(int*)up(loff.data(),loff.size()*4),*dLC=(int*)up(lco.data(),(k+1)*4);
    // gates list must contain only comb/PO for evaluation: build filtered lists
    std::vector<int> G2,poff2(k+1,0),loff2,lco2(k+1,0);
    for(int b=0;b<k;b++){ for(int g=poff[b];g<poff[b+1];g++) if(kind[G[g]]==0||kind[G[g]]==3) G2.push_back(G[g]); poff2[b+1]=G2.size(); }
    for(int b=0;b<k;b++){ int s=poff2[b]; while(s<poff2[b+1]){ loff2.push_back(s); int lv=lvl[G2[s]]; while(s<poff2[b+1]&&lvl[G2[s]]==lv) s++; } lco2[b+1]=loff2.size(); }
    loff2.push_back(poff2[k]);
    cudaFree(dG);cudaFree(dPO);cudaFree(dLO);cudaFree(dLC);
    dG=(int*)up(G2.data(),G2.size()*4+4); dPO=(int*)up(poff2.data(),(k+1)*4); dLO=(int*)up(loff2.data(),loff2.size()*4); dLC=(int*)up(lco2.data(),(k+1)*4);
    long long*dclk; unsigned long long*dns; int*dev; unsigned char*dfv;
    CK(cudaMalloc(&dclk,k*8));CK(cudaMalloc(&dns,k*8));CK(cudaMalloc(&dev,k*4));CK(cudaMalloc(&dfv,G2.size()+1));
    std::vector<std::vector<long long>> C(k),NS(k); std::vector<float> tot; std::vector<int> evs(k); std::vector<unsigned char> fvv(G2.size());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for(int r=0;r<reps+1;r++){
      cudaEventRecord(e0);
      kern<<<k,nth,shm>>>(N,FL.size(),dFL,npi,dPI,dV,ncyc,loops,dI,dF,dTT,dDR,dDF,dDP,dPO,dG,dLO,dLC,dclk,dns,dev,dfv);
      cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
      float ms; cudaEventElapsedTime(&ms,e0,e1);
      std::vector<long long> c(k); std::vector<unsigned long long> n(k);
      CK(cudaMemcpy(c.data(),dclk,k*8,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(n.data(),dns,k*8,cudaMemcpyDeviceToHost));
      if(r==0) continue; // warm-up
      tot.push_back(ms); for(int b=0;b<k;b++){C[b].push_back(c[b]);NS[b].push_back(n[b]);}
    }
    CK(cudaMemcpy(evs.data(),dev,k*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(fvv.data(),dfv,G2.size(),cudaMemcpyDeviceToHost));
    int mism=0; for(int b=0;b<k;b++) for(int g=poff2[b];g<poff2[b+1];g++) if(fvv[g]!=val[G2[g]]) mism++;
    auto med=[](std::vector<long long> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    std::sort(tot.begin(),tot.end());
    std::vector<long long> mc(k),mn(k); long long mx=0,sum=0,mxn=0; for(int b=0;b<k;b++){mc[b]=med(C[b]);mn[b]=med(NS[b]);mx=std::max(mx,mc[b]);sum+=mc[b];mxn=std::max(mxn,mn[b]);}
    printf("CONFIG cfg%d k=%d evalgates=%zu launch_ms_median=%.4f block_clk_max=%lld block_clk_mean=%.0f imbalance(max/mean)=%.3f block_ns_max=%lld final_value_mismatch=%d/%zu\n",
           cf,k,G2.size(),tot[tot.size()/2],mx,(double)sum/k,mx/((double)sum/k),mxn,mism,G2.size());
    printf("  per_block_clk_median:"); for(int b=0;b<k;b++) printf(" %lld",mc[b]); printf("\n  per_block_ns_median:"); for(int b=0;b<k;b++) printf(" %lld",mn[b]);
    printf("\n  per_block_gates:"); for(int b=0;b<k;b++) printf(" %d",poff2[b+1]-poff2[b]); printf("\n  per_block_levels:"); for(int b=0;b<k;b++) printf(" %d",lco2[b+1]-lco2[b]);
    printf("\n  per_block_events:"); for(int b=0;b<k;b++) printf(" %d",evs[b]); printf("\n");
    cudaFree(dG);cudaFree(dPO);cudaFree(dLO);cudaFree(dLC);cudaFree(dclk);cudaFree(dns);cudaFree(dev);cudaFree(dfv);
  }
  return 0;
}
