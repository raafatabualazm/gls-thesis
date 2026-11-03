// Per-partition timing microbenchmark (2-value, cycle-level timed eval) for the thesis partitioner configs.
// One CUDA block per partition, cooperative launch. Flops have one owner each (global double buffer);
// CLOSED configs (fan-in-closed, replicated) eval in shared memory with 1 grid.sync/cycle; GLOBAL configs
// (non-replicating: SimPart DSP, Wang) eval from global memory with a grid.sync per level (Lmax+2 syncs/cycle).
// Checked against a host reference: every comb gate, every flop, and the total event count.
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

// ---- Correct partitioned semantics (fix for the "every block updates all flops" bug) ----
// Each flop has exactly one owner block (host: the block holding its D driver, else round-robin).
// Flop state lives in a global double buffer FV/FA[2][nflop]; cycle c reads buffer c&1, owners write c+1.
// PIs are recomputed by every block from the stimulus (read-only, no ownership needed).
// mode 0 (CLOSED): every block's comb set is closed under comb fan-in and holds its flops' D cones
//   (replicating configs). Block evaluates privately in shared memory; 1 grid.sync per cycle.
// mode 1 (GLOBAL): non-replicating configs. All comb values/arrivals in global memory; gates are
//   evaluated by global level with a grid.sync after every level, then 2 syncs for the flop commit.
//   Cost: (Lmax+2) grid syncs per cycle; documented in the launch_ms column.
// Busy time per block = clock64 accumulated over compute phases only (excludes grid.sync waits).
#include <cooperative_groups.h>
namespace cg=cooperative_groups;
__global__ void kern(int mode,int N,int nflop,const int*flops,const int*fowner,int npi,const int*pis,const int*vec,int ncyc,int loops,
  const int*nfan,const int*fan,const unsigned*tt,const int*dr,const int*df,const int*dpos,
  const int*gates,const unsigned char*prim,const int*goff,int nlev,   // goff[b*(nlev+1)+l]: block b's gates of global level l
  unsigned char*FV,int*FA,unsigned char*gval,int*garr,
  long long*clk,unsigned long long*ns,int*events,unsigned char*finalv,const int*poff){
  cg::grid_group grid=cg::this_grid();
  extern __shared__ unsigned char sm[];
  int b=blockIdx.x,K=gridDim.x;
  unsigned char*val; int*arr;
  if(mode==0){ val=sm; arr=(int*)(sm+((N+15)/16)*16); } else { val=gval; arr=garr; }
  __shared__ int ev; __shared__ long long busy; if(threadIdx.x==0){ev=0;busy=0;} __syncthreads();
  if(mode==0) for(int i=threadIdx.x;i<N;i+=blockDim.x){val[i]=0;arr[i]=-1;}
  else for(int i=b*blockDim.x+threadIdx.x;i<N;i+=K*blockDim.x){val[i]=0;arr[i]=-1;}
  for(int k=b*blockDim.x+threadIdx.x;k<nflop;k+=K*blockDim.x){int i=flops[k],s=fan[i*6+dpos[i]]; unsigned char nv= s>=0?0:(s==-3); FV[k]=nv; FA[k]=nv?dr[i]:-1; if(nv&&fowner[k]==b)atomicAdd(&ev,1);} // cycle-0 flop update from the all-0 reset state, as the host ref
  grid.sync();
  unsigned long long t0=gtimer();
  const int*go=goff+b*(nlev+1);
  int cyc=0;
  for(int L=0;L<loops;L++) for(int c=0;c<ncyc;c++,cyc++){
    int cur=cyc&1;
    long long c0=clock64();
    // load flop state for this cycle (mode0: every block into shared; mode1: owners into global)
    for(int k=threadIdx.x;k<nflop;k+=blockDim.x){ if(mode==1&&fowner[k]!=b) continue; int i=flops[k]; val[i]=FV[cur*nflop+k]; arr[i]=FA[cur*nflop+k]; }
    // PIs (deterministic from stimulus); mode1: block 0 writes
    if(mode==0||b==0) for(int q=threadIdx.x;q<npi;q+=blockDim.x){int p=pis[q]; unsigned char nv=(unsigned char)vec[c*npi+q]; if(nv!=val[p]){val[p]=nv;arr[p]=0; if(b==0)atomicAdd(&ev,1);} else arr[p]=-1;}
    __syncthreads(); if(threadIdx.x==0) busy+=clock64()-c0;
    if(mode==1) grid.sync();
    for(int l=0;l<nlev;l++){
      long long c1=clock64();
      for(int g=go[l]+threadIdx.x; g<go[l+1]; g+=blockDim.x){
        int i=gates[g]; unsigned m=0; int t=-1;
        for(int j=0;j<nfan[i];j++){int s=fan[i*6+j]; unsigned bv= s>=0?val[s]:(s==-3); m|=bv<<j; if(s>=0&&arr[s]>t) t=arr[s];}
        unsigned char nv=(tt[i]>>m)&1;
        if(nv!=val[i]){val[i]=nv; arr[i]=t+(nv?dr[i]:df[i]); if(prim[g])atomicAdd(&ev,1);} else arr[i]=-1;
      }
      __syncthreads(); if(threadIdx.x==0) busy+=clock64()-c1;
      if(mode==1) grid.sync();
    }
    // owners commit next-cycle flop state from this cycle's D values
    long long c2=clock64();
    int nxt=cur^1;
    for(int k=threadIdx.x;k<nflop;k+=blockDim.x){ if(fowner[k]!=b) continue; int i=flops[k]; int s=fan[i*6+dpos[i]];
      unsigned char nv= s>=0?val[s]:(s==-3); unsigned char ov=FV[cur*nflop+k];
      FV[nxt*nflop+k]=nv; FA[nxt*nflop+k]= nv!=ov?(nv?dr[i]:df[i]):-1; if(nv!=ov&&cyc+1<loops*ncyc)atomicAdd(&ev,1); }
    __syncthreads(); if(threadIdx.x==0) busy+=clock64()-c2;
    grid.sync();   // FV[nxt] complete; in mode1 also all reads of flop vals in gval done before owners overwrite next cycle
  }
  unsigned long long t1=gtimer();
  if(threadIdx.x==0){clk[b]=busy; ns[b]=t1-t0; events[b]=ev;}
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
  size_t shm=((N+15)/16)*16+N*4+16;
  int nflop=FL.size(), Lmax=0; for(int i:order) Lmax=std::max(Lmax,(int)lvl[i]); int nlev=Lmax+1;
  std::vector<int> isComb(N,0); for(int i:order) isComb[i]=1;
  unsigned char*dFV; int*dFA; unsigned char*dgv; int*dga;
  CK(cudaMalloc(&dFV,2*nflop+1));CK(cudaMalloc(&dFA,8*nflop+4));CK(cudaMalloc(&dgv,N));CK(cudaMalloc(&dga,4*N));
  int nsm=pr.multiProcessorCount;
  for(int cf=0;cf<ncfg;cf++){
    auto sizes=rd(f),gl=rd(f); int k=sizes.size();
    std::vector<int> poff(k+1,0); for(int b=0;b<k;b++)poff[b+1]=poff[b]+sizes[b];
    // per-block comb gate sets
    std::vector<std::vector<int>> S(k); std::vector<std::vector<char>> in(k,std::vector<char>(N,0));
    long long tot_in=0;
    for(int b=0;b<k;b++) for(int g=poff[b];g<poff[b+1];g++){int i=gl[g]; if(isComb[i]&&!in[b][i]){in[b][i]=1;S[b].push_back(i);tot_in++;}}
    // closure check: every comb fan-in of a block's gate is in the block
    bool closed=true; for(int b=0;b<k&&closed;b++) for(int i:S[b]) for(int j=0;j<I[i];j++){int s=F[i*6+j]; if(s>=0&&isComb[s]&&!in[b][s]){closed=false;break;}}
    // flop owner: a block containing the D driver (if comb), else round-robin; closed also needs D cone in owner
    std::vector<int> own(nflop); std::vector<int> fcnt(k,0);
    for(int q=0;q<nflop;q++){int i=FL[q],s=F[i*6+DP[i]],o=-1;
      if(s>=0&&isComb[s]){ for(int b=0;b<k;b++) if(in[b][s]){o=b;break;} if(o<0) closed=false; }
      if(o<0){o=0;for(int b=1;b<k;b++) if(fcnt[b]<fcnt[o])o=b;} own[q]=o; fcnt[o]++; }
    // every comb gate must be evaluated by someone
    std::vector<char> cov(N,0); for(int b=0;b<k;b++) for(int i:S[b]) cov[i]=1; int uncovered=0; for(int i:order) if(!cov[i]) uncovered++;
    int mode=closed?0:1;
    if(mode==1){ // non-replicating eval: keep the first copy of each gate, add uncovered gates to block 0
      std::vector<char> seen(N,0); for(int b=0;b<k;b++){std::vector<int> t; for(int i:S[b]) if(!seen[i]){seen[i]=1;t.push_back(i);} S[b]=t;}
      for(int i:order) if(!seen[i]) S[0].push_back(i);
    } else if(uncovered){ // gates in no partition (e.g. cones not reaching a partitioned endpoint): add them and their comb fan-in cone to block 0 (replicated), keeping closure
      std::vector<int> st; for(int i:order) if(!cov[i]) st.push_back(i);
      while(!st.empty()){int i=st.back(); st.pop_back(); if(in[0][i]) continue; in[0][i]=1; S[0].push_back(i); tot_in++; for(int j=0;j<I[i];j++){int s=F[i*6+j]; if(s>=0&&isComb[s]&&!in[0][s]) st.push_back(s);}}
    }
    // build gate lists sorted by global level, goff[b][l], primary flags
    std::vector<int> G2,poff2(k+1,0),goff((size_t)k*(nlev+1)); std::vector<unsigned char> prim; std::vector<char> pseen(N,0); int blevels[64]={0};
    for(int b=0;b<k;b++){ auto v=S[b]; std::stable_sort(v.begin(),v.end(),[&](int x,int y){return lvl[x]<lvl[y];});
      size_t p=0; for(int l=0;l<nlev;l++){ goff[(size_t)b*(nlev+1)+l]=G2.size(); bool any=false; while(p<v.size()&&lvl[v[p]]==l){G2.push_back(v[p]); prim.push_back(!pseen[v[p]]); pseen[v[p]]=1; p++; any=true;} if(any&&b<64) blevels[b]++; }
      goff[(size_t)b*(nlev+1)+nlev]=G2.size(); poff2[b+1]=G2.size(); }
    int*dG=(int*)up(G2.data(),G2.size()*4+4),*dPO=(int*)up(poff2.data(),(k+1)*4),*dGO=(int*)up(goff.data(),goff.size()*4),*dOW=(int*)up(own.data(),nflop*4+4);
    unsigned char*dPR=(unsigned char*)up(prim.data(),prim.size()+1);
    long long*dclk; unsigned long long*dns; int*dev; unsigned char*dfv;
    CK(cudaMalloc(&dclk,k*8));CK(cudaMalloc(&dns,k*8));CK(cudaMalloc(&dev,k*4));CK(cudaMalloc(&dfv,G2.size()+1));
    size_t sh= mode==0?shm:16;
    if(sh>48*1024) CK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh));
    int perSM=0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM,kern,nth,sh)); if(perSM*nsm<k){fprintf(stderr,"cfg%d: k=%d exceeds co-resident blocks %d\n",cf,k,perSM*nsm);exit(1);}
    std::vector<std::vector<long long>> C(k),NS(k); std::vector<float> tot; std::vector<int> evs(k); std::vector<unsigned char> fvv(G2.size()),fl(2*nflop);
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int ifl=nflop,inpi=npi,incyc=ncyc,iloops=loops,iN=N;
    for(int r=0;r<reps+1;r++){
      void* args[]={&mode,&iN,&ifl,&dFL,&dOW,&inpi,&dPI,&dV,&incyc,&iloops,&dI,&dF,&dTT,&dDR,&dDF,&dDP,&dG,&dPR,&dGO,&nlev,&dFV,&dFA,&dgv,&dga,&dclk,&dns,&dev,&dfv,&dPO};
      cudaEventRecord(e0);
      CK(cudaLaunchCooperativeKernel((void*)kern,k,nth,args,sh,0));
      cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
      float ms; cudaEventElapsedTime(&ms,e0,e1);
      std::vector<long long> c(k); std::vector<unsigned long long> n(k);
      CK(cudaMemcpy(c.data(),dclk,k*8,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(n.data(),dns,k*8,cudaMemcpyDeviceToHost));
      if(r==0) continue; // warm-up
      tot.push_back(ms); for(int b=0;b<k;b++){C[b].push_back(c[b]);NS[b].push_back(n[b]);}
    }
    CK(cudaMemcpy(evs.data(),dev,k*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(fvv.data(),dfv,G2.size(),cudaMemcpyDeviceToHost)); CK(cudaMemcpy(fl.data(),dFV,2*nflop,cudaMemcpyDeviceToHost));
    int mism=0; for(size_t g=0;g<G2.size();g++) if(fvv[g]!=val[G2[g]]) mism++;
    // flop check: state for the cycle after the last one = ref flop update applied to final val
    int last=(loops*ncyc)&1, fmism=0; for(int q=0;q<nflop;q++){int i=FL[q],s=F[i*6+DP[i]]; unsigned char nv=s>=0?val[s]:(s==-3); if(fl[last*nflop+q]!=nv) fmism++;}
    long long gev=0; for(int b=0;b<k;b++) gev+=evs[b];
    auto med=[](std::vector<long long> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
    std::sort(tot.begin(),tot.end());
    std::vector<long long> mc(k),mn(k); long long mx=0,sum=0,mxn=0; for(int b=0;b<k;b++){mc[b]=med(C[b]);mn[b]=med(NS[b]);mx=std::max(mx,mc[b]);sum+=mc[b];mxn=std::max(mxn,mn[b]);}
    // ref events counted the same way: flop changes are committed one cycle later, so the final cycle's commit is extra and cycle 0's initial flop update is absent
    printf("CONFIG cfg%d k=%d uncovered_added=%d mode=%s grid_syncs_per_cycle=%d evalgates=%zu replication=%.3f launch_ms_median=%.4f busy_clk_max=%lld busy_clk_mean=%.0f imbalance(max/mean)=%.3f block_ns_max=%lld comb_mismatch=%d/%zu flop_mismatch=%d/%d events=%lld host_ref_events=%lld\n",
           cf,k,uncovered,mode?"GLOBAL":"CLOSED",mode?nlev+2:1,G2.size(),(double)tot_in/order.size(),tot[tot.size()/2],mx,(double)sum/k,mx/((double)sum/k),mxn,mism,G2.size(),fmism,nflop,gev,refev);
    printf("  per_block_busy_clk_median:"); for(int b=0;b<k;b++) printf(" %lld",mc[b]);
    printf("\n  per_block_gates:"); for(int b=0;b<k;b++) printf(" %d",poff2[b+1]-poff2[b]); printf("\n  per_block_levels:"); for(int b=0;b<k&&b<64;b++) printf(" %d",blevels[b]);
    printf("\n  per_block_flops:"); for(int b=0;b<k;b++) printf(" %d",fcnt[b]); printf("\n");
    cudaFree(dG);cudaFree(dPO);cudaFree(dGO);cudaFree(dOW);cudaFree(dPR);cudaFree(dclk);cudaFree(dns);cudaFree(dev);cudaFree(dfv);
  }
  return 0;
}
