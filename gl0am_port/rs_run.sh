W=~/gls-work; export MAMBA_ROOT_PREFIX=$W/mamba; eval "$($W/bin/micromamba shell hook -s bash)"; micromamba activate gl0am
NV=$(python -c "import nvidia.cuda_nvrtc,os;print(os.path.dirname(nvidia.cuda_nvrtc.__file__)+'/lib')")
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$NV CUPY_NVCC_PATH=$CONDA_PREFIX/bin/nvcc NVCC_PREPEND_FLAGS="-ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export GL0AM_ZD_STORAGE=100000000 GL0AM_RESIM_STORAGE=300000000 GL0AM_INIT_SEQ_ZERO=1
R=$W/gl0am_run2; mkdir -p $R; cd $R
tar xzf "/mnt/c/Users/Raafat Abualazm/gls-work/gl0am_in2.tgz"
S=$W/GL0AM/GL0AM; python rs_patch.py $S/GL0AM.runSim.py
cp $S/GL0AM.cupy . ; cp $W/gl0am_run/kahypar_cut.ini .
K=8
run() { b=$1; dur=$2; per=$3; fe=$4; rep=$5
 echo "##### $b"
 /usr/bin/time -f "generateGraph wall_s=%e" python $S/GL0AM.generateGraph.py --block $b --graphPrecompile ${b}_GraphPrecompile.pkl.gz 2>&1 | tail -3
 /usr/bin/time -f "splitLargeFOs wall_s=%e" python $S/GL0AM.splitLargeFOs.py --combo_graph ${b}_fullSDF_DGLGraph --target_combo_graph ${b}_combo_DGLGraph 2>&1 | tail -2
 /usr/bin/time -f "partitioningSetup wall_s=%e" python $S/GL0AM.partitioningSetup.py --block $b --graph4 ${b}_combo_DGLGraph 2>&1 | grep -v 'note:' | tail -2
 python - $b $K <<'P'
import sys,kahypar
b,k=sys.argv[1],int(sys.argv[2]); L=open(b+'.hgr').read().split('\n'); h=L[0].split(); ne,nv=int(h[0]),int(h[1]); fmt=h[2] if len(h)>2 else '0'
nets=[];nw=[]
for l in L[1:1+ne]:
    x=[int(t) for t in l.split()]
    if fmt in('1','11'): nw.append(x[0]); x=x[1:]
    else: nw.append(1)
    nets.append([v-1 for v in x])
vw=[1]*nv
if fmt in('10','11'): vw=[int(l) for l in L[1+ne:1+ne+nv]]
idx=[0];fl=[]
for n in nets: fl+=n; idx.append(len(fl))
H=kahypar.Hypergraph(nv,ne,idx,fl,k,nw,vw); c=kahypar.Context(); c.loadINIconfiguration('kahypar_cut.ini'); c.setK(k); c.setEpsilon(0.03); c.setSeed(1); c.suppressOutput(True)
kahypar.partition(H,c); bl=[H.blockID(i) for i in range(nv)]
open(f'{b}.hgr.part.{k}','w').write('\n'.join(map(str,bl))+'\n')
print('kahypar verts',nv,'nets',ne)
P
 /usr/bin/time -f "partitioningPostprocessing wall_s=%e" python $S/GL0AM.partitioningPostprocessing.py --block $b --graph2 ${b}_update_DGLGraph --graph3 ${b}_sram_DGLGraph --graph4 ${b}_combo_DGLGraph --hMetisResults $b.hgr.part.$K --partitions $K --graphPrecompile ${b}_GraphPrecompile.pkl.gz --graph ${b}_fullSDF_DGLGraph 2>&1 | tail -2
 GL0AM_DUMP_WAVES=$R/${b}_waves /usr/bin/time -f "runSim wall_s=%e maxrss_kB=%M" python $S/GL0AM.runSim.py --block $b --instance_name $b --testname $b --cycles 1 --period $per --input_trace_file $b.waveforms.pkl.gz --graphPrecompile ${b}_GraphPrecompile.pkl.gz --graph ${b}_fullSDF_DGLGraph --graph2 ${b}_update_DGLGraph --graph3 ${b}_sram_DGLGraph --graph4 ${b}_combo_DGLGraph --timescale 1 --pin_net_file $b.pin_net --num_splits 32 --duration $dur --clk clk --first_edge $fe --num_of_sub_chunks 1 --zeroDelayReportSignals $rep --clkTreeFile $b.clkTree --partitionsFile $b.$K.partitions > ${b}_runSim.log 2>&1
 echo "runSim exit=$?"; grep -v "note:\|cells processed" ${b}_runSim.log | tail -40
 ls -la outGL0AM.$b.saif.gz ${b}_waves_0.npz 2>&1
}
run gcd 135000 5000 2500 "resp_msg[15:0]"
run acc8 10015000 5000 2500 "acc[7:0]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $R && tar czf "/mnt/c/Users/Raafat Abualazm/gls-work/gl0am_out2.tgz" outGL0AM.*.saif.gz *_waves_0.npz *_runSim.log 2>/dev/null; echo done
