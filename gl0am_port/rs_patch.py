import sys
p = sys.argv[1]; s = open(p, newline='').read(); nl = '\r\n' if '\r\n' in s else '\n'
def rep(a, b, cnt=1):
    global s
    assert s.count(a) == cnt, (a, s.count(a)); s = s.replace(a, b)
if 'GL0AM_ZD_STORAGE' not in s:
    rep("output_waveforms_storage = cp.full( 2500000000, dtype=cp.uint64,  fill_value = 0 )",
        "output_waveforms_storage = cp.full( int(os.environ.get('GL0AM_ZD_STORAGE', 2500000000)), dtype=cp.uint64,  fill_value = 0 ) #cuda12-port: sized by env for small GPUs")
    rep("new_waveforms_total = cp.full( (6000000000,), end_token, dtype=cp.uint32 )",
        "new_waveforms_total = cp.full( (int(os.environ.get('GL0AM_RESIM_STORAGE', 6000000000)),), end_token, dtype=cp.uint32 ) #cuda12-port: sized by env")
    rep("if ( append_length >= 6000000000) :", "if ( append_length >= new_waveforms_total.shape[0]) :")
    rep("current_logic_value = cp.full( (len(bg.nodes())+2,), fill_value=2, dtype=cp.int8)  ",
        "current_logic_value = cp.full( (len(bg.nodes())+2,), fill_value=2, dtype=cp.int8)  " + nl +
        "if os.environ.get('GL0AM_INIT_SEQ_ZERO'): current_logic_value[cp.asarray(nodes_list_gpu_seqs).astype(cp.int64)] = 0 #cuda12-port test hook: flops start at 0 like the reference sims")
    rep("saveRegisterWaveformsTotal[perDriverPerSplitPointersBegin[:,0]] = 9223372036854775808",
        "saveRegisterWaveformsTotal[perDriverPerSplitPointersBegin[:,0]] = 9223372036854775808" + nl +
        "if os.environ.get('GL0AM_INIT_SEQ_ZERO'): saveRegisterWaveformsTotal[perDriverPerSplitPointersBegin[:numOfRegisters,0]] = 0")
    rep("  update_saif()" + nl, "  update_saif()" + nl +
        "  if os.environ.get('GL0AM_DUMP_WAVES'): np.savez_compressed(os.environ['GL0AM_DUMP_WAVES'] + '_' + str(subchunk_id) + '.npz', w=new_waveforms_total[0:append_length].get(), s=bg.ndata['waveform_start'].cpu().numpy(), e=bg.ndata['waveform_end'].cpu().numpy()) #cuda12-port: waveform dump for validation" + nl)
open(p, 'w', newline='').write(s); print('patched', p)
