import sys,time,warnings,functools
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
import numpy as np, coremltools as ct
P="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/ve_grid.mlpackage"
T,L=int(sys.argv[1]),int(sys.argv[2])
t0=time.time()
m=ct.models.MLModel(P,compute_units=ct.ComputeUnit.CPU_AND_NE,function_name=f"T{T}_L{L}")
x={"noisy_latent":np.random.randn(1,144,L).astype(np.float32),"text_emb":np.random.randn(1,256,T).astype(np.float32),
   "style_ttl":np.random.randn(1,50,256).astype(np.float32),"latent_mask":np.ones((1,1,L),np.float32),
   "text_mask":np.ones((1,1,T),np.float32),"current_step":np.zeros(1,np.float32),"total_step":np.full(1,8.,np.float32)}
m.predict(x); load=time.time()-t0
t=[]
for _ in range(15):
    s=time.perf_counter(); m.predict(x); t.append((time.perf_counter()-s)*1000)
print(f"T{T}_L{L}: first-load+compile {load:6.2f}s   steady {np.median(t):7.2f} ms/step  ({np.median(t)*8:6.1f} ms for 8)")
