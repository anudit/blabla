import onnx, onnxslim, numpy as np, warnings, functools, os
warnings.filterwarnings("ignore"); print=functools.partial(print,flush=True)
from onnx import helper, TensorProto, shape_inference

SRC="/Users/anudit/Documents/GitHub/tts-metal/tts-metal/tts-metal/supertonic/vector_estimator.onnx"
DST="/Users/anudit/Documents/GitHub/tts-metal/ane/ve_hoisted.onnx"
m = onnxslim.slim(SRC)
g = m.graph
CUT = "/vector_estimator/vector_field/time_encoder/Unsqueeze"
cut = next(n for n in g.node if n.name == CUT)
cut_out = cut.output[0]
print("cut node:", cut.name, "->", cut_out)

# infer its shape
inferred = shape_inference.infer_shapes(m)
vi = {v.name: v for v in list(inferred.graph.value_info)+list(inferred.graph.output)}
shp = [d.dim_value if d.HasField('dim_value') else d.dim_param for d in vi[cut_out].type.tensor_type.shape.dim]
print("time_emb shape:", shp)

# nodes to delete: the whole time_encoder cone feeding cut_out
prod = {o:n for n in g.node for o in n.output}
init = {i.name for i in g.initializer}
dead=set()
def mark(name):
    if name in init or name not in prod: return
    n = prod[name]
    if n.name in dead: return
    dead.add(n.name)
    for i in n.input: mark(i)
mark(cut_out)
print(f"deleting {len(dead)} time-encoder nodes")

keep = [n for n in g.node if n.name not in dead]
del g.node[:]; g.node.extend(keep)

# new input replaces cut_out
newin = helper.make_tensor_value_info("time_emb", TensorProto.FLOAT, shp)
for n in g.node:
    for i,x in enumerate(n.input):
        if x == cut_out: n.input[i] = "time_emb"
# drop now-unused current_step/total_step
used = {i for n in g.node for i in n.input}
keep_in = [i for i in g.input if i.name in used]
dropped = [i.name for i in g.input if i.name not in used]
del g.input[:]; g.input.extend(keep_in); g.input.append(newin)
print("dropped inputs:", dropped)
print("final inputs:", [i.name for i in g.input])

onnx.checker.check_model(m, full_check=False)
onnx.save(m, DST)
print("saved", DST, os.path.getsize(DST)//1024//1024, "MB")
