"""Build the bundled Core ML embedding model from potion-base-8M/model.safetensors.
Build-only requirement: coremltools and numpy. The app has no Python dependency.
"""
from pathlib import Path
import json
import struct
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'potion-base-8M' / 'model.safetensors'
DEST = ROOT / 'tts-metal' / 'tts-metal' / 'Embedding' / 'PotionEmbedding.mlpackage'
with SOURCE.open('rb') as file:
    header_size = struct.unpack('<Q', file.read(8))[0]
    header = json.loads(file.read(header_size))
    info = header['embeddings']
    assert info['dtype'] == 'F32' and info['shape'] == [29528, 256]
    file.seek(8 + header_size + info['data_offsets'][0])
    weights = np.frombuffer(file.read(29528 * 256 * 4), dtype='<f4').reshape(29528, 256).copy()

# Model2vec mean-pools embeddings then L2-normalizes. A token-count vector
# multiplied by the exact embedding table computes the same direction. The
# dense linear layer and normalization are scheduled on ANE; Swift performs tokenization.
weights = weights.T.astype(np.float16)
@mb.program(input_specs=[mb.TensorSpec(shape=(1, 29528), dtype=types.fp16)],
            opset_version=ct.target.macOS15)
def program(token_counts):
    outputs = []
    for start in range(0, 29528, 4096):
        end = min(start + 4096, 29528)
        counts = mb.slice_by_index(x=token_counts, begin=[0, start], end=[1, end])
        outputs.append(mb.linear(x=counts, weight=weights[:, start:end],
                                 bias=np.zeros(256, dtype=np.float16)))
    value = outputs[0]
    for part in outputs[1:]:
        value = mb.add(x=value, y=part)
    squared = mb.square(x=value)
    magnitude = mb.reduce_sum(x=squared, axes=[1], keep_dims=True)
    safe = mb.maximum(x=magnitude, y=np.float16(1e-8))
    inverse = mb.rsqrt(x=safe)
    return mb.mul(x=value, y=inverse, name='embedding')

model = ct.convert(program, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
                   compute_precision=ct.precision.FLOAT16,
                   inputs=[ct.TensorType(name='token_counts')],
                   compute_units=ct.ComputeUnit.CPU_AND_NE)
if DEST.exists():
    import shutil
    shutil.rmtree(DEST)
model.save(str(DEST))
print(DEST)
