#!/usr/bin/env python3
"""
Pure-Python (no deps) ONNX protobuf introspector.

Dumps, for each model:
  - graph inputs / outputs (name, elem_type, shape)
  - initializers (name, dtype, dims)  -> the weights the Metal port must load by name
  - nodes in topological order (op_type, name, inputs, outputs, key attributes)

We only decode the subset of the ONNX / protobuf wire format we need. Enough to
drive a hand-written Metal port: we need exact tensor names, shapes, and the op graph.

Usage:
  python3 tools/onnx_introspect.py supertonic-3/onnx/text_encoder.onnx
  python3 tools/onnx_introspect.py supertonic-3/onnx/*.onnx --summary
"""
import sys
import struct

# ---- protobuf primitives ---------------------------------------------------

def read_varint(buf, pos):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos

def read_tag(buf, pos):
    tag, pos = read_varint(buf, pos)
    return tag >> 3, tag & 0x7, pos

def skip(buf, pos, wire):
    if wire == 0:
        _, pos = read_varint(buf, pos)
    elif wire == 1:
        pos += 8
    elif wire == 2:
        ln, pos = read_varint(buf, pos)
        pos += ln
    elif wire == 5:
        pos += 4
    else:
        raise ValueError(f"bad wire {wire}")
    return pos

def read_len(buf, pos):
    ln, pos = read_varint(buf, pos)
    return buf[pos:pos + ln], pos + ln

def fields(buf):
    """Yield (field_number, wire_type, value) where value is bytes/int/float raw."""
    pos = 0
    n = len(buf)
    while pos < n:
        fn, wire, pos = read_tag(buf, pos)
        if wire == 0:
            v, pos = read_varint(buf, pos)
        elif wire == 1:
            v = struct.unpack_from('<d', buf, pos)[0]; pos += 8
        elif wire == 2:
            v, pos = read_len(buf, pos)
        elif wire == 5:
            v = struct.unpack_from('<f', buf, pos)[0]; pos += 4
        else:
            raise ValueError(f"bad wire {wire}")
        yield fn, wire, v

# ---- ONNX message decoders -------------------------------------------------

DTYPE = {0:"UNDEF",1:"float32",2:"uint8",3:"int8",4:"uint16",5:"int16",
         6:"int32",7:"int64",8:"string",9:"bool",10:"float16",11:"float64",
         12:"uint32",13:"uint64",14:"complex64",16:"bfloat16"}

def dec_tensor_shape(buf):
    dims = []
    for fn, wire, v in fields(buf):
        if fn == 1 and wire == 2:  # dimension
            d = None
            for dfn, dwire, dv in fields(v):
                if dfn == 1:  # dim_value
                    d = dv
                elif dfn == 2 and dwire == 2:  # dim_param (symbolic)
                    d = dv.decode('utf-8', 'replace')
            dims.append(d)
    return dims

def dec_type(buf):
    # TypeProto -> tensor_type (field 1)
    elem = None; shape = []
    for fn, wire, v in fields(buf):
        if fn == 1 and wire == 2:  # tensor_type
            for tfn, twire, tv in fields(v):
                if tfn == 1:
                    elem = tv
                elif tfn == 2 and twire == 2:
                    shape = dec_tensor_shape(tv)
    return DTYPE.get(elem, elem), shape

def dec_value_info(buf):
    name = ""; typ = (None, [])
    for fn, wire, v in fields(buf):
        if fn == 1 and wire == 2:
            name = v.decode('utf-8', 'replace')
        elif fn == 2 and wire == 2:
            typ = dec_type(v)
    return name, typ

def dec_tensor(buf):
    # TensorProto: 1=dims(varint,repeated), 2=data_type, 8=name
    dims = []; dtype = None; name = ""
    for fn, wire, v in fields(buf):
        if fn == 1 and wire == 0:
            dims.append(v)
        elif fn == 2 and wire == 0:
            dtype = v
        elif fn == 8 and wire == 2:
            name = v.decode('utf-8', 'replace')
    return name, DTYPE.get(dtype, dtype), dims

def dec_attr(buf):
    # AttributeProto: 1=name,2=f,3=i,4=s,7=floats,8=ints,20=type
    name = ""; atype = None; val = None
    for fn, wire, v in fields(buf):
        if fn == 1 and wire == 2:
            name = v.decode('utf-8', 'replace')
        elif fn == 2:  # f
            val = v
        elif fn == 3:  # i
            val = v
        elif fn == 4 and wire == 2:  # s
            val = v.decode('utf-8', 'replace')
        elif fn == 7 and wire == 2:  # floats packed
            val = list(struct.unpack('<%df' % (len(v)//4), v))
        elif fn == 8 and wire == 2:  # ints packed
            ints = []; p = 0
            while p < len(v):
                iv, p = read_varint(v, p); ints.append(iv)
            val = ints
    return name, val

def dec_node(buf):
    inputs = []; outputs = []; op = ""; name = ""; attrs = {}
    for fn, wire, v in fields(buf):
        if fn == 1 and wire == 2:
            inputs.append(v.decode('utf-8', 'replace'))
        elif fn == 2 and wire == 2:
            outputs.append(v.decode('utf-8', 'replace'))
        elif fn == 3 and wire == 2:
            name = v.decode('utf-8', 'replace')
        elif fn == 4 and wire == 2:
            op = v.decode('utf-8', 'replace')
        elif fn == 5 and wire == 2:
            an, av = dec_attr(v)
            attrs[an] = av
    return dict(op=op, name=name, inputs=inputs, outputs=outputs, attrs=attrs)

def dec_graph(buf):
    inits = []; nodes = []; ins = []; outs = []
    for fn, wire, v in fields(buf):
        if fn == 1 and wire == 2:      # node
            nodes.append(dec_node(v))
        elif fn == 5 and wire == 2:    # initializer
            inits.append(dec_tensor(v))
        elif fn == 11 and wire == 2:   # input
            ins.append(dec_value_info(v))
        elif fn == 12 and wire == 2:   # output
            outs.append(dec_value_info(v))
    return dict(inits=inits, nodes=nodes, inputs=ins, outputs=outs)

def load_model(path):
    with open(path, 'rb') as f:
        data = f.read()
    graph = None
    for fn, wire, v in fields(data):
        if fn == 7 and wire == 2:  # ModelProto.graph
            graph = dec_graph(v)
    return graph

# ---- reporting -------------------------------------------------------------

def report(path, summary=False):
    g = load_model(path)
    print("=" * 78)
    print(path)
    print("=" * 78)
    print("\n-- INPUTS --")
    for name, (dt, shp) in g['inputs']:
        print(f"  {name:30s} {dt} {shp}")
    print("\n-- OUTPUTS --")
    for name, (dt, shp) in g['outputs']:
        print(f"  {name:30s} {dt} {shp}")

    ops = {}
    for nd in g['nodes']:
        ops[nd['op']] = ops.get(nd['op'], 0) + 1
    print(f"\n-- OP HISTOGRAM ({len(g['nodes'])} nodes) --")
    for op, c in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {op}")

    print(f"\n-- INITIALIZERS ({len(g['inits'])}) --")
    init_names = set()
    for name, dt, dims in g['inits']:
        init_names.add(name)
        if not summary:
            print(f"  {name:55s} {dt:8s} {dims}")
    # param count
    total = 0
    for name, dt, dims in g['inits']:
        n = 1
        for d in dims: n *= d
        total += n
    print(f"  ~params: {total:,}")

    if not summary:
        print(f"\n-- NODES --")
        for i, nd in enumerate(g['nodes']):
            a = {k: v for k, v in nd['attrs'].items()
                 if k in ('kernel_shape','strides','pads','dilations','group','axis','perm','epsilon','alpha','to')}
            ins = [x for x in nd['inputs'] if x not in init_names]
            wts = [x for x in nd['inputs'] if x in init_names]
            print(f"  [{i:4d}] {nd['op']:22s} in={ins} w={wts} -> {nd['outputs']} {a if a else ''}")

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    summary = '--summary' in sys.argv
    for p in args:
        report(p, summary=summary)
