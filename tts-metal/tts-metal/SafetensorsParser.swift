//
//  SafetensorsParser.swift
//  tts-metal
//
//  Minimal reader for HuggingFace `.safetensors` files. The format is:
//    [8-byte little-endian u64 header length][JSON header][raw tensor bytes]
//  The JSON header maps tensor name -> { dtype, shape, data_offsets:[begin,end] }
//  where offsets are relative to the start of the raw byte block. The RE-USE
//  checkpoint stores every tensor as F32, so this reader only needs to decode
//  F32 (with F16 handled for completeness), which keeps it far simpler than the
//  ONNX path used for the Kitten model.
//

import Foundation

struct SafeTensor {
    let shape: [Int]
    let data: [Float]
    var count: Int { shape.reduce(1, *) }
}

enum SafetensorsParser {

    /// Parse a safetensors blob into name -> tensor (F32 values, row-major).
    static func parse(_ blob: Data) throws -> [String: SafeTensor] {
        guard blob.count >= 8 else {
            throw err("file too small")
        }
        let headerLen: UInt64 = blob.prefix(8).withUnsafeBytes { $0.load(as: UInt64.self) }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerLen)
        guard headerEnd <= blob.count else { throw err("header length out of range") }

        let headerData = blob.subdata(in: headerStart..<headerEnd)
        guard let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw err("header is not a JSON object")
        }

        var tensors: [String: SafeTensor] = [:]
        tensors.reserveCapacity(json.count)

        // The raw byte region begins right after the header.
        try blob.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            let base = raw.baseAddress!
            for (name, value) in json {
                if name == "__metadata__" { continue }
                guard let entry = value as? [String: Any],
                      let dtype = entry["dtype"] as? String,
                      let shapeAny = entry["shape"] as? [Any],
                      let offsets = entry["data_offsets"] as? [Any],
                      offsets.count == 2 else {
                    continue
                }
                let shape = shapeAny.map { ($0 as? Int) ?? Int(($0 as? NSNumber)?.intValue ?? 0) }
                let begin = intVal(offsets[0])
                let end = intVal(offsets[1])
                let byteStart = headerEnd + begin
                let byteEnd = headerEnd + end
                guard byteEnd <= blob.count, byteEnd >= byteStart else {
                    throw err("tensor \(name) out of range")
                }
                let count = shape.isEmpty ? 1 : shape.reduce(1, *)
                let ptr = base + byteStart
                var floats = [Float](repeating: 0, count: count)

                switch dtype {
                case "F32":
                    floats.withUnsafeMutableBytes { dst in
                        memcpy(dst.baseAddress!, ptr, count * 4)
                    }
                case "F16":
                    let u16 = ptr.assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count { floats[i] = float16ToFloat(u16[i]) }
                case "F64":
                    let f64 = ptr.assumingMemoryBound(to: Double.self)
                    for i in 0..<count { floats[i] = Float(f64[i]) }
                default:
                    // Skip integer / bool tensors — RE-USE has none that we consume.
                    continue
                }
                tensors[name] = SafeTensor(shape: shape, data: floats)
            }
        }
        return tensors
    }

    private static func intVal(_ any: Any) -> Int {
        if let i = any as? Int { return i }
        if let n = any as? NSNumber { return n.intValue }
        return 0
    }

    private static func float16ToFloat(_ h: UInt16) -> Float {
        let sign = UInt32(h & 0x8000) << 16
        let exp = UInt32(h & 0x7C00) >> 10
        let mant = UInt32(h & 0x03FF)
        var bits: UInt32
        if exp == 0 {
            if mant == 0 {
                bits = sign
            } else {
                var e: UInt32 = 127 - 15 + 1
                var m = mant
                while (m & 0x0400) == 0 { m <<= 1; e -= 1 }
                m &= 0x03FF
                bits = sign | (e << 23) | (m << 13)
            }
        } else if exp == 0x1F {
            bits = sign | 0x7F800000 | (mant << 13)
        } else {
            bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13)
        }
        return Float(bitPattern: bits)
    }

    private static func err(_ msg: String) -> NSError {
        NSError(domain: "Safetensors", code: 1, userInfo: [NSLocalizedDescriptionKey: msg])
    }
}
