//
//  OnnxParser.swift
//  tts-metal
//
//  Swift port of src/onnx.ts — minimal protobuf wire-format parser for ONNX initializers,
//  plus NPZ / NPY parsing for the voices archive.
//

import Foundation

struct OnnxTensor {
    let name: String
    let dims: [Int]
    let dataType: Int
    let rawData: Data
}

enum OnnxDtype {
    static let float32 = 1
    static let uint8 = 2
    static let int8 = 3
    static let int64 = 7
    static let float16 = 10
}

final class OnnxParser {
    private let buffer: Data

    init(_ data: Data) {
        self.buffer = data
    }

    /// Streams each initializer tensor to `onTensor` as it's parsed, rather
    /// than collecting the whole model into a `[String: OnnxTensor]` first —
    /// keeps the peak transient allocation to one tensor's size instead of a
    /// whole model's (parsing e.g. vector_estimator.onnx into a dictionary
    /// up front peaks at ~256MB of `Data` copies alive simultaneously).
    /// Callers should wrap `onTensor` in an `autoreleasepool` if it converts
    /// `rawData` through Foundation APIs that may allocate autoreleased
    /// objects, so each tensor's scratch memory is freed before the next one
    /// is parsed rather than accumulating until the loop's pool drains.
    func parseInitializers(onTensor: (OnnxTensor) throws -> Void) throws {
        let count = buffer.count
        guard let graphRange = findField(targetField: 7, start: 0, end: count) else {
            throw NSError(domain: "OnnxParser", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not find graph"])
        }
        var offset = graphRange.start
        while offset < graphRange.end {
            guard let tag = readTag(offset) else { break }
            if tag.fieldNumber == 5 && tag.wireType == 2 {
                let len = try readVarint(tag.dataStart)
                let tStart = len.end
                let tEnd = tStart + Int(len.value)
                if let tensor = try parseTensorProto(start: tStart, end: tEnd), !tensor.name.isEmpty {
                    try onTensor(tensor)
                }
                offset = tEnd
            } else {
                offset = try skipField(tag)
            }
        }
    }

    private func parseTensorProto(start: Int, end: Int) throws -> OnnxTensor? {
        var name = ""
        var dims: [Int] = []
        var dataType = 0
        var rawData: Data? = nil

        var offset = start
        while offset < end {
            guard let tag = readTag(offset) else { break }
            switch tag.fieldNumber {
            case 1: // dims
                if tag.wireType == 0 {
                    let v = try readVarint(tag.dataStart)
                    dims.append(Int(v.value))
                    offset = v.end
                } else if tag.wireType == 2 {
                    let len = try readVarint(tag.dataStart)
                    var pos = len.end
                    let packEnd = pos + Int(len.value)
                    while pos < packEnd {
                        let v = try readVarint(pos)
                        dims.append(Int(v.value))
                        pos = v.end
                    }
                    offset = packEnd
                } else {
                    offset = try skipField(tag)
                }
            case 2:
                let v = try readVarint(tag.dataStart)
                dataType = Int(v.value)
                offset = v.end
            case 4: // float_data
                if tag.wireType == 2 {
                    let len = try readVarint(tag.dataStart)
                    let start = len.end
                    let slice = buffer.subdata(in: start..<(start + Int(len.value)))
                    rawData = slice
                    offset = start + Int(len.value)
                } else {
                    offset = try skipField(tag)
                }
            case 5: // int32_data (varint packed)
                if tag.wireType == 2 {
                    let len = try readVarint(tag.dataStart)
                    let start = len.end
                    let packEnd = start + Int(len.value)
                    var values: [Int32] = []
                    var pos = start
                    while pos < packEnd {
                        let v = try readVarint(pos)
                        values.append(Int32(truncatingIfNeeded: Int(v.value)))
                        pos = v.end
                    }
                    var bytes = Data(count: values.count * 4)
                    bytes.withUnsafeMutableBytes { ptr in
                        let i32ptr = ptr.bindMemory(to: Int32.self).baseAddress!
                        for (i, v) in values.enumerated() { i32ptr[i] = v }
                    }
                    rawData = bytes
                    offset = packEnd
                } else if tag.wireType == 0 {
                    let v = try readVarint(tag.dataStart)
                    var bytes = Data(count: 4)
                    bytes.withUnsafeMutableBytes { ptr in
                        let i32 = ptr.bindMemory(to: Int32.self).baseAddress!
                        i32.pointee = Int32(truncatingIfNeeded: Int(v.value))
                    }
                    rawData = bytes
                    offset = v.end
                } else {
                    offset = try skipField(tag)
                }
            case 7: // int64_data
                if tag.wireType == 2 {
                    let len = try readVarint(tag.dataStart)
                    let start = len.end
                    rawData = buffer.subdata(in: start..<(start + Int(len.value)))
                    offset = start + Int(len.value)
                } else {
                    offset = try skipField(tag)
                }
            case 8: // name
                let len = try readVarint(tag.dataStart)
                let s = len.end
                name = String(data: buffer.subdata(in: s..<(s + Int(len.value))), encoding: .utf8) ?? ""
                offset = s + Int(len.value)
            case 9: // raw_data
                let len = try readVarint(tag.dataStart)
                let s = len.end
                rawData = buffer.subdata(in: s..<(s + Int(len.value)))
                offset = s + Int(len.value)
            default:
                offset = try skipField(tag)
            }
        }
        if name.isEmpty { return nil }
        return OnnxTensor(name: name, dims: dims, dataType: dataType,
                          rawData: rawData ?? Data())
    }

    private struct Field {
        let fieldNumber: Int
        let wireType: Int
        let dataStart: Int
    }

    private func readTag(_ offset: Int) -> Field? {
        guard offset < buffer.count else { return nil }
        let v = try? readVarint(offset)
        guard let v = v else { return nil }
        let tag = Int(v.value)
        return Field(fieldNumber: tag >> 3, wireType: tag & 0x7, dataStart: v.end)
    }

    private struct VarintResult { let value: UInt64; let end: Int }

    private func readVarint(_ offset: Int) throws -> VarintResult {
        var value: UInt64 = 0
        var shift: UInt64 = 0
        var pos = offset
        while pos < buffer.count {
            let byte = UInt64(buffer[pos])
            value |= (byte & 0x7f) << shift
            pos += 1
            if (byte & 0x80) == 0 { break }
            shift += 7
            if shift > 63 { break }
        }
        return VarintResult(value: value, end: pos)
    }

    private func skipField(_ field: Field) throws -> Int {
        switch field.wireType {
        case 0:
            return (try readVarint(field.dataStart)).end
        case 1:
            return field.dataStart + 8
        case 2:
            let len = try readVarint(field.dataStart)
            return len.end + Int(len.value)
        case 5:
            return field.dataStart + 4
        default:
            throw NSError(domain: "OnnxParser", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Unknown wire type: \(field.wireType)"])
        }
    }

    private func findField(targetField: Int, start: Int, end: Int) -> (start: Int, end: Int)? {
        var offset = start
        while offset < end {
            guard let tag = readTag(offset) else { break }
            if tag.fieldNumber == targetField && tag.wireType == 2 {
                let len = try? readVarint(tag.dataStart)
                guard let len = len else { return nil }
                return (len.end, len.end + Int(len.value))
            }
            offset = (try? skipField(tag)) ?? end
        }
        return nil
    }
}

// ── Dequantization helpers ──

enum OnnxDequant {
    static func float16ToFloat32(_ f16: Data) -> [Float] {
        let count = f16.count / 2
        var out = [Float](repeating: 0, count: count)
        f16.withUnsafeBytes { ptr in
            let u16 = ptr.bindMemory(to: UInt16.self).baseAddress!
            for i in 0..<count {
                let h = u16[i]
                let sign = (h >> 15) & 1
                let exp = (h >> 10) & 0x1f
                let frac = h & 0x3ff
                let f: Float
                if exp == 0 {
                    f = (sign == 0 ? 1 : -1) * pow(2.0, -14) * Float(frac) / 1024.0
                } else if exp == 31 {
                    f = frac == 0 ? (sign == 0 ? .infinity : -.infinity) : .nan
                } else {
                    f = (sign == 0 ? 1 : -1) * pow(2.0, Float(exp) - 15) * (1 + Float(frac) / 1024)
                }
                out[i] = f
            }
        }
        return out
    }

    static func float32Data(_ data: Data) -> [Float] {
        let count = data.count / 4
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { ptr in
            let f32 = ptr.bindMemory(to: Float.self).baseAddress!
            for i in 0..<count { out[i] = f32[i] }
        }
        return out
    }

    static func dequantInt8(_ data: Data, scale: [Float], zeroPoint: Int8?) -> [Float] {
        let s = scale.first ?? 1
        let zp = Float(Int(zeroPoint ?? 0))
        let count = data.count
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { ptr in
            let i8 = ptr.bindMemory(to: Int8.self).baseAddress!
            for i in 0..<count { out[i] = (Float(i8[i]) - zp) * s }
        }
        return out
    }

    static func dequantUint8(_ data: Data, scale: [Float], zeroPoint: UInt8?) -> [Float] {
        let s = scale.first ?? 1
        let zp = Float(Int(zeroPoint ?? 0))
        let count = data.count
        var out = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { ptr in
            let u8 = ptr.bindMemory(to: UInt8.self).baseAddress!
            for i in 0..<count { out[i] = (Float(u8[i]) - zp) * s }
        }
        return out
    }

    static func float16Array(_ data: Data) -> [Float] { float16ToFloat32(data) }
}

// ── NPZ / NPY parsing ──

struct NpyArray {
    let shape: [Int]
    let data: [Float]
}

enum NpzParser {
    static func parse(_ data: Data) -> [String: NpyArray] {
        var result: [String: NpyArray] = [:]
        var offset = 0
        let bytes = [UInt8](data)
        while offset < bytes.count - 4 {
            if bytes[offset] != 0x50 || bytes[offset + 1] != 0x4b ||
                bytes[offset + 2] != 0x03 || bytes[offset + 3] != 0x04 {
                break
            }
            let compressionMethod = readU16(data, offset + 8)
            var compressedSize = Int(readU32(data, offset + 18))
            let fileNameLen = Int(readU16(data, offset + 26))
            let extraLen = Int(readU16(data, offset + 28))
            let fileName = String(data: data.subdata(in: (offset + 30)..<(offset + 30 + fileNameLen)), encoding: .utf8) ?? ""

            if compressedSize == 0xFFFFFFFF {
                let baseExtra = offset + 30 + fileNameLen
                let extraEnd = baseExtra + extraLen
                var eo = baseExtra
                while eo + 4 <= extraEnd {
                    let headerId = readU16(data, eo)
                    let dataSize = Int(readU16(data, eo + 2))
                    if headerId == 0x0001 && dataSize >= 16 {
                        let lo = readU32(data, eo + 12)
                        let hi = readU32(data, eo + 16)
                        compressedSize = Int(UInt64(lo) + UInt64(hi) << 32)
                        break
                    }
                    eo += 4 + dataSize
                }
            }
            let dataStart = offset + 30 + fileNameLen + extraLen
            if fileName.hasSuffix(".npy") && compressionMethod == 0 {
                let npyData = data.subdata(in: dataStart..<(dataStart + compressedSize))
                if let parsed = parseNpy(npyData) {
                    let name = String(fileName.dropLast(".npy".count))
                    result[name] = parsed
                }
            }
            offset = dataStart + compressedSize
        }
        return result
    }

    private static func readU16(_ data: Data, _ at: Int) -> UInt16 {
        var v: UInt16 = 0
        data.subdata(in: at..<(at + 2)).withUnsafeBytes { ptr in
            v = ptr.load(as: UInt16.self)
        }
        return v
    }

    private static func readU32(_ data: Data, _ at: Int) -> UInt32 {
        var v: UInt32 = 0
        data.subdata(in: at..<(at + 4)).withUnsafeBytes { ptr in
            v = ptr.load(as: UInt32.self)
        }
        return v
    }

    static func parseNpy(_ data: Data) -> NpyArray? {
        let bytes = [UInt8](data)
        guard bytes.count > 10, bytes[0] == 0x93, bytes[1] == 0x4e else { return nil }
        let majorVersion = bytes[6]
        var headerLen = 0
        var headerStart = 0
        if majorVersion == 1 {
            headerLen = Int(readU16(data, 8))
            headerStart = 10
        } else {
            headerLen = Int(readU32(data, 8))
            headerStart = 12
        }
        let headerStr = String(data: data.subdata(in: headerStart..<(headerStart + headerLen)), encoding: .ascii) ?? ""
        let dataStart = headerStart + headerLen

        print("[NpzParser] header='\(headerStr)'")
        let shape = parseShape(headerStr)
        let dtype = parseDtype(headerStr)
        print("[NpzParser] shape=\(shape) dtype='\(dtype)'")
        let total: Int = shape.isEmpty ? 0 : shape.reduce(1, *)

        var floats: [Float] = []
        switch dtype {
        case "<f4", "=f4", "float32":
            let slice = data.subdata(in: dataStart..<(dataStart + total * 4))
            floats = OnnxDequant.float32Data(slice)
        case "<f2", "=f2", "float16":
            let slice = data.subdata(in: dataStart..<(dataStart + total * 2))
            floats = OnnxDequant.float16Array(slice)
        case "<i8", "=i8", "int64":
            let slice = data.subdata(in: dataStart..<(dataStart + total * 8))
            slice.withUnsafeBytes { ptr in
                let raw = ptr.bindMemory(to: Int64.self).baseAddress!
                for i in 0..<total { floats.append(Float(raw[i])) }
            }
        default:
            print("[NpzParser] Unsupported dtype: \(dtype)")
            return nil
        }
        return NpyArray(shape: shape, data: floats)
    }

    private static func parseShape(_ header: String) -> [Int] {
        guard let open = header.range(of: "shape"),
              let paren = header.range(of: "(", range: open.upperBound..<header.endIndex),
              let close = header.range(of: ")", range: paren.upperBound..<header.endIndex) else {
            return []
        }
        let inside = header[paren.upperBound..<close.lowerBound]
        let parts = inside.split(separator: ",").compactMap { s -> Int? in
            let trimmed = s.trimmingCharacters(in: .whitespaces)
            return trimmed.isEmpty ? nil : Int(trimmed)
        }
        return parts
    }

    private static func parseDtype(_ header: String) -> String {
        for key in ["descr", "dtype"] {
            guard let keyRange = header.range(of: key),
                  let colon = header.range(of: ":", range: keyRange.upperBound..<header.endIndex) else { continue }
            let afterColon = colon.upperBound..<header.endIndex
            for q in ["'", "\""] {
                if let q1 = header.range(of: q, range: afterColon),
                   let q2 = header.range(of: q, range: q1.upperBound..<header.endIndex) {
                    let v = String(header[q1.upperBound..<q2.lowerBound])
                    if !v.isEmpty { return v }
                }
            }
        }
        return "<f4"
    }
}