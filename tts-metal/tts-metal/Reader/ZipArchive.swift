//
//  ZipArchive.swift
//  tts-metal
//
//  Minimal read-only ZIP archive reader (stored + raw-deflate entries) used by the
//  EPUB and DOCX loaders. Uses Apple's Compression framework for DEFLATE — no
//  third-party dependencies.
//

import Foundation
import Compression

struct ZipEntry {
    let name: String
    let isDirectory: Bool
    let offset: UInt64      // offset of local file header in the archive
    let compressedSize: UInt64
    let uncompressedSize: UInt64
    let method: UInt16      // 0 = stored, 8 = deflate
    let crc32: UInt32
}

struct ZipArchive {
    let data: Data
    let entries: [String: ZipEntry]   // keyed by full path

    init?(data: Data) {
        self.data = data
        guard let eocd = ZipArchive.findEOCD(data) else { return nil }
        guard let parsed = ZipArchive.parseCentralDirectory(data, eocdEnd: eocd) else { return nil }
        self.entries = parsed
    }

    // MARK: - EOCD

    private static let eocdSignature: UInt32 = 0x06054b50

    private static func findEOCD(_ data: Data) -> Int? {
        let minSize = 22
        guard data.count >= minSize else { return nil }
        let start = max(0, data.count - 65_557)
        var idx = data.count - minSize
        while idx >= start {
            if readU32(data, idx) == eocdSignature { return idx }
            idx -= 1
        }
        return nil
    }

    private static func parseCentralDirectory(_ data: Data, eocdEnd: Int) -> [String: ZipEntry]? {
        let cdCount = Int(readU16(data, eocdEnd + 10))
        var cdOffset = Int(readU32(data, eocdEnd + 16))
        var out: [String: ZipEntry] = [:]
        out.reserveCapacity(cdCount)
        for _ in 0..<cdCount {
            guard readU32(data, cdOffset) == 0x02014b50 else { return out.isEmpty ? nil : out }
            let method = readU16(data, cdOffset + 10)
            let crc = readU32(data, cdOffset + 16)
            let csize = readU32(data, cdOffset + 20)
            let usize = readU32(data, cdOffset + 24)
            let nameLen = Int(readU16(data, cdOffset + 28))
            let extraLen = Int(readU16(data, cdOffset + 30))
            let commentLen = Int(readU16(data, cdOffset + 32))
            let lho = UInt64(readU32(data, cdOffset + 42))
            let nameData = data.subdata(in: (cdOffset + 46)..<(cdOffset + 46 + nameLen))
            let name = String(data: nameData, encoding: .utf8) ?? String(decoding: nameData, as: UTF8.self)
            if !name.hasSuffix("/") {
                out[name] = ZipEntry(name: name, isDirectory: false, offset: lho,
                                     compressedSize: UInt64(csize), uncompressedSize: UInt64(usize),
                                     method: method, crc32: crc)
            }
            cdOffset += 46 + nameLen + extraLen + commentLen
        }
        return out.isEmpty ? nil : out
    }

    // MARK: - Extraction

    func read(_ name: String) -> Data? {
        guard let entry = lookup(name) else { return nil }
        let lho = Int(entry.offset)
        guard data.count > lho + 30 else { return nil }
        let nameLen = Int(Self.readU16(data, lho + 26))
        let extraLen = Int(Self.readU16(data, lho + 28))
        let payloadStart = lho + 30 + nameLen + extraLen
        let payloadEnd = payloadStart + Int(entry.compressedSize)
        guard payloadEnd <= data.count else { return nil }
        let payload = data.subdata(in: payloadStart..<payloadEnd)

        switch entry.method {
        case 0:
            return payload
        case 8:
            return Self.inflate(payload, expectedSize: Int(entry.uncompressedSize))
        default:
            return nil
        }
    }

    /// Case-insensitive lookup that tolerates a leading slash.
    func lookup(_ name: String) -> ZipEntry? {
        if let e = entries[name] { return e }
        let trimmed = name.hasPrefix("/") ? String(name.dropFirst()) : name
        if let e = entries[trimmed] { return e }
        return entries.first { $0.key.lowercased() == trimmed.lowercased() }?.value
    }

    func first(where predicate: (ZipEntry) -> Bool) -> ZipEntry? {
        entries.values.first(where: predicate)
    }

    // MARK: - Inflate (raw deflate)

    static func inflate(_ input: Data, expectedSize: Int) -> Data? {
        let capacity = max(expectedSize, input.count * 8, 4096)
        var dst = [UInt8](repeating: 0, count: capacity)
        let src = [UInt8](input)
        let written = dst.withUnsafeMutableBufferPointer { dstPtr -> Int in
            src.withUnsafeBufferPointer { srcPtr -> Int in
                compression_decode_buffer(dstPtr.baseAddress!, dstPtr.count,
                                          srcPtr.baseAddress!, srcPtr.count,
                                          nil,
                                          COMPRESSION_ZLIB)
            }
        }
        guard written > 0 else { return nil }
        return Data(dst.prefix(written))
    }

    // MARK: - Little-endian readers

    static func readU16(_ d: Data, _ off: Int) -> UInt16 {
        let b = [UInt8](d.subdata(in: off..<(off + 2)))
        return UInt16(b[0]) | (UInt16(b[1]) << 8)
    }

    static func readU32(_ d: Data, _ off: Int) -> UInt32 {
        let b = [UInt8](d.subdata(in: off..<(off + 4)))
        return UInt32(b[0]) | (UInt32(b[1]) << 8) | (UInt32(b[2]) << 16) | (UInt32(b[3]) << 24)
    }
}
