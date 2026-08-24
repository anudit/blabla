//
//  MOBILoader.swift
//  tts-metal
//
//  MOBI/AZW parsing: PDB record table + PalmDOC LZ77 decompression, then HTML
//  conversion. KF8/AZW3 files are handled best-effort (text records are laid out
//  the same way). HUFF/CDIC-compressed files are not supported.
//  Ported from blabla's loadMOBI / decompressPalmDoc.
//

import Foundation

enum MOBILoader {

    static func load(url: URL, fileName: String, sourceID: String) throws -> ReaderDocument {
        guard let data = try? Data(contentsOf: url) else {
            throw LoaderError.failed("Could not read file.")
        }
        guard data.count > 132 else { throw LoaderError.failed("File too small to be a MOBI.") }

        let u16 = { (off: Int) -> UInt16 in ZipArchive.readU16(data, off) }
        let u32 = { (off: Int) -> UInt32 in ZipArchive.readU32(data, off) }

        // PDB header
        let numRecords = Int(u16(76))
        guard numRecords > 0, data.count >= 78 + numRecords * 8 else {
            throw LoaderError.failed("Invalid PDB header.")
        }
        var recordOffsets: [Int] = []
        for i in 0..<numRecords {
            let entryOff = 78 + i * 8
            recordOffsets.append(Int(u32(entryOff) & 0x00FF_FFFF))
        }
        guard let r0 = recordOffsets.first, data.count >= r0 + 16 else {
            throw LoaderError.failed("Invalid MOBI: missing record 0.")
        }

        // PalmDOC header (record 0)
        let compression = Int(u16(r0 + 0))
        let textLength = Int(u32(r0 + 4))
        let textRecordCount = Int(u16(r0 + 8))
        let encryption = Int(u16(r0 + 12))
        guard compression == 1 || compression == 2 else {
            throw LoaderError.failed(compression == 17_480
                ? "HUFF/CDIC-compressed MOBI is not supported."
                : "Unsupported MOBI compression type \(compression).")
        }
        if encryption != 0 {
            throw LoaderError.failed("This book is encrypted (DRM) and cannot be opened.")
        }

        // Concatenate + decompress text records 1..textRecordCount
        var htmlData = Data()
        htmlData.reserveCapacity(textLength)
        let lastRecord = min(1 + textRecordCount, recordOffsets.count)
        for rec in 1..<lastRecord {
            let start = recordOffsets[rec]
            let end = rec + 1 < recordOffsets.count ? recordOffsets[rec + 1] : data.count
            guard start < end, end <= data.count else { continue }
            let chunk = data.subdata(in: start..<end)
            switch compression {
            case 1:
                htmlData.append(chunk)
            case 2:
                if let d = decompressPalmDoc(chunk) { htmlData.append(d) }
            default:
                break
            }
            if htmlData.count >= textLength && textLength > 0 { break }
        }

        guard let html = String(data: htmlData, encoding: .utf8)
            ?? String(data: htmlData, encoding: .isoLatin1),
              !html.isEmpty else {
            throw LoaderError.failed("MOBI contained no readable text.")
        }

        // Strip trailing multimedia/extras records noise
        let cutIndex = html.range(of: "<!--END OF RECORD-->")?.lowerBound ?? html.endIndex
        let bodyHTML = String(html[html.startIndex..<cutIndex])

        let parser = HTMLToBlocks()
        let result = parser.parse(bodyHTML)
        var blocks = result.blocks
        if blocks.isEmpty { blocks = [DocBlock(content: .paragraph(stripTags(bodyHTML)))] }
        guard !blocks.isEmpty else { throw LoaderError.failed("MOBI contained no readable text.") }

        var title = result.title ?? fileName.replacingOccurrences(of: "\\.[^.]+$", with: "", options: .regularExpression)
        // Try the MOBI full-name field in record 0's MOBI header
        // (header id "MOBI" at +16; full-name offset at +84, length at +88).
        if data.count >= r0 + 92, String(data: data.subdata(in: (r0 + 16)..<(r0 + 20)), encoding: .ascii) == "MOBI",
           let nameOffset = optionalU32(data, r0 + 84), let nameSize = optionalU32(data, r0 + 88),
           Int(nameSize) > 0 {
            let s = r0 + Int(nameOffset), e = min(s + Int(nameSize), data.count)
            if s < e, let t = String(data: data.subdata(in: s..<e), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty {
                title = t
            }
        }

        return ReaderDocument(title: title, fileType: .mobi, sourceID: sourceID, fileName: title,
                              frontmatter: nil, blocks: blocks, outline: result.outline)
    }

    private static func optionalU16(_ d: Data, _ off: Int) -> UInt16? {
        off + 2 <= d.count ? ZipArchive.readU16(d, off) : nil
    }
    private static func optionalU32(_ d: Data, _ off: Int) -> UInt32? {
        off + 4 <= d.count ? ZipArchive.readU32(d, off) : nil
    }

    /// PalmDOC LZ77 decompression (port of blabla's decompressPalmDoc).
    static func decompressPalmDoc(_ input: Data) -> Data? {
        var out = [UInt8]()
        out.reserveCapacity(input.count * 4)
        let bytes = [UInt8](input)
        var i = 0
        while i < bytes.count {
            let b = bytes[i]
            i += 1
            if b == 0 {
                out.append(0)
            } else if b <= 8 {
                let n = Int(b)
                guard i + n <= bytes.count else { return out.isEmpty ? nil : Data(out) }
                out.append(contentsOf: bytes[i..<(i + n)])
                i += n
            } else if b <= 0x7F {
                out.append(b)
            } else if b <= 0xBF {
                guard i < bytes.count else { return out.isEmpty ? nil : Data(out) }
                let b2 = bytes[i]; i += 1
                let distance = ((Int(b) & 0x3F) << 3) | (Int(b2) >> 3)
                let length = (Int(b2) & 0x7) + 3
                guard distance > 0, distance <= out.count else { continue }
                var src = out.count - distance
                for _ in 0..<length {
                    out.append(out[src])
                    src += 1
                }
            } else {
                // 0xC0..0xFF: space + ASCII char
                out.append(0x20)
                out.append(b & 0x7F)
            }
        }
        return Data(out)
    }
}
