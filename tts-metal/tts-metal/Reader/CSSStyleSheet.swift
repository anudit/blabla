//
//  CSSStyleSheet.swift
//  tts-metal
//
//  Just enough CSS to recover what a book's stylesheet says about the text
//  itself: alignment, italic/roman, bold, and super/subscript. Publisher
//  exports carry almost none of this in the markup — an equation is
//  `<p class="x06-Equation-1P">`, centred only because the stylesheet says
//  `p.x06-Equation-1P { text-align: center }`, and a Greek variable is
//  `<i class="grk-ital">` — so without reading the CSS those all render as
//  plain justified body text.
//
//  Only simple selectors are understood (`tag`, `.class`, `tag.class`);
//  descendant/child combinators, pseudo-classes and attribute selectors are
//  skipped. That covers what InDesign, Calibre and Sigil actually emit.
//

import Foundation

struct CSSStyleSheet {

    enum Align { case justify, left, center, right }

    /// The declarations this reader acts on. `nil` = not specified, so a
    /// later or more specific rule only overrides what it actually sets.
    struct Declarations {
        var italic: Bool?          // font-style: italic/oblique → true, normal → false
        var bold: Bool?            // font-weight: bold/600+ → true, normal/≤500 → false
        var vertical: Int?         // vertical-align: super → 1, sub → -1, baseline → 0
        var align: Align?

        mutating func merge(_ o: Declarations) {
            if let v = o.italic { italic = v }
            if let v = o.bold { bold = v }
            if let v = o.vertical { vertical = v }
            if let v = o.align { align = v }
        }

        var isEmpty: Bool { italic == nil && bold == nil && vertical == nil && align == nil }
    }

    private var byTag: [String: Declarations] = [:]
    private var byClass: [String: Declarations] = [:]
    private var byTagClass: [String: Declarations] = [:]   // "p.x06-Equation-1P"

    static let empty = CSSStyleSheet()

    var isEmpty: Bool { byTag.isEmpty && byClass.isEmpty && byTagClass.isEmpty }

    init() {}

    init(css: String) { add(css: css) }

    /// Appends the rules of another stylesheet; later rules win, as in a
    /// document that links several sheets in order.
    mutating func add(css: String) {
        let src = RegexCache.replace(css, pattern: #"/\*[\s\S]*?\*/"#, with: " ")
        // `[^{}]+\{[^{}]*\}` matches only innermost rule bodies, so the rules
        // inside an @media block are read like top-level ones and the
        // @media prelude itself never matches.
        guard let re = RegexCache.regex(#"([^{}]+)\{([^{}]*)\}"#) else { return }
        let ns = src as NSString
        for m in re.matches(in: src, range: NSRange(location: 0, length: ns.length)) {
            let selectors = ns.substring(with: m.range(at: 1))
            let decls = Self.parseDeclarations(ns.substring(with: m.range(at: 2)))
            guard !decls.isEmpty else { continue }
            for raw in selectors.split(separator: ",") {
                let sel = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                guard !sel.isEmpty, !sel.hasPrefix("@"),
                      sel.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "." || $0 == "-" || $0 == "_" })
                else { continue }
                let parts = sel.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
                // Compound classes (".a.b") are rare enough to skip.
                switch parts.count {
                case 1: byTag[parts[0], default: Declarations()].merge(decls)
                case 2 where parts[0].isEmpty: byClass[parts[1], default: Declarations()].merge(decls)
                case 2: byTagClass[sel, default: Declarations()].merge(decls)
                default: continue
                }
            }
        }
    }

    /// Cascaded declarations for an element, lowest specificity first:
    /// tag rules, then class rules, then tag.class rules.
    func declarations(tag: String, classes: [String]) -> Declarations {
        var d = byTag[tag] ?? Declarations()
        for c in classes { if let r = byClass[c] { d.merge(r) } }
        for c in classes { if let r = byTagClass["\(tag).\(c)"] { d.merge(r) } }
        return d
    }

    static func parseDeclarations(_ body: String) -> Declarations {
        var d = Declarations()
        for decl in body.split(separator: ";") {
            guard let colon = decl.firstIndex(of: ":") else { continue }
            let prop = decl[..<colon].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let value = decl[decl.index(after: colon)...]
                .replacingOccurrences(of: "!important", with: "")
                .trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            switch prop {
            case "font-style":
                if value.hasPrefix("italic") || value.hasPrefix("oblique") { d.italic = true }
                else if value == "normal" { d.italic = false }
            case "font-weight":
                if value == "bold" || value == "bolder" { d.bold = true }
                else if value == "normal" || value == "lighter" { d.bold = false }
                else if let n = Int(value) { d.bold = n >= 600 }
            case "vertical-align":
                switch value {
                case "super", "top", "text-top": d.vertical = 1
                case "sub", "bottom", "text-bottom": d.vertical = -1
                case "baseline": d.vertical = 0
                default: break
                }
            case "text-align":
                switch value {
                case "center", "-webkit-center": d.align = .center
                case "right", "end", "-webkit-right": d.align = .right
                case "left", "start", "-webkit-left": d.align = .left
                case "justify": d.align = .justify
                default: break
                }
            default:
                break
            }
        }
        return d
    }
}
