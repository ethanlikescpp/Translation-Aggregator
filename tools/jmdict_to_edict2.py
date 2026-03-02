#!/usr/bin/env python3
"""
jmdict_to_edict2.py — Convert JMdict XML to EDICT2 format for use with JParser.

JMdict provides much richer dictionary data than the older EDICT2:
  - More entries (~200k vs ~170k)
  - Better part-of-speech tagging
  - More nuanced sense/gloss grouping
  - Frequency information (news/ichi/spec/gai rankings)

Usage:
    python3 jmdict_to_edict2.py [JMdict_e.xml | JMdict_e.gz] [output.edict2]

If no arguments given, looks for JMdict_e.gz or JMdict_e.xml in the current
directory and writes jmdict.edict2 alongside it.

The output file can be placed in Translation Aggregator's "dictionaries" folder.
TA will automatically compile it to a .bin on first run (same as edict2).

Dependencies: Python 3.6+, no third-party packages required.
"""

import sys
import os
import gzip
import re
import xml.etree.ElementTree as ET
from typing import Optional

# ---------------------------------------------------------------------------
# POS tag mapping: JMdict entity → EDICT2-style annotation
# This lets JParser recognise verb types for conjugation.
# ---------------------------------------------------------------------------
POS_MAP = {
    # Ichidan (ru-verbs)
    "v1":    "v1",
    "v1-s":  "v1",
    # Godan (u-verbs)
    "v5aru": "v5r-i",
    "v5b":   "v5b",
    "v5g":   "v5g",
    "v5k":   "v5k",
    "v5k-s": "v5k-s",
    "v5m":   "v5m",
    "v5n":   "v5n",
    "v5r":   "v5r",
    "v5r-i": "v5r-i",
    "v5s":   "v5s",
    "v5t":   "v5t",
    "v5u":   "v5u",
    "v5u-s": "v5u-s",
    "v5uru": "v5u",
    # Irregular
    "vk":    "vk",      # kuru
    "vs":    "vs",      # suru (generic)
    "vs-i":  "vs-i",    # suru (compound)
    "vs-s":  "vs-s",
    "vz":    "vz",      # zuru (literary -ずる)
    "vn":    "vn",
    "vr":    "vr",
    # Adjectives
    "adj-i": "adj-i",   # い-adjective
    "adj-na":"adj-na",  # な-adjective
    "adj-no":"adj",
    "adj-pn":"adj",
    "adj-t": "adj",
    "adj-f": "adj",
    # Nouns
    "n":     "n",
    "n-adv": "adv-n",
    "n-t":   "n",
    "n-suf": "suf",
    "n-pref":"pref",
    # Other
    "adv":   "adv",
    "conj":  "conj",
    "exp":   "exp",
    "int":   "int",
    "prt":   "prt",
    "pref":  "pref",
    "suf":   "suf",
    "ctr":   "ctr",
}

# Frequency markers to emit the (P) common-word flag.
# JMdict uses ke_pri/re_pri elements; "ichi1","news1","spec1","gai1" = common.
COMMON_PRI = {"ichi1", "news1", "spec1", "gai1"}


def open_jmdict(path: str):
    """Open JMdict file, handling both plain XML and .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def strip_entity(text: str) -> str:
    """Strip XML entity names like &v1; → v1"""
    return text.strip("&;")


def parse_entry(entry: ET.Element):
    """
    Parse a single <entry> element and yield EDICT2-style lines.

    EDICT2 format:
        kanji [reading] /(tag) gloss1/gloss2/.../EntLnnnnnn/
    For kana-only words:
        reading /(tag) gloss1/gloss2/.../EntLnnnnnn/
    """
    # --- Collect kanji elements ---
    kanji_elements = []
    for k_ele in entry.findall("k_ele"):
        keb = k_ele.findtext("keb", "").strip()
        if not keb:
            continue
        pri_set = {p.text for p in k_ele.findall("ke_pri") if p.text}
        common = bool(pri_set & COMMON_PRI)
        kanji_elements.append((keb, common))

    # --- Collect reading elements ---
    reading_elements = []
    for r_ele in entry.findall("r_ele"):
        reb = r_ele.findtext("reb", "").strip()
        if not reb:
            continue
        # re_nokanji: this reading is itself the "word form" (kana-only entry)
        no_kanji = r_ele.find("re_nokanji") is not None
        pri_set = {p.text for p in r_ele.findall("re_pri") if p.text}
        common = bool(pri_set & COMMON_PRI)
        # re_restr restricts this reading to specific kanji
        restrictions = [r.text for r in r_ele.findall("re_restr") if r.text]
        reading_elements.append((reb, common, no_kanji, restrictions))

    # --- Collect senses ---
    # Group by POS run (JMdict POS tags are "sticky" — once set they carry
    # forward to subsequent senses until changed).
    senses = []
    current_pos = []
    for sense in entry.findall("sense"):
        pos_nodes = sense.findall("pos")
        if pos_nodes:
            # New POS group — decode entity names from text
            current_pos = []
            for p in pos_nodes:
                # ElementTree turns &v1; into the entity expansion; we need
                # the entity name.  They're stored as plain text in lxml but
                # as entity references in stdlib ET depending on DOCTYPE.
                # We handle both: if text looks like a known key, use it.
                raw = (p.text or "").strip().strip("&;")
                current_pos.append(raw)

        # Stage-kana / dialect / field restrictions (informational only)
        misc = [m.text for m in sense.findall("misc") if m.text]
        stagk = [s.text for s in sense.findall("stagk") if s.text]
        stagr = [s.text for s in sense.findall("stagr") if s.text]

        glosses = [g.text for g in sense.findall("gloss")
                   if g.text and g.get("{http://www.w3.org/XML/1998/namespace}lang", "eng") == "eng"]
        if not glosses:
            continue

        senses.append({
            "pos": list(current_pos),
            "misc": misc,
            "stagk": stagk,
            "stagr": stagr,
            "glosses": glosses,
        })

    if not senses:
        return []

    ent_seq = entry.findtext("ent_seq", "0").strip()

    lines = []

    # Build the English definition string shared across all headword combos.
    # Format:  /(pos_tag) gloss1/gloss2/(pos_tag2) gloss3/.../EntLnnnn/
    def build_eng(sense_list):
        parts = []
        for s in sense_list:
            pos_tags = []
            for p in s["pos"]:
                edict_tag = POS_MAP.get(p, p)
                if edict_tag:
                    pos_tags.append(f"({edict_tag})")
            misc_tags = [f"({m})" for m in s["misc"]]
            prefix = " ".join(pos_tags + misc_tags)
            for gloss in s["glosses"]:
                if prefix:
                    parts.append(f"{prefix} {gloss}")
                else:
                    parts.append(gloss)
        if not parts:
            return None
        return "/" + "/".join(parts) + f"/EntL{ent_seq}/"

    # Determine commonness for full entry
    any_common_k = any(c for _, c in kanji_elements)
    any_common_r = any(c for _, c, _, _ in reading_elements)
    is_common = any_common_k or any_common_r

    def common_flag(flag: bool) -> str:
        return "/(P)" if flag else ""

    if kanji_elements:
        for keb, k_common in kanji_elements:
            # Find readings that apply to this kanji form
            readings = [
                (reb, r_common)
                for reb, r_common, no_kanji, restrictions in reading_elements
                if not no_kanji and (not restrictions or keb in restrictions)
            ]
            if not readings:
                # Fall back to all non-restricted readings
                readings = [
                    (reb, r_common)
                    for reb, r_common, no_kanji, restrictions in reading_elements
                    if not restrictions
                ]

            # Filter senses that apply to this kanji form
            relevant_senses = [
                s for s in senses
                if not s["stagk"] or keb in s["stagk"]
            ]
            if not relevant_senses:
                relevant_senses = senses

            eng = build_eng(relevant_senses)
            if not eng:
                continue

            common = k_common or any(rc for _, rc in readings)
            p_flag = "/(P)" if common else ""

            if readings:
                reading_str = "; ".join(reb for reb, _ in readings)
                line = f"{keb} [{reading_str}]{p_flag}{eng}"
            else:
                line = f"{keb}{p_flag}{eng}"
            lines.append(line)
    else:
        # Kana-only entries
        kana_only = [
            (reb, r_common)
            for reb, r_common, no_kanji, restrictions in reading_elements
        ]
        if kana_only:
            eng = build_eng(senses)
            if eng:
                common = any(c for _, c in kana_only)
                p_flag = "/(P)" if common else ""
                word = "; ".join(reb for reb, _ in kana_only)
                line = f"{word}{p_flag}{eng}"
                lines.append(line)

    return lines


def convert(input_path: str, output_path: str):
    print(f"Reading {input_path} …")

    # We use iterparse to avoid loading the whole 60MB XML into RAM at once.
    # The JMdict DTD uses lots of entities; ET's iterparse handles them fine
    # as long as the DOCTYPE is present.
    with open_jmdict(input_path) as f:
        context = ET.iterparse(f, events=("end",))
        entry_count = 0
        line_count = 0

        with open(output_path, "w", encoding="utf-8") as out:
            # EDICT2 header line
            out.write("\uFF1F\u3000                                                               "
                      "                                          /EDICT2 converted from JMdict/\n")

            for event, elem in context:
                if elem.tag != "entry":
                    continue
                try:
                    lines = parse_entry(elem)
                    for line in lines:
                        out.write(line + "\n")
                        line_count += 1
                    entry_count += 1
                except Exception as e:
                    seq = elem.findtext("ent_seq", "?")
                    print(f"  Warning: skipped entry {seq}: {e}", file=sys.stderr)
                finally:
                    elem.clear()  # Free memory

                if entry_count % 10000 == 0:
                    print(f"  … {entry_count} entries processed", end="\r")

    print(f"\nDone! {entry_count} entries → {line_count} lines written to {output_path}")
    print(f"Place '{os.path.basename(output_path)}' in Translation Aggregator's 'dictionaries' folder.")
    print("TA will compile it to a .bin on first run automatically.")


def find_jmdict() -> Optional[str]:
    """Look for JMdict in common locations."""
    candidates = [
        "JMdict_e.gz", "JMdict_e.xml", "JMdict_e",
        "JMdict.gz", "JMdict.xml", "JMdict",
        os.path.join("dictionaries", "JMdict_e.gz"),
        os.path.join("dictionaries", "JMdict_e.xml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    if len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    input_path = sys.argv[1] if len(sys.argv) >= 2 else None
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None

    if not input_path:
        input_path = find_jmdict()
        if not input_path:
            print("Error: Could not find JMdict_e.gz or JMdict_e.xml in current directory.")
            print("Download from: https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project")
            print("Usage: python3 jmdict_to_edict2.py [JMdict_e.gz] [output.edict2]")
            sys.exit(1)
        print(f"Auto-detected input: {input_path}")

    if not output_path:
        base = re.sub(r"\.(gz|xml)$", "", os.path.basename(input_path), flags=re.IGNORECASE)
        output_path = base + ".edict2"

    convert(input_path, output_path)


if __name__ == "__main__":
    main()
