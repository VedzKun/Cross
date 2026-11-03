"""
Synthetic regulatory document pair generator for semdiff training data.

Generates labeled sentence pairs by:
  (a) Paraphrasing → cosmetic changes
  (b) Changing numbers/dates/obligations → substantive changes
  (c) Flipping negations/modals/key obligations → critical changes
  (d) Machine-translating pairs into Hindi/Tamil

Does NOT require external APIs. Uses:
  - Local rule-based transformations for (a)(b)(c)
  - Optional deep_translator / googletrans for (d) [only used if installed]

Output:
  data/synthetic/pairs.csv     — training pairs (text_a, text_b, label, source_lang, pair_lang, doc_id)
  data/synthetic/docs/         — synthetic document text files (for doc-level splitting)
"""

from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import unicodedata

# -------------------------------------------------------------------------
# Seed regulatory text snippets (English, Hindi, Tamil)
# -------------------------------------------------------------------------

SEED_SNIPPETS_EN = [
    # Format: (clause_id, text)
    ("1.1", "Every registered entity shall submit a quarterly compliance report to the Reserve Bank of India within 30 days of the end of each quarter."),
    ("1.2", "The minimum capital requirement for a Category-I Authorised Dealer shall not be less than Rs. 500 crore at all times."),
    ("1.3", "No person shall engage in foreign exchange transactions without prior approval of the designated authority under Section 10(1) of the Act."),
    ("1.4", "Any breach of the provisions of this circular shall attract a penalty of up to Rs. 5 lakh per day of continuing default."),
    ("1.5", "The interest rate on floating rate instruments shall be reset every six months with effect from 1 April 2024."),
    ("1.6", "Banks may extend credit facilities to micro and small enterprises up to a limit of Rs. 10 crore under the priority sector norms."),
    ("1.7", "All entities must obtain written consent from the borrower before sharing personal credit information with any third party."),
    ("1.8", "The board of directors shall meet at least four times in every financial year with not more than 120 days intervening between two consecutive meetings."),
    ("1.9", "Mutual funds shall disclose the total expense ratio on a daily basis on the website of the respective asset management company."),
    ("1.10", "The appointment of statutory auditor shall be subject to prior approval of the Reserve Bank where the net owned fund of the non-banking financial company exceeds Rs. 1,000 crore."),
    ("2.1", "Systemically important non-deposit taking non-banking financial companies shall maintain a minimum Tier-I capital of 10 per cent of aggregate risk weighted assets on an ongoing basis."),
    ("2.2", "The credit card issuer shall not levy interest at a rate exceeding 36 per cent per annum on any outstanding dues."),
    ("2.3", "Every payment aggregator shall maintain a nodal account with a scheduled commercial bank for the settlement of transactions."),
    ("2.4", "Insurance companies are required to settle valid claims within 30 days of receiving all necessary documents from the policyholder."),
    ("2.5", "The anti-money laundering programme of every reporting entity shall be reviewed at least once every two years by the board of directors."),
    ("3.1", "Exports made under the advance authorisation scheme shall be exempt from integrated goods and services tax and compensation cess."),
    ("3.2", "No bank shall grant unsecured loans or advances aggregating more than Rs. 25 lakh to any single borrower who is a director or relative of a director."),
    ("3.3", "The recovery agent shall not make threatening calls or visit the residence of the borrower between 7 p.m. and 7 a.m."),
    ("3.4", "Portfolio managers shall maintain separate books of accounts for each client's portfolio and shall not mix client funds."),
    ("3.5", "Every alternate investment fund shall appoint a custodian who is registered with the Securities and Exchange Board of India."),
]

SEED_SNIPPETS_HI = [
    ("4.1", "प्रत्येक पंजीकृत संस्था को प्रत्येक तिमाही की समाप्ति के 30 दिनों के भीतर भारतीय रिज़र्व बैंक को त्रैमासिक अनुपालन रिपोर्ट प्रस्तुत करनी होगी।"),
    ("4.2", "कोई भी व्यक्ति अधिनियम की धारा 10(1) के अंतर्गत नामित प्राधिकरण की पूर्व अनुमति के बिना विदेशी मुद्रा लेनदेन में संलग्न नहीं होगा।"),
    ("4.3", "श्रेणी-I अधिकृत डीलर के लिए न्यूनतम पूंजी आवश्यकता हर समय रु. 500 करोड़ से कम नहीं होनी चाहिए।"),
    ("4.4", "इस परिपत्र के प्रावधानों के किसी भी उल्लंघन पर जारी चूक के प्रति दिन रु. 5 लाख तक का जुर्माना लगाया जाएगा।"),
    ("4.5", "बैंक प्राथमिकता क्षेत्र मानदंडों के तहत सूक्ष्म और लघु उद्यमों को रु. 10 करोड़ की सीमा तक ऋण सुविधाएं दे सकते हैं।"),
]

SEED_SNIPPETS_TA = [
    ("5.1", "ஒவ்வொரு பதிவு செய்யப்பட்ட நிறுவனமும் ஒவ்வொரு காலாண்டின் முடிவிலிருந்து 30 நாட்களுக்குள் இந்திய ரிசர்வ் வங்கிக்கு காலாண்டு இணக்க அறிக்கையை சமர்ப்பிக்க வேண்டும்."),
    ("5.2", "பிரிவு 10(1) இன் கீழ் நியமிக்கப்பட்ட அதிகாரியின் முன் அனுமதி இல்லாமல் எந்த நபரும் அந்நிய செலாவணி பரிவர்த்தனைகளில் ஈடுபட கூடாது."),
    ("5.3", "பிரிவு-I அங்கீகரிக்கப்பட்ட டீலருக்கான குறைந்தபட்ச மூலதன தேவை எப்போதும் ரூ. 500 கோடிக்கு குறைவாக இருக்கக் கூடாது."),
]

# -------------------------------------------------------------------------
# Transformation functions
# -------------------------------------------------------------------------

def _cosmetic_transform(text: str) -> str:
    """Light paraphrase: synonym swaps, minor reordering (no meaning change)."""
    transforms = [
        (r"\bshall\b", "will"),
        (r"\bmust\b", "is required to"),
        (r"\bprior to\b", "before"),
        (r"\bwith respect to\b", "regarding"),
        (r"\bin accordance with\b", "as per"),
        (r"\bprovided that\b", "subject to the condition that"),
        (r"\bnotwithstanding\b", "despite"),
        (r"\binter alia\b", "among other things"),
        (r"\butilize\b", "use"),
        (r"\bcommence\b", "begin"),
        (r"\bterminate\b", "end"),
        (r"\bsubsequent to\b", "after"),
        (r"\bprior\b", "previous"),
        (r"\bfurthermore\b", "additionally"),
        (r"\bhowever\b", "nevertheless"),
    ]
    result = text
    random.shuffle(transforms)
    for pattern, replacement in transforms[:random.randint(1, 3)]:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _substantive_transform(text: str) -> str:
    """
    Change numbers, dates, amounts, or deadlines → substantive change.
    """
    def _bump_number(m: re.Match) -> str:
        num_str = m.group(0).replace(",", "")
        try:
            val = float(num_str)
            # 10-40% change
            delta = val * random.uniform(0.10, 0.40)
            new_val = val + delta if random.random() > 0.5 else val - delta
            new_val = max(1, int(round(new_val)))
            return f"{new_val:,}"
        except ValueError:
            return m.group(0)

    def _bump_percentage(m: re.Match) -> str:
        try:
            val = float(m.group(1))
            delta = val * random.uniform(0.10, 0.50)
            new_val = round(val + delta if random.random() > 0.5 else val - delta, 2)
            return f"{max(0.01, new_val)}{m.group(2)}"
        except ValueError:
            return m.group(0)

    def _bump_days(m: re.Match) -> str:
        days_map = {
            "30 days": random.choice(["45 days", "60 days", "15 days", "21 days"]),
            "60 days": random.choice(["45 days", "90 days", "30 days"]),
            "90 days": random.choice(["60 days", "120 days"]),
            "six months": random.choice(["three months", "twelve months", "nine months"]),
        }
        for k, v in days_map.items():
            if k.lower() in m.group(0).lower():
                return m.group(0).replace(k, v)
        return m.group(0)

    result = text
    # Bump standalone integers (e.g., "500", "10", "30")
    result = re.sub(r"\b(\d{1,3}(?:,\d{3})*)\b(?!\s*(?:per|crore|lakh|%|days|months|years))", _bump_number, result)
    # Bump percentages
    result = re.sub(r"(\d+(?:\.\d+)?)\s*(per\s*cent|%)", _bump_percentage, result)
    # Bump day/month deadlines
    result = re.sub(r"\b(?:30|60|90)\s+days?\b|\bsix months?\b", _bump_days, result)
    return result


def _critical_transform(text: str) -> str:
    """
    Critical change: flip negation, change strong modal, alter obligation.
    """
    transforms = [
        # Flip "shall not" → "shall" and vice versa
        (r"\bshall not\b", "SHALL_NOT_TEMP"),
        (r"\bshall\b", "must"),
        ("SHALL_NOT_TEMP", "shall"),
        # Add/remove "not"
        (r"\bmay\b", "shall not"),
        (r"\bmust obtain\b", "need not obtain"),
        (r"\bwithout\b", "with"),
        # Obligation flip
        (r"\bis required to\b", "is not required to"),
        (r"\bmandatory\b", "optional"),
        (r"\bcompulsory\b", "voluntary"),
        (r"\bno person shall\b", "every person shall"),
        (r"\bno bank shall\b", "every bank shall"),
    ]

    result = text
    applied = 0
    for pattern, replacement in transforms:
        new_result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if new_result != result:
            result = new_result
            applied += 1
            if applied >= 2:
                break

    # If nothing changed, add/remove a negation
    if applied == 0:
        if " not " in result:
            result = result.replace(" not ", " ", 1)
        else:
            # Insert "not" after first verb
            result = re.sub(r"\b(shall|must|will|may)\b", r"\1 not", result, count=1, flags=re.IGNORECASE)

    return result


def _translate_hint(text: str, lang: str) -> str:
    """
    Return a simple placeholder translation (or attempt real translation if
    deep_translator is installed). Falls back to the original text + a language tag.
    """
    try:
        from deep_translator import GoogleTranslator
        target = "hi" if lang == "hi" else "ta"
        translated = GoogleTranslator(source="en", target=target).translate(text[:2000])
        return translated or text
    except Exception:
        # Fallback: mark as translation placeholder
        prefix = "[HI] " if lang == "hi" else "[TA] "
        return prefix + text


# -------------------------------------------------------------------------
# Pair generation
# -------------------------------------------------------------------------

def generate_pairs(
    num_cosmetic: int = 150,
    num_substantive: int = 100,
    num_critical: int = 100,
    include_translations: bool = True,
    seed: int = 42,
) -> List[Dict]:
    random.seed(seed)
    pairs = []
    doc_counter = 0

    all_en_snippets = SEED_SNIPPETS_EN[:]

    # ----- Cosmetic -----
    used = list(all_en_snippets) * (num_cosmetic // len(all_en_snippets) + 2)
    random.shuffle(used)
    for i in range(num_cosmetic):
        clause_id, text_a = used[i % len(used)]
        text_b = _cosmetic_transform(text_a)
        if text_a == text_b:
            text_b = text_a + " (amended)"
        pairs.append({
            "pair_id": f"syn_cos_{i:04d}",
            "doc_id": f"doc_cos_{doc_counter // 10:03d}",
            "clause_id": clause_id,
            "text_a": text_a,
            "text_b": text_b,
            "label": "cosmetic",
            "lang_a": "en",
            "lang_b": "en",
            "source": "synthetic_cosmetic",
        })
        doc_counter += 1

    # ----- Substantive -----
    random.shuffle(used)
    for i in range(num_substantive):
        clause_id, text_a = used[i % len(used)]
        text_b = _substantive_transform(text_a)
        pairs.append({
            "pair_id": f"syn_sub_{i:04d}",
            "doc_id": f"doc_sub_{doc_counter // 10:03d}",
            "clause_id": clause_id,
            "text_a": text_a,
            "text_b": text_b,
            "label": "substantive",
            "lang_a": "en",
            "lang_b": "en",
            "source": "synthetic_substantive",
        })
        doc_counter += 1

    # ----- Critical -----
    random.shuffle(used)
    for i in range(num_critical):
        clause_id, text_a = used[i % len(used)]
        text_b = _critical_transform(text_a)
        pairs.append({
            "pair_id": f"syn_crit_{i:04d}",
            "doc_id": f"doc_crit_{doc_counter // 10:03d}",
            "clause_id": clause_id,
            "text_a": text_a,
            "text_b": text_b,
            "label": "critical",
            "lang_a": "en",
            "lang_b": "en",
            "source": "synthetic_critical",
        })
        doc_counter += 1

    # ----- Cross-lingual pairs (EN ↔ HI) -----
    if include_translations:
        for (clause_id_en, text_en), (clause_id_hi, text_hi) in zip(
            SEED_SNIPPETS_EN[:5], SEED_SNIPPETS_HI[:5]
        ):
            pairs.append({
                "pair_id": f"syn_xling_en_hi_{clause_id_en}",
                "doc_id": f"doc_xling_en_hi",
                "clause_id": clause_id_en,
                "text_a": text_en,
                "text_b": text_hi,
                "label": "cosmetic",  # same meaning, different language
                "lang_a": "en",
                "lang_b": "hi",
                "source": "synthetic_crosslingual",
            })

        for (clause_id_en, text_en), (clause_id_ta, text_ta) in zip(
            SEED_SNIPPETS_EN[:3], SEED_SNIPPETS_TA[:3]
        ):
            pairs.append({
                "pair_id": f"syn_xling_en_ta_{clause_id_en}",
                "doc_id": f"doc_xling_en_ta",
                "clause_id": clause_id_en,
                "text_a": text_en,
                "text_b": text_ta,
                "label": "cosmetic",
                "lang_a": "en",
                "lang_b": "ta",
                "source": "synthetic_crosslingual",
            })

        # Cross-lingual critical: translate critical pairs
        for i in range(min(5, num_critical)):
            clause_id, text_a = all_en_snippets[i % len(all_en_snippets)]
            text_b_en = _critical_transform(text_a)
            text_b_hi = _translate_hint(text_b_en, "hi")
            pairs.append({
                "pair_id": f"syn_xling_crit_{i:04d}",
                "doc_id": "doc_xling_crit",
                "clause_id": clause_id,
                "text_a": text_a,
                "text_b": text_b_hi,
                "label": "critical",
                "lang_a": "en",
                "lang_b": "hi",
                "source": "synthetic_crosslingual_critical",
            })

    return pairs


def write_pairs_csv(pairs: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pair_id", "doc_id", "clause_id", "text_a", "text_b",
                  "label", "lang_a", "lang_b", "source"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairs:
            writer.writerow(row)
    print(f"Wrote {len(pairs)} pairs to {output_path}")


def write_labeling_template(pairs: List[Dict], output_path: Path) -> None:
    """Write a CSV template for hand-labeling with blank label column."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pair_id", "doc_id", "clause_id", "text_a", "text_b",
                  "label_human", "label_synthetic", "lang_a", "lang_b",
                  "notes", "annotator"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in pairs:
            writer.writerow({
                "pair_id": row["pair_id"],
                "doc_id": row["doc_id"],
                "clause_id": row["clause_id"],
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                "label_human": "",     # to be filled by annotator
                "label_synthetic": row["label"],
                "lang_a": row["lang_a"],
                "lang_b": row["lang_b"],
                "notes": "",
                "annotator": "",
            })
    print(f"Wrote labeling template ({len(pairs)} rows) to {output_path}")


def write_synthetic_docs(pairs: List[Dict], docs_dir: Path) -> None:
    """Write one synthetic document text file per doc_id (for doc-level splitting)."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    doc_texts: Dict[str, List[str]] = {}
    for p in pairs:
        doc_id = p["doc_id"]
        if doc_id not in doc_texts:
            doc_texts[doc_id] = []
        doc_texts[doc_id].append(f"Clause {p['clause_id']}: {p['text_a']}")

    for doc_id, sentences in doc_texts.items():
        doc_path = docs_dir / f"{doc_id}.txt"
        doc_path.write_text("\n\n".join(sentences), encoding="utf-8")

    print(f"Wrote {len(doc_texts)} synthetic document files to {docs_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic regulatory document pairs.")
    parser.add_argument("--cosmetic", type=int, default=150)
    parser.add_argument("--substantive", type=int, default=100)
    parser.add_argument("--critical", type=int, default=100)
    parser.add_argument("--no-translations", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    pairs = generate_pairs(
        num_cosmetic=args.cosmetic,
        num_substantive=args.substantive,
        num_critical=args.critical,
        include_translations=not args.no_translations,
        seed=args.seed,
    )
    write_pairs_csv(pairs, args.output_dir / "pairs.csv")
    write_labeling_template(pairs, args.output_dir / "labeling_template.csv")
    write_synthetic_docs(pairs, args.output_dir / "docs")

    print("\nSummary:")
    from collections import Counter
    counts = Counter(p["label"] for p in pairs)
    for label, cnt in sorted(counts.items()):
        print(f"  {label}: {cnt}")

