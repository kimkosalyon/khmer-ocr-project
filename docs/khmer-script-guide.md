# Khmer Script Guide

The Khmer script is an abugida: consonant letters carry an inherent vowel, and dependent vowel signs modify that vowel. For OCR and NLP work, it is useful to separate the modern writing-system inventory from the broader Unicode inventory, because Unicode also encodes obsolete, deprecated, and transliteration-only characters.

Primary references:

- Unicode Consortium, [Khmer names list](https://www.unicode.org/charts/nameslist/n_1780.html)
- Sok, M. (2016), [Phonological Principles for Automatic Phonetic Transcription of Khmer Words](https://aclanthology.org/Y16-3013.pdf)

## Verified Counts

| Category | Practical modern count | Unicode / orthographic note |
|---|---:|---|
| Consonants | 33 | Unicode encodes 35 consonant letters; ឝ and ឞ are only for Pali/Sanskrit transliteration. |
| Independent vowels | 13 or 14 | Unicode encodes 14 ordinary independent-vowel code points from U+17A5 to U+17B3 if ឱ and ឲ are counted separately. Unicode says ឲ is a variant of ឱ. |
| Dependent vowel signs | 16 basic signs | Common teaching and NLP inventories may count 23 or 24 when compound vowel forms such as អាំ, អិះ, អុះ, អេះ, and អោះ are counted as vowels. |
| Diacritics and signs | Varies | Unicode separates vowel signs, shifters, pronunciation marks, punctuation, currency, digits, and divination numerals. |
| Khmer digits | 10 | U+17E0 to U+17E9. |

Counts vary because sources use different units: code points, modern letters, teaching vowels, or orthographic vowel combinations. For OCR vocabularies, the safest inventory is the exact Unicode characters observed in the training labels, with explicit normalization and filtering rules.

## 1. Consonants

Modern Khmer uses 33 consonant letters. The two additional encoded consonants, ឝ and ឞ, are obsolete for ordinary Khmer and are used only for Pali/Sanskrit transliteration.

| # | Letter | Unicode | Unicode name | Series |
|---:|---|---|---|---|
| 1 | ក | U+1780 | Khmer Letter Ka | 1st |
| 2 | ខ | U+1781 | Khmer Letter Kha | 1st |
| 3 | គ | U+1782 | Khmer Letter Ko | 2nd |
| 4 | ឃ | U+1783 | Khmer Letter Kho | 2nd |
| 5 | ង | U+1784 | Khmer Letter Ngo | 2nd |
| 6 | ច | U+1785 | Khmer Letter Ca | 1st |
| 7 | ឆ | U+1786 | Khmer Letter Cha | 1st |
| 8 | ជ | U+1787 | Khmer Letter Co | 2nd |
| 9 | ឈ | U+1788 | Khmer Letter Cho | 2nd |
| 10 | ញ | U+1789 | Khmer Letter Nyo | 2nd |
| 11 | ដ | U+178A | Khmer Letter Da | 1st |
| 12 | ឋ | U+178B | Khmer Letter Ttha | 1st |
| 13 | ឌ | U+178C | Khmer Letter Do | 2nd |
| 14 | ឍ | U+178D | Khmer Letter Ttho | 2nd |
| 15 | ណ | U+178E | Khmer Letter Nno | 1st |
| 16 | ត | U+178F | Khmer Letter Ta | 1st |
| 17 | ថ | U+1790 | Khmer Letter Tha | 1st |
| 18 | ទ | U+1791 | Khmer Letter To | 2nd |
| 19 | ធ | U+1792 | Khmer Letter Tho | 2nd |
| 20 | ន | U+1793 | Khmer Letter No | 2nd |
| 21 | ប | U+1794 | Khmer Letter Ba | 1st |
| 22 | ផ | U+1795 | Khmer Letter Pha | 1st |
| 23 | ព | U+1796 | Khmer Letter Po | 2nd |
| 24 | ភ | U+1797 | Khmer Letter Pho | 2nd |
| 25 | ម | U+1798 | Khmer Letter Mo | 2nd |
| 26 | យ | U+1799 | Khmer Letter Yo | 2nd |
| 27 | រ | U+179A | Khmer Letter Ro | 2nd |
| 28 | ល | U+179B | Khmer Letter Lo | 2nd |
| 29 | វ | U+179C | Khmer Letter Vo | 2nd |
| 30 | ស | U+179F | Khmer Letter Sa | 1st |
| 31 | ហ | U+17A0 | Khmer Letter Ha | 1st |
| 32 | ឡ | U+17A1 | Khmer Letter La | 1st |
| 33 | អ | U+17A2 | Khmer Letter Qa | 1st |

Transliteration-only consonants:

| Letter | Unicode | Unicode name | Note |
|---|---|---|---|
| ឝ | U+179D | Khmer Letter Sha | Pali/Sanskrit transliteration only. |
| ឞ | U+179E | Khmer Letter Sso | Pali/Sanskrit transliteration only. |

## 2. Independent Vowels

Unicode encodes two discouraged cloned independent vowels, followed by the ordinary independent-vowel inventory.

Discouraged cloned independent vowels:

| Letter | Unicode | Unicode name | Unicode note |
|---|---|---|---|
| ឣ | U+17A3 | Khmer Independent Vowel Qaq | Use អ instead. |
| ឤ | U+17A4 | Khmer Independent Vowel Qaa | Use អា instead. |

Ordinary independent vowels:

| Letter | Unicode | Unicode name | Note |
|---|---|---|---|
| ឥ | U+17A5 | Khmer Independent Vowel Qi |  |
| ឦ | U+17A6 | Khmer Independent Vowel Qii |  |
| ឧ | U+17A7 | Khmer Independent Vowel Qu |  |
| ឨ | U+17A8 | Khmer Independent Vowel Quk | Obsolete ligature for ឧក; the sequence is preferred. |
| ឩ | U+17A9 | Khmer Independent Vowel Quu |  |
| ឪ | U+17AA | Khmer Independent Vowel Quuv |  |
| ឫ | U+17AB | Khmer Independent Vowel Ry |  |
| ឬ | U+17AC | Khmer Independent Vowel Ryy |  |
| ឭ | U+17AD | Khmer Independent Vowel Ly |  |
| ឮ | U+17AE | Khmer Independent Vowel Lyy |  |
| ឯ | U+17AF | Khmer Independent Vowel Qe |  |
| ឰ | U+17B0 | Khmer Independent Vowel Qai |  |
| ឱ | U+17B1 | Khmer Independent Vowel Qoo Type One | Normal variant. |
| ឲ | U+17B2 | Khmer Independent Vowel Qoo Type Two | Variant of ឱ used in only two words. |
| ឳ | U+17B3 | Khmer Independent Vowel Qau |  |

This table contains 15 code points because it includes the obsolete ឨ and the variant ឲ. Depending on the convention, educational summaries commonly report 13 or 14 independent vowels.

## 3. Dependent Vowel Signs

Unicode's basic dependent vowel signs are U+17B6 to U+17C5:

| Letter | Unicode | Unicode name |
|---|---|---|
| ា | U+17B6 | Khmer Vowel Sign Aa |
| ិ | U+17B7 | Khmer Vowel Sign I |
| ី | U+17B8 | Khmer Vowel Sign Ii |
| ឹ | U+17B9 | Khmer Vowel Sign Y |
| ឺ | U+17BA | Khmer Vowel Sign Yy |
| ុ | U+17BB | Khmer Vowel Sign U |
| ូ | U+17BC | Khmer Vowel Sign Uu |
| ួ | U+17BD | Khmer Vowel Sign Ua |
| ើ | U+17BE | Khmer Vowel Sign Oe |
| ឿ | U+17BF | Khmer Vowel Sign Ya |
| ៀ | U+17C0 | Khmer Vowel Sign Ie |
| េ | U+17C1 | Khmer Vowel Sign E |
| ែ | U+17C2 | Khmer Vowel Sign Ae |
| ៃ | U+17C3 | Khmer Vowel Sign Ai |
| ោ | U+17C4 | Khmer Vowel Sign Oo |
| ៅ | U+17C5 | Khmer Vowel Sign Au |

Several following signs are often treated as vowel signs or vowel-forming marks in practical orthography:

| Letter | Unicode | Unicode name | Note |
|---|---|---|---|
| ំ | U+17C6 | Khmer Sign Nikahit | Usually regarded as vowel sign am, along with om and aam. |
| ះ | U+17C7 | Khmer Sign Reahmuk | Also called srak ah; visarga-like sign. |
| ៈ | U+17C8 | Khmer Sign Yuukaleapintu | Inserts a short inherent vowel with abrupt glottal stop. |

Common compound vowel forms include អុំ, អំ, អាំ, អះ, អិះ, អុះ, អេះ, and អោះ. Some analyses count these as dependent vowels, which is why sources may report 23 or 24 dependent vowels instead of the 16 basic Unicode vowel-sign code points.

## 4. Diacritics, Punctuation, And Symbols

| Character | Unicode | Unicode name | Function |
|---|---|---|---|
| ៉ | U+17C9 | Khmer Sign Muusikatoan | Shifts second register to first. |
| ៊ | U+17CA | Khmer Sign Triisap | Shifts first register to second. |
| ់ | U+17CB | Khmer Sign Bantoc | Shortens the vowel sound in the previous orthographic syllable. |
| ៌ | U+17CC | Khmer Sign Robat | Historical repha-like mark. |
| ៍ | U+17CD | Khmer Sign Toandakhiat | Indicates that the base character is not pronounced. |
| ៎ | U+17CE | Khmer Sign Kakabat | Used with some exclamations. |
| ៏ | U+17CF | Khmer Sign Ahsda | Stressed intonation in some single-consonant words. |
| ័ | U+17D0 | Khmer Sign Samyok Sannya | Pronunciation exception mark, mostly for loan words. |
| ៑ | U+17D1 | Khmer Sign Viriam | Mostly obsolete killer mark. |
| ្ | U+17D2 | Khmer Sign Coeng | Makes the following Khmer letter render as a subscript. |
| ។ | U+17D4 | Khmer Sign Khan | Full stop. |
| ៕ | U+17D5 | Khmer Sign Bariyoosan | End of section or text. |
| ៖ | U+17D6 | Khmer Sign Camnuc Pii Kuuh | Colon. |
| ៗ | U+17D7 | Khmer Sign Lek Too | Repetition sign. |
| ៘ | U+17D8 | Khmer Sign Beyyal | Et cetera; Unicode discourages this character in favor of a spelled-out abbreviation. |
| ៙ | U+17D9 | Khmer Sign Phnaek Muan | Beginning of a book or treatise. |
| ៚ | U+17DA | Khmer Sign Koomuut | End of a book or treatise. |
| ៛ | U+17DB | Khmer Currency Symbol Riel | Currency symbol. |
| ៜ | U+17DC | Khmer Sign Avakrahasanya | Rare omitted Sanskrit vowel sign. |
| ៝ | U+17DD | Khmer Sign Atthacan | Mostly obsolete final-consonant/inherent-vowel mark. |

Khmer digits are ០ ១ ២ ៣ ៤ ៥ ៦ ៧ ៨ ៩ at U+17E0 to U+17E9.

## OCR Notes

- Build the model vocabulary from normalized training labels, not from a fixed educational alphabet list.
- Decide whether to keep transliteration-only letters ឝ and ឞ. They are encoded but should be rare or absent in ordinary Khmer OCR data.
- Decide whether to normalize discouraged characters: ឣ to អ, ឤ to អា, and often ឨ to ឧក.
- Preserve logical Unicode order in labels. Some dependent vowel signs render before or around the consonant but follow the consonant in encoded order.
- Treat coeng, U+17D2, as a real label character. It controls subscript rendering and is required for consonant clusters.
- Keep punctuation, digits, and currency symbols only if they appear in the target OCR domain.

## Verification Notes On The Drafts

The first draft is broadly right that Khmer is an abugida and that modern Khmer has 33 consonants, but it mixes modern alphabet counts with Unicode code-point inventory. It lists ឝ and ឞ inside the consonant table even though that makes 35 consonant rows, not 33 modern consonants. Its dependent-vowel list is incomplete, and the unsupported "largest alphabet in the world" claim should not be used in a technical guide.

The second draft is closer for a technical project because it flags count variation and separates obsolete characters. It still needs clearer wording: Unicode has 16 basic dependent vowel-sign code points, while 23 or 24 is an orthographic/teaching count that includes compound forms. It also says "14 independent vowels" while its own table includes 15 code points if ឱ and ឲ are separate and ឨ is included.
