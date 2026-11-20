# Bulgarian proofreader files

What `tomo/grammar.py` checks Tomo's Bulgarian replies with.

| File | What it is | Source and licence |
|---|---|---|
| `bg_BG.aff`, `bg_BG.dic` | The Bulgarian Hunspell dictionary: 78,000 stems and their endings, about 890,000 word forms | The bgoffice project, as shipped in [LibreOffice's dictionaries](https://github.com/LibreOffice/dictionaries/tree/master/bg_BG) (version 2017.09.30). GPL, LGPL or MPL, at your choice. |
| `tagger.npz` | The grammar tagger: about 1.3 million weights (float16), trained by `scripts/train_grammar.py` | Trained on Bulgarian Wikipedia (CC BY-SA 4.0) and on [thebogko's Bulgarian spelling and grammar mistakes](https://huggingface.co/datasets/thebogko/bulgarian-grammar-mistakes) (Apache-2.0). |

To train the tagger again, put the dictionary, a Bulgarian Wikipedia parquet file
(`wikimedia/wikipedia`, `20231101.bg`) and the two thebogko CSV files in one folder, then:

```
pip install torch pyarrow
python scripts/train_grammar.py --data FOLDER --out assets/grammar/tagger.npz
```
