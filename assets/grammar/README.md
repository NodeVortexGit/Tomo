# Bulgarian proofreader files

What `tomo/grammar.py` checks Tomo's Bulgarian replies with.

| File | What it is | Source and licence |
|---|---|---|
| `bg_BG.aff`, `bg_BG.dic` | The Bulgarian Hunspell dictionary: 78,000 stems and their endings, about 890,000 word forms | The bgoffice project, as shipped in [LibreOffice's dictionaries](https://github.com/LibreOffice/dictionaries/tree/master/bg_BG) (version 2017.09.30). GPL, LGPL or MPL, at your choice. |
| `tomo.dic` | Tomo's own words, read after `bg_BG.dic`: common words it lacks ("също", "три", "ми"), newer ones ("интернет", "гигабайт", "таймер", with their endings), forms of address ("мамо", "шефе"), and the words Bulgarian Wikipedia uses often that `bg_BG.dic` doesn't know | Ours (MIT, as Tomo). |
| `tagger.npz` | The grammar tagger: about 1.4 million weights (float16, 2.7 MB), trained by `scripts/train_grammar.py` and calibrated by `scripts/calibrate_grammar.py` (its confidence cut-offs are stored in it) | Trained on Bulgarian Wikipedia (CC BY-SA 4.0), Bulgarian film subtitles for spoken Bulgarian ([OPUS OpenSubtitles 2018](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles)), and on [thebogko's Bulgarian spelling and grammar mistakes](https://huggingface.co/datasets/thebogko/bulgarian-grammar-mistakes) (Apache-2.0). |

To train the tagger again, put a Bulgarian Wikipedia parquet file (`wikimedia/wikipedia`,
`20231101.bg`), `subs.txt` (the start of OPUS's `OpenSubtitles/v2018/mono/bg.txt.gz`, a
line each) and the two thebogko CSV files in one folder, then:

```
pip install torch pyarrow
python scripts/train_grammar.py --data FOLDER --out assets/grammar/tagger.npz
python scripts/calibrate_grammar.py --data FOLDER --tagger assets/grammar/tagger.npz --write
```

The second step measures, on sentences the tagger never saw, how often a change it suggests is
right at each confidence, separately for the article and for other endings, and stores the
cut-offs: from where it's right nine times in ten the proofreader fixes the word by itself,
from where it's right half the time it asks the model.
