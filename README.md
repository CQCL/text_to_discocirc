# DisCoCirc

Automatic generation of DisCoCirc circuits using CCG.

To install, run `pip install -e .`,
which installs the package in "editable mode".
(No need to reinstall after each edit.)

You will also need a spacy model:
```bash
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
python -m coreferee install en
```

## Example

```python
from discocirc.pipeline.text_to_circuit import text_to_circuit

text_to_circuit("Frank hangs Claudio. Harmonica shoots Snakey. Harmonica shoots Frank.").draw()
```
