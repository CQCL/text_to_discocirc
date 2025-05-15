# DisCoCirc

Automatic generation of DisCoCirc circuits using CCG.

## Setup Steps

1. Run `!git clone https://github.com/CQCL/text_to_discocirc` to clone the repo
2. Then run `pip install -e ./text_to_discocirc` to install the package in "editable mode" (No need to reinstall after each edit.).
3. You will also need a spacy model, so run the following:
```bash
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
pip install fastcoref
```
4. Add the following to the top of your notebook or Python file as well, to install the coreference disambiguation model's pipe for SpaCy
```python
from fastcoref import spacy_component
from discocirc.pipeline.text_to_circuit import text_to_circuit
```

## Example

```python
from discocirc.pipeline.text_to_circuit import text_to_circuit

text_to_circuit("Frank hangs Claudio. Harmonica shoots Snakey. Harmonica shoots Frank.").draw()
```
