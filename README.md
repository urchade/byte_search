# byte-search

## Installation
```bash
pip install byte-search
```

## Usage
```python
from byte_search.index import SearchIndex

docs = [
    "Contrastive Fine-tuning Improves Robustness for Neural Rankers",
    "Unsupervised Neural Machine Translation for Low-Resource Domains via Meta-Learning",
    "Spatial Dependency Parsing for Semi-Structured Document Information Extraction"
    ]

my_index = SearchIndex(text_list=docs, device='cpu', d_model=64)

my_index.show_topk(query='machine translation', k=1)

```

## Reference

```bibtex
@misc{zaratiana2021contrastive,
      title={Contrastive String Representation Learning using Synthetic Data}, 
      author={Urchade Zaratiana},
      year={2021},
      eprint={2110.04217},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
