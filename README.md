<div align="center">

# Joint-Dataset Learning and Cross-Consistent Regularization for Text-to-Motion Retrieval

</div>


## Description
Official PyTorch implementation of the paper:
<div align="center">

[**Joint-Dataset Learning and Cross-Consistent Regularization for Text-to-Motion Retrieval**](https://arxiv.org/pdf/2407.02104)

</div>

If you find this code hewlpful, feel free to give a star :star: and cite the following:

```bibtex
@article{messina2024joint,
  title={Joint-Dataset Learning and Cross-Consistent Regularization for Text-to-Motion Retrieval},
  author={Messina, Nicola and Sedmidubsky, Jan and Falchi, Fabrizio and Rebok, Tom{\'a}{\v{s}}},
  journal={arXiv preprint arXiv:2407.02104},
  year={2024}
}
```

## Installation :construction_worker:

<details><summary>Create environment</summary>
&emsp;

Create a python virtual environnement:
```bash
python -m venv ~/.venv/Project
source ~/.venv/Project/bin/activate
```

Install [PyTorch](https://pytorch.org/get-started/locally/)
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install remaining packages:
```
pip install -r requirements.txt
```

</details>

<details><summary>Set up the HumanML3D and KITML datasets</summary>

We use the same data process used by [TMR](https://github.com/Mathux/TMR), so please check their repo for acquiring the details on 
- data processing and preparation
- extraction of guoh3d features
- extraction of text embeddings

</details>

## Training :rocket:

Work in progress :memo:

## Evaluation :bar_chart:

Work in progress :memo:


## License :books:
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including PyTorch, PyTorch3D, Hugging Face, Hydra, and uses datasets which each have their own respective licenses that must also be followed.

## Acknowledgements

We really thanks the authors of [TMR](https://github.com/Mathux/TMR), on which this code is largely based.