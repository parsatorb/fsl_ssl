# Regularizers for Few-shot image classification

Arash Moayyedi, Parsa Torabian, Sina Farsangi

# Training (including test after training is done)
For replicating Su et. al and for the orthogonal experiments, use the training file ``train_ortho.py`` and testing file ``test_ortho.py``

For jigsaw, run ```python train_ortho.py --dataset CUB --train_aug --jigsaw```
For rotation, run ```python train_ortho.py --dataset CUB --train_aug --rotation```

You can change methods by setting the ``--method`` flag to one of ("MAML", "protonet").

For running PIRL experiments, use ``train.py`` (only compatible with "protonet" method and ``--jigsaw`` flags).

Hyperparameters can be seen by looking at ``io_utils.py``. All hyperparameters used are those set to default.

We always use ``--train_aug`` when running experiments. ``--rotation`` and ``--jigsaw`` flags set accordingly.
For orthogonality experiments, ``ortho_factor`` is always set to 0.001. The following flags for ``ortho_loss`` correspond to what was used in the report:
- > "weights_normal" - "Ortho"
- > "srip" - "SRIP"
- > "snapped" - "Snapped"


The code is based on [the repo](https://github.com/cvl-umass/fsl_ssl) of "When Does Self-supervision Improve Few-shot Learning?", ICLR, 2020.
[Project page](https://people.cs.umass.edu/~jcsu/papers/fsl_ssl/), [arXiv](https://arxiv.org/abs/1910.03560), [slides](http://supermoe.cs.umass.edu/fsl_ssl/long_video_slides.pdf)

## Enviroment
 - Python3
 - PyTorch (tested on > 1.0.0)

## Getting started
### Prepare datasets
* Please download images of [Caltech-UCSD birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), [Stanford cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), [fgvc-aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/), [Stanford dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/), and [Oxford flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html), and put them under `filelists/${dset}/images`.

### Base/Val/Novel splits
* Require three data split json file: 'base.json', 'val.json', 'novel.json' for each dataset.
* Splits are included in this repo.  
