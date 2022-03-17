# Unsupervised Instance Segmentation

## Goals

To integrate recognition and segmentation in machine learning: to be able to point to and count "things" as part of recognition.

## Architecture

Classic Transformer architecture on an _n_ x _m_ "Thing" matrix of _n_ "things," each with an _m_-dimensional vector encoding.

## Training

Use _intra-input_ prediction instead of labels:
1. Start with a random initialization of the _n_x_m_ "thing" matrix.
2. For each training step,
    1. Give the network coordinates as input.
    2. Verify network output against the actual value at that location (with a loss function of your choice).
    3. Update all encodings to reduce loss.
3. Stop at a certain point (tbd) and move on to another training input.

Note that, internally, coordinates are converted to a positional encoding, and _that_ is input to the network, not the raw coordinates.

## Details

### Positional Encoding

In this model, positional encodings are 2 x _d_ x 2 tensors; for a positional encoding _P_, _P<sub>ijk</sub>_ (or, Python-flavored, _P_[_i_, _j_, _k_]) corresponds to
- **_i_**: Relative to the whole size of the input if _i_=0 and relative to a defined sample rate (if none, one pixel/datapoint) otherwise;
- **_j_**: Corresponding to a frequency 2<sup>_j_</sup> above the base rate;
- **_k_**: Cosine component if _k_=0, sine otherwise (i.e. real if _k_=0, imaginary otherwise).

Positional similarity is calculated by—and here I wish GitHub enabled LaTeX—the following (ugh) steps:
1. Accept arguments _A_ (positional encoding of the first position) and _B_ (of the second), and _s_ (frequency/scale after which you don't care).
2. Let _C_ = pointwise multiplication of _A_ and _B_ (sine with sine, frequency with frequency, relative with relative).
3. Sum over the last dimension of _C_.
4. Let _S_ = sigmoid(arange(_d_) - _s_).
5. _C_ = _S_ + (1-_S_)_C_. (i.e., dramatically reduce the dynamic range of scales after _s_, yet still in a differentiable manner.)
6. Multiply along the second (_d_) dimension.

_Et voilà_, a pair of scalars representing similarity, one for each style of representation (_i_ above).
