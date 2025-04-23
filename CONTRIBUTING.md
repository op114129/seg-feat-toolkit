############################################

Runtime dependencies for the toolkit

Pin to major versions for reproducibility

############################################

=== Core deep-learning stack ===

(Choose the CUDA-matching wheels when installing Torch)

torch>=2.2,<3.0
torchvision>=0.17,<1.0

=== OpenMMLab segmentation ===

mmcv-full>=2.0.0          # Pre-built wheels per CUDA version
mmsegmentation>=1.0.0

=== Image I/O / processing ===

opencv-python-headless>=4.10
scikit-image>=0.23
pywavelets>=1.5            # Wavelet support – indirect dep. of scikit-image

=== Data & plotting ===

matplotlib>=3.8
pandas>=2.2
numpy>=1.26

=== Optional (dev/test) ===

pre-commit>=3.7       # code style hooks

pytest>=8.0           # if you add unit tests
