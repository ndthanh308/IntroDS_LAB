# Code for the submission  `Finite-Sum Optimization: Adaptivity to Smoothness and Loopless Variance Reduction`

Here is the code for the submission `Finite-Sum Optimization: Adaptivity to Smoothness and Loopless Variance Reduction`. You first need to install the requirements:

```
pip install -r requirements.txt
```
Note that this is only the requirements and no versions are specified. If you
want to use the specific versions used in our experiments, you can check the
```versions.txt``` file. Next, you need to download the data:

```
wget  https://zenodo.org/record/5765804/files/2k_cell_per_study_10studies.tar.bz2?download=1
```
Extract the data at the right place

```
tar -xf 2k_cell_per_study_10studies.tar.bz2\?download=1
mv 2k_cell_per_study_10studies.h5ad data/
```


Three graphics are output, you can launch each of them using  ```python3 launch.py --plot {plot_name}```, where {plot_name} can be one of:

- ```ada_and_vr``` (plot comparing AdaVR against AdaGrad, SAGA and LSVRG)
- ```accel``` (plot comparing AdaVR against accelerated algorithms)
- ```adam_and_rms``` (plot comparing Adam and RMSprop with and without SAGA or LSVRG)

Launching the script will output a window with the graphs and create a file {plot_name}.pdf containing the plot.
