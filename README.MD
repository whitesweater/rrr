# Terrain Generation Testing

Generate a DEM with [pretrained weights](https://drive.google.com/drive/u/0/folders/10-BNODcQzSNM0QUacxADrHzedF172L_6).

[Sketches and corresponding DEMs](https://drive.google.com/drive/u/0/folders/1Tc6M39ozKqlfrkrklPaXspW8GLdu2PZy) are
available for demo use. Note that we exclude the last two DEMs
(namely USGS_1M_18_x26y462_PA_Northcentral_2019_B19_tile_4_3.tif and
USGS_1M_18_x26y462_PA_Northcentral_2019_B19_tile_4_4.tif) for unseen skectch validation and our checkpoint was trained
on the other
98 DEMs.

See [Inference Usage](#inference-usage)

See [Training Usage](#training-usage)

## Installing the dependencies

Before running the scripts, make sure to install the library's training
dependencies:

**Important**

To make sure you can successfully run the latest versions of the example
scripts, we highly recommend **installing from source** and keeping the
install up to date as we update the example scripts frequently and
install some example-specific requirements. To do this, execute the
following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Then cd in the example folder and run

```bash
pip install -r requirements.txt
```

And initialize an
[🤗Accelerate](https://github.com/huggingface/accelerate/) environment
with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions
about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell e.g. a
notebook

```python
from accelerate.utils import write_basic_config

write_basic_config()
```

## Inference Usage

Replace **"your downloaded weight path"** with your downloaded directory + '/checkpoint500/controlnet/'.

```python
controlnet_path = "your downloaded weight path"
```

Replace **"your sketch path"** with your sketch path.

```python
control_image = load_image("your sketch path")
```

Run

```python
python
terrain_generation_test.py
```

## Training Usage

1. Replace **"YourDataset.py"** in the training command (refer to **script** file) with your customized dataset. A example is demonstrated as in our **TerrainDataset.py**.
   
2. Modify the hyperparameters in the training command from the **script** file and start training! 🚀

