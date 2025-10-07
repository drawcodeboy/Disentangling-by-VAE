# Disentangling by VAE
* <b>Disentangled Representation Learning</b> by VAE Repository
* <b>VAE, Beta-VAE</b> are available.

# Preview
* The image below shows the result of the VAE after 100 epochs.
* By training for more epochs and using a Beta-VAE, you can obtain more effective results.

<table align="center">
  <tr>
    <td align="center">
      <img src="assets/intervention/vae_GT.jpg" width="200"><br>
      <em>Ground Truth</em>
    </td>
    <td align="center">
      <img src="assets/intervention/vae_reconstruction.jpg" width="200"><br>
      <em>Reconstruction</em>
    </td>
    <td align="center">
      <img src="assets/intervention/vae_intervention.jpg" width="200"><br>
      <em>Intervention (Wall)</em>
    </td>
  </tr>
</table>


# How to use
### 1. Setings
* Download dataset here > <a href="https://github.com/google-deepmind/3d-shapes">Google Deepmind: 3D Shapes</a>
```
conda create -n drl_base python=3.12
conda activate drl_base
pip install -r requirements.txt

# Download 3dshapes.h5 and put it in the 'data' directory.
python subtasks/data_save/exec.py # This will save images and labels for the PyTorch dataset.
```

### 2. Train, Test, and Intervention(Manipulation)
```
# Train
python train.py --config=vae.3dshapes

# Test
python test.py --config=vae.3dshapes

# Intervention(Manipulation)
python subtasks/intervention/exec.py --config=vae.3dshapes
```