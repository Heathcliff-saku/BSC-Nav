# <center>Brain-inspired Spatial Cognition for Navigation (BSC-Nav)</center>

This repository is the official implementation of our paper (From reactive to cognitive: brain-inspired spatial intelligence for embodied agents) 

by [Shouwei Ruan](https://heathcliff-saku.github.io/), [Liyuan Wang](https://lywang3081.github.io/), Caixin Kang, Qihui Zhu, Songming Liu, Xingxing Wei and [Hang Su](https://scholar.google.com/citations?user=dxN1_X0AAAAJ&hl=en), Collaboration between Tsinghua and Beihang University.

<table>
  <tr>
    <td align="center">
      <img src="./assets/demo-1.gif" height="180" alt="GIF 1">
    </td>
    <td align="center">
      <img src="./assets/demo-2.gif" height="200" alt="GIF 2">
    </td>
    <td align="center">
      <img src="./assets/demo-3.gif" height="200" alt="GIF 3">
    </td>
    <td align="center">
      <img src="./assets/demo-4.gif" height="180" alt="GIF 4">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center">
      <img src="./assets/real2.gif" height="200" alt="GIF 5">
    </td>
    <td align="center">
      <img src="./assets/real3.gif" height="200" alt="GIF 6">
    </td>
    <td align="center">
      <img src="./assets/real1.gif" height="200" alt="GIF 7">
    </td>
  </tr>
</table>


We proposed BSC-Nav (Brain-inspired spatial cognition for navigation), which leverages bio-inspired spatial cognition mechanisms to continuously understand the surrounding environment by constructing structured spatial memory, supporting general navigation and more advanced spatial intelligence. Our paper is currently under review, and all features will be made publicly available in the near future.

<div align="center">
  <img src="/assets/framework.jpeg" alt="framework" width="500" />
</div>

### 🧐 Environment & Dataset preparation

1. Preparing **Habitat-lab** & **Habitat-sim** Env (refer https://github.com/facebookresearch/habitat-lab for detailed installation instructions)

```
# create conda env
conda create -n bscnav python=3.9 cmake=3.14.0
conda activate bscnav

# install habitat-sim with bullet physics (also see https://github.com/facebookresearch/habitat-sim#installation)
conda install habitat-sim withbullet -c conda-forge -c aihabitat

# install habitat-lab stable version
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab

```

2. Clone this repo and install dependencies

```
git clone --branch sim https://github.com/Heathcliff-saku/BSC-Nav.git
cd BSC-Nav
pip install -r requirements.txt 
# You may need to manually resolve version conflicts
```

3. Download scene dataset and benchmark episode files

- BSC-Nav is evaluated under HM3D and MP3D scene datasets, for download, We strongly recommend referring to the following guidelines: 

    HM3D: http://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d
    
    MP3D: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#matterport3d-mp3d-dataset

-   (**Skip the following step if you only want to perform a demo test**) The episode file is used for benchmarking and can be downloaded from the following address:
    
    *Object-goal navigation*: 

    https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md, we need `objectnav_hm3d_v2.zip` and `objectnav_mp3d_v1.zip`

    *Open-vocabulary object navigation (OVON)*:

    https://github.com/naokiyokoyama/ovon?tab=readme-ov-file#dart-downloading-the-datasets

    *Text-instance navigation*:
    
    https://github.com/XinyuSun/PSL-InstanceNav?tab=readme-ov-file#install-psl

    *Image-instance navigation*:
    https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md, we need `instance_imagenav_hm3d_v3.zip`


- In addition to the episode file, you need to use the corresponding configuration file (.yaml file) at the same time and make sure it is unzip and organized as follows:

```
-- BSC-Nav
   -- data_episode
      -- eqa
            ...
      -- imagenav
            ...
      -- objnav
         -- train
         -- val
         -- val_mini
      -- ovnav
            ...
      -- textnav
            ...
      -- vln
            ...
```

### 🎮 Quick start with BSC-Nav !


### 🎯 Benchmarks

0. Structured spatial memory construction:
```
python create_memory_for_dataset.py
```

1. For *Object-goal navigation* :
For detailed parameter settings, please refer to `args.py`
```
nohup python -u objnav_benchmark.py --no_record --no_vis --load_single_floor --use_only_working_memory "$@" >> run_objnav_hm3d.txt 2>&1
```

2. For *Open-vocabulary object navigation (OVON)*
```
nohup python -u ovnav_benchmark.py --no_record --no_vis --load_single_floor \
    --dataset 'hm3d' --benchmark_dataset 'hm3d' \
    --HM3D_CONFIG_PATH '**path_to**/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/ovon_hm3d.yaml' \
    --HM3D_EPISODE_PREFIX '**path_to**/data_episode/ovnav/hm3d/hm3d/val_seen/val_seen.json.gz' \
    --eval_episodes 1000 "$@" > run_ovnav_hm3d.txt 2>&1
```

3. For *Text-instance navigation*:
```
nohup python -u textnav_benchmark.py --no_record --no_vis --load_single_floor \
        --dataset 'hm3d' --benchmark_dataset 'hm3d' \
        --HM3D_EPISODE_PREFIX '**path_to**/data_episode/imagenav/instance_imagenav_hm3d_v3/val/val.json.gz' \
        --HM3D_CONFIG_PATH '**path_to**/habitat-lab/habitat-lab/habitat/config/benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml' \
        --eval_episodes 1000 "$@" >> run_textnav_hm3d.txt 2>&1
```

4. For *Image-instance navigation*:

```
nohup python -u imagenav_benchmark.py --no_record --no_vis --load_single_floor \
        --dataset 'hm3d' --benchmark_dataset 'hm3d' \
        --HM3D_EPISODE_PREFIX '**path_to**/data_episode/imagenav/instance_imagenav_hm3d_v3/val/val.json.gz' \
        --HM3D_CONFIG_PATH '**path_to**/habitat-lab/habitat-lab/habitat/config/benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml' \
        --eval_episodes 1000 "$@" >> run_imgnav_hm3d.txt 2>&1
```

5. For *Long-horizon Instruction-based Navigation (LIN)*

```
nohup python -u vlnce_benchmark.py --no_vis --load_single_floor \
    --dataset 'mp3d' --benchmark_dataset 'mp3d' \
    --MP3D_CONFIG_PATH '**path_to**/habitat-lab/habitat-lab/habitat/config/benchmark/nav/vln_r2r_ges.yaml' \
    --MP3D_EPISODE_PREFIX '**path_to**/data_episode/vln/vln_r2r_mp3d_v1/val_unseen/val_unseen.json.gz' \
    --dataset_dir '**path_to**/mp3d/mp3d_habitat/mp3d' \
    --scene_dataset_config_file **path_to**/mp3d/mp3d_habitat/mp3d/mp3d.scene_dataset_config.json \
    --eval_episodes 1000 --scene_name '5LpN3gDmAk7' --nav_task 'vlnce' --success_distance 3.0 "$@" > run_vlnnav_mp3d.txt 2>&1
```

6. For A-EQA
```
nohup python agent_eqa.py --no_vis > run_eqa.txt 2>&1 &^C
```

### 📆 TODO
- [x] Upload all experimental scripts for both simulator and real environments
- [x] Release general navigation reproduction instructions
- [x] Provide vision-language navigation and A-EQA documentation for reproduction
- [ ] Provide physical environment navigation and mobile manipulation reproduction instructions

### 🤗 Contact

If you have any questions or suggestions, look forward to your contact with us:

Shouwei Ruan: shouweiruan@buaa.edu.cn
