import os
import glob
import sys
import argparse
import logging
import json

import numpy as np

import torch
import torchvision

from scipy.io.wavfile import read
from omegaconf import OmegaConf

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
  is_torch = isinstance(f0, torch.Tensor)
  f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * \
      np.log(1 + f0 / 700)
  f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
      (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

  f0_mel[f0_mel <= 1] = 1
  f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
  f0_coarse = (
      f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
  assert f0_coarse.max() <= 255 and f0_coarse.min(
  ) >= 1, (f0_coarse.max(), f0_coarse.min())
  return f0_coarse


def transform(mel, height):  # 68-92
    # r = np.random.random()
    # rate = r * 0.3 + 0.85 # 0.85-1.15
    # height = int(mel.size(-2) * rate)
    tgt = torchvision.transforms.functional.resize(mel, (height, mel.size(-1)))
    if height >= mel.size(-2):
        return tgt[:, :mel.size(-2), :]
    else:
        silence = tgt[:, -1:, :].repeat(1, mel.size(-2)-height, 1)
        silence += torch.randn_like(silence) / 10
        return torch.cat((tgt, silence), 1)


def stretch(mel, width):  # 0.5-2
    return torchvision.transforms.functional.resize(mel, (mel.size(-2), width))


def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if iteration is None:
    iteration = 1
  if learning_rate is None:
    learning_rate = 0.0002
  if optimizer is not None and checkpoint_dict['optimizer'] is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict = {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
      checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  # ckptname = checkpoint_path.split(os.sep)[-1]
  # newest_step = int(ckptname.split(".")[0].split("_")[1])
  # val_steps = 2000
  # last_ckptname = checkpoint_path.replace(str(newest_step), str(newest_step - val_steps*3))
  # if newest_step >= val_steps*3:
  #   os.system(f"rm {last_ckptname}")
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
      iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(10, 2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                 interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                 interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.yaml",
                      help='YAML file for configuration')
  args = parser.parse_args()

  hparams = OmegaConf.load(args.config)
  model_dir = os.path.join("./logs", hparams.train.model)
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  config_save_path = os.path.join(model_dir, "config.json")
  os.system(f"cp {args.config} {config_save_path}")
  hparams.model_dir = model_dir
  return hparams


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)

  formatter = logging.Formatter(
      "%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger