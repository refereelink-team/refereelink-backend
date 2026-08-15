# Weights Directory

Store downloaded model weights for the local soccer app here.

The asset setup script writes `.pt` files into this directory so model assets
are managed separately from sample videos in `assets/data/`.

## Multi-view foul model

The multi-view review page can use SoccerNet's VARS MViT V2 Small model.
Keep the official checkpoint out of Git and place it at:

```text
assets/weights/14_model.pth.tar
```

The page still runs with clearly labelled scripted demo results when the
runtime assets are absent. Run `tools/setup_mvfoul.sh` to fetch only the
official `VARS model` source directory. You may override both runtime paths:

```text
SC_MVFOUL_CODE_PATH=/path/to/VARS model
SC_MVFOUL_WEIGHTS_PATH=/path/to/14_model.pth.tar
```
