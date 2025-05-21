# LangPath

LangPathModel is a transformer-based model designed to generate 2D paths in an autoregressive manner, conditioned on natural language input.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

To train the LangPathModel, run the following command:

```bash
python training.py \
  --niter 50 \
  --start_lr 0.001 \
  --lr_step 15 \
  --lr_gamma 0.1 \
  --weight_decay 1e-5 \
  --d_model 128 \
  --num_heads 8 \
  --num_decoder_layers 5 \
  --dropout 0.2 \
  --batch_size 500 \
  --dataset_path /path/to/data/cats_and_dogs.pt \
  --output_path /path/to/output_model.pth
```

Replace `/path/to/...` with the appropriate file paths on your system.

## Evaluation

To generate and visualize a path from a text prompt using a trained model:

```bash
python evaluate_model.py \
  --model_path path/to/model.pth \
  --text "bottom circle" \
  --d_model 128 \
  --num_heads 8 \
  --num_decoder_layers 2 \
  --frames 200 \
  --interval 100 \
  --save path/to/video.mp4
```

Omit `--save` to display the animation interactively instead of saving it.

## Dataset Creation

Generate synthetic datasets with simple geometric trajectories:

```bash
python data/squares.py --square_center 250 500 --text "left square" --filename "left_square.pt" --num_origins 5000

python data/circles.py --circle_center 250 500 --text "left circle" --filename "left_circle.pt" --num_origins 5000
```
These scripts create `.pt` files containing training data of paths shaped as squares or circles, labeled with the given text prompts.

## Contributing

Contributions are not currently being accepted as the project is in early development.

## License

This project does not yet have a license. Please contact the author for usage permissions.

