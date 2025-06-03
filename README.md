# LangPath

LangPathModel is a transformer-based model designed to generate 2D paths in an autoregressive manner, conditioned on natural language input.

As a tutorial please check our tutorial.ipynb.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training

Position yourself into the LangPathModel directory.

To train the LangPathModel, run the following command:

```bash
python LangPathModel/src/training.py \
  --niter 100 \
  --start_lr 0.001 \
  --lr_step 10 \
  --weight_decay 1e-5 \
  --d_model 64 \
  --num_heads 8 \
  --num_decoder_layers 2 \
  --dropout 0.2 \
  --gamma 0.1 \ 
  --batch_size 512 \
  --dataset_path /path/to/your_dataset.pt \
  --output_path /path/to/new_model.pth
```
gamma and start_lr are torch.optim.lr_scheduler.StepLR parameters.

Replace `/path/to/...` with the appropriate file paths on your system.

## Evaluation
**Doesnt work**

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

![alt text](https://github.com/AntonioSkara132/LangPathModel/blob/main/data/visualization.png)

**NEW**
	
Because of increasing size of our data we decided to put comined data on hugging face.

Link to hugging face is [here](https://huggingface.co/datasets/Tonio123/CaptyShapes)

You can also generate dataset by shapes.py in data directoy by modifying classes list.

In classes add:
```python
dict(shape=wanted_shape(circle/square), params=dict(center=(x, y),  text= shape_caption, n=number_of_paths_to_be_generated)),
```

**OLD**

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

