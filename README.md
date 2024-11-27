# Autoencoders Package
This package provides a collection of autoencoder implementations.

We use wandb.ai for experiment tracking.

## Dataset
We utilize the CelebA dataset (aligned) for training and testing various autoencoders. You can download the dataset from:

https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## Installation
To install the package, run the following command:
```
pip install -e .
```

Then, train a model using the following command:

```
python scripts/train_model.py --model_name vae --data_folder DATA_FOLDER
```

Replace DATA_FOLDER with the actual path to your dataset.

## Contributing
Contributions are welcome! If you'd like to add a new autoencoder implementation or improve existing ones, please submit a pull request.

## License
This package is released under the MIT License.