# X-ray Spectroscopy & Machine Learning

## About

...

### Project Team
Tom Penfold, Chemistry, Newcastle University  ([lorem.ipsum@newcastle.ac.uk](mailto:lorem.ipsum@newcastle.ac.uk))  
Nik Khadijah Nik Aznan, RSE Team, Newcastle University  ([nik.nik-aznan@ncl.ac.uk](mailto:nik.nik-aznan@ncl.ac.uk))  
Kathryn Garside, RSE Team, Newcastle University ([kathryn.garside@newcastle.ac.uk](mailto:kathryn.garside@newcastle.ac.uk))

## Models

### AEGAN (main branch)
Trains model via a coupled Autoencoder Generalised adverserial network. The couple autoencoder consists of two generative networks, each built from an encoder and decoder block. A shared layer connects the two networks and enables cross-domain paths through the network, allowing reconstruction and prediction for both of the inputs. A discriminator network aims to check for differences between real inputs and that which have either been reconstructed or predicted, forcing the generative network to perform better.

```python cli_aegan.py learn in_aegan.json```  

Example model parameters can be found in `in_aegan.json`. The user can specify hidden size of linear layers (*hidden_size*), dropout (*dropout*), the number of hidden layers in the encoder-decoder (*n_hl_gen*), shared (*n_hl_shared*) and discriminator (*n_nl_dis*) networks, activation function (*activation*), loss function for the generative (*loss_gen*) and discriminator (*loss_dis*) networks, learning rates (*lr_gen* and *lr_dis*).

A general layer in the model is MLP consisting of a linear layer, batch norm layer, activation. 

(TODO: add dropout layers)


The trained model can then be used for inference of both structure and spectrum. Three predict methods are provided for the model depending on the input sources. 

```python cli_aegan.py predict_aegan model_dir xyz_dir xanes_dir``` 

```python cli_aegan.py predict_aegan_spectrum model_dir xyz_dir```  
```python cli_aegan.py predict_aegan_structure model_dir xanes_dir```  
 

Since the model affords reconstruction of the input data as well as prediction the predict method currently performs both. For example, with structure data as the input the predict method returns a reconstruction of the stucture and prediction of the spectrum. 

(TODO: Add generalised CNN option to model.)

### AE (ae-dev branch)

ae_mlp, ae_cnn
...

### Original XANES PyTorch Implementation

...

####


### RSE Contact
C. Adipiscing  
RSE Team  
Newcastle University  
([consectetur.adpiscing@newcastle.ac.uk](mailto:consectetur.adpiscing@newcastle.ac.uk))  

## Built With

This section is intended to list the frameworks and tools you're using to develop this software. Please link to the home page or documentatation in each case.

[Framework 1](https://something.com)  
[Framework 2](https://something.com)  
[Framework 3](https://something.com)  

## Getting Started

### Prerequisites
The code has been designed to support python 3. The project has the following dependencies and version requirements:

- tensorflow>=2.1.0
- numpy
- scipy
- scikit-learn
- ase
- tqdm


### Installation

How to build or install the applcation.

### Running Locally

To run the training :
`python cli.py learn in.json`

To train xanes in ae-dev : 
run `python cli_ae.py train_xanes in.json`.
Make sure the path is correct in `in.json`

To predict xyz in ae-dev :
run `python cli_ae.py predict_xyz ./model_0xx in_predict.json`.
Make sure the path is correct in `in_predict.json`.

### Running Tests

How to run tests on your local system.

## Deployment

### Local

Deploying to a production style setup but on the local system. Examples of this would include `venv`, `anaconda`, `Docker` or `minikube`. 

### Production

Deploying to the production system. Examples of this would include cloud, HPC or virtual machine. 

## Usage

Any links to production environment, video demos and screenshots.

## Roadmap

- [x] Initial Research  
- [ ] Minimum viable product <-- You are Here  
- [ ] Alpha Release  
- [ ] Feature-Complete Release  

## Contributing

### Main Branch
Protected and can only be pushed to via pull requests. Should be considered stable and a representation of production code.

### Dev Branch
Should be considered fragile, code should compile and run but features may be prone to errors.

### Feature Branches
A branch per feature being worked on.

https://nvie.com/posts/a-successful-git-branching-model/

## License

## Citiation

Please cite the associated papers for this work if you use this code:

```
@article{xxx2021paper,
  title={Title},
  author={Author},
  journal={arXiv},
  year={2021}
}
```


## Acknowledgements
This work was funded by a grant from the UK Research Councils, EPSRC grant ref. EP/L012345/1, “Example project title, please update”.
