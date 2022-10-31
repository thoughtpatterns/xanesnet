# X-ray Spectroscopy & Machine Learning
A template repo for the standard RSE project

## About

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed sollicitudin ante at eleifend eleifend. Sed non vestibulum nisi. Aliquam vel condimentum quam. Donec fringilla et purus at auctor. Praesent euismod vitae metus non consectetur. Sed interdum aliquet nisl at efficitur. Nulla urna quam, gravida eget elementum eget, mattis nec tortor. Fusce ut neque tellus. Integer at magna feugiat lacus porta posuere eget vitae metus.

Curabitur a tempus arcu. Maecenas blandit risus quam, quis convallis justo pretium in. Suspendisse rutrum, elit at venenatis cursus, dolor ligula iaculis dui, ut dignissim enim justo at ligula. Donec interdum dignissim egestas. Nullam nec ultrices enim. Nam quis arcu tincidunt, auctor purus sit amet, aliquam libero. Fusce rhoncus lectus ac imperdiet varius. Sed gravida urna eros, ac luctus justo condimentum nec. Integer ultrices nibh in neque sagittis, at pretium erat pretium. Praesent feugiat purus id iaculis laoreet. Proin in tellus tristique, congue ante in, sodales quam. Sed imperdiet est tortor, eget vestibulum tortor pulvinar volutpat. In et pretium nisl.

### Project Team
Tom Penfold, Chemistry, Newcastle University  ([lorem.ipsum@newcastle.ac.uk](mailto:lorem.ipsum@newcastle.ac.uk))  
Nik Khadijah Nik Aznan, RSE Team, Newcastle University  ([nik.nik-aznan@ncl.ac.uk](mailto:nik.nik-aznan@ncl.ac.uk))  
Kathryn Garside, RSE Team, Newcastle University ([kathryn.garside@newcastle.ac.uk](mailto:kathryn.garside@newcastle.ac.uk))


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
