## Setup

Create conda environment from `environment.yml` file
```
$ conda env create -f environment.yml
```
Activate environment
```
$ conda activate RN3
```
If you don't use conda install python 3 normally and use `pip install` to install remaining dependencies. The list of dependencies can be found in the `environment.yml` file.

## Usage

	$ ./run.sh

or

  	$ python sort_of_clevr_generator.py

to generate sort-of-clevr dataset
and

 	 $ python main.py 

to train the binary RN model. 
Alternatively, use 

 	 $ python main.py --relation-type=ternary

to train the ternary RN model.
