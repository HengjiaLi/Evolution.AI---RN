Pytorch implementation of the project: Exploring Positional Encodings in CNNs for Visual Question Answering Tasks

## Setup

For training and testing the models discussed in the report, please upload the folder to Gdrive and use Google Colab, where all dependensies are pre-installed.

## Usage

Use Google Colab to execute the file run.ipynb, to train/test the target model using the pixel-version dataset. Alternatively, execute run_state to train/test model using the state-description dataset.

## Main Scripts and Files

run.ipynb -- A notebook for train/test the model using pixel dataset\
run_state.ipynb -- A notebook for train/test the model using state description dataset

more_clevr.py -- Generate the extended pixel version of the Sort-of-CLEVR dataset.\
more_clevr_state.py -- Generate the extended state description version of the Sort-of-CLEVR dataset.

model_sim.py -- Defines the simplified RN model with arbitrary X-Y coordinate PE\
model_sim_noH.py -- Defines the simplified RN model with arbitrary X-Y coordinate PE, with the horizontal coordinate been removed\
model_simnoPE.py -- Defines the simplified RN model with padding but no PE\
model_simnoPadding.py -- Defines the simplified RN model without padding\
model_simReplicate -- Defines the simplified RN model with replicate padding

model_sin.py -- Defines the simplified RN model with sinusoidal PE\
model_sin_relative.py -- Defines the simplified RN model with sinusoidal PE + relative distance

model_sincomplex.py -- Defines the simplified RN model with complex PE

main.py -- Define the training/testing functions of the RN model for the pixel dataset\
main_state.py -- Define the training/testing functions of the RN model for the state description dataset

eval_on_more_clevr.py -- Evaluate the trained RN model using the pixel test set\
eval_on_more_clevr_state.py -- Evaluate the trained RN model using the state description test set
