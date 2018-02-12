# punkProse

Punctuation generation for speech transcripts using lexical, syntactic and prosodic features. 

Modification on forked repository (by reducing training to one stage and addition of more word-level prosodic features). This version lets use any combination of word-aligned features. 

Prosodically annotated files are in proscript format (https://github.com/alpoktem/proscript). For example data and extraction scripts see: https://github.com/alpoktem/ted_preprocess

## How does it perform?

On prosodically annotated TED corpus consisting of 1038 talks (155174 sentences): 

PUNCTUATION      | PRECISION | RECALL    | F-SCORE
--- | --- | --- | ---
Comma (,)           | 61.3 | 48.9 | 54.4
Question Mark (?)   | 71.8 | 70.6 | 71.2
Period  (.)        | 82.6 | 83.5 | 83.0
_Overall_        | _73.7_ | _67.3_ | _70.3_

These scores are obtained with a model trained with leveled pause duration and mean f0 features together with word and POS tags. 

## Example Run
* Requirements: 
	- Python 3.x
	- Numpy
	- Theano
	- yaml 

Data directory (path `$datadir`) should look like the output folder (`sample_ted_punc_data`) in https://github.com/alpoktem/ted_preprocess. Vocabularies and sampled training/testing/development sets are stored here. 

Sample run explained here is provided in `run.sh`.

### Training

Training is done on sequenced data stored in `train_samples` under `$datadir`. 

Dataset features to train with are given with the flag `-f`. Other training parameters are specified through the `parameters.yaml` file.
To train with word, pause, POS and mean f0:

`modelId="mod_word-pause-pos-mf0"`

`python main.py -m $modelId -f word -f pause_before -f pos -f f0_mean -p parameters.yaml`

### Testing

Testing is done on proscript data using `punctuator.py`. Either single `<input-file>` or `<input-directory>` is given as input using `-i` or `-d` respectively. Even if there's punctuation information on this data, it is ignored. Predictions for each file in the `$test_samples` directory are put into `$out_preditions` directory. Input files should contain the parameters that the model was trained with. 

`model_name="Model_single-stage_""$modelId""_h100_lr0.05.pcl"`

`python punctuator.py -m Model_single-stage_mod_word-pause-pos-mf0_h100_lr0.05.pcl -d $test_samples -o $out_predictions`

### Scoring the testing output:
Predictions are compared with groundtruth data using `error_calculator.py`. It either takes two files to compare or two directories containing groundtruth/prediction files. Use `-r` for reducing punctuation marks. 

`python error_calculator.py -g $groundtruthData -p $out_predictions -r`

## Citing

More details can be found in the publication: https://link.springer.com/chapter/10.1007/978-3-319-68456-7_11

This work can be cited as:

	@inproceedings{punkProse,
		author = {Alp Oktem and Mireia Farrus and Leo Wanner},
		title = {Attentional Parallel RNNs for Generating Punctuation in Transcribed Speech},
		booktitle = {5th International Conference on Statistical Language and Speech Processing SLSP 2017},
		year = {2017},
		address = {Le Mans, France}
	}
