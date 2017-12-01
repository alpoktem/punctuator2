# punkProse

Punctuation generation for speech transcripts using lexical and prosodic features. 

Modification on forked repository (by reducing training to one stage and addition of more word-level prosodic features) . 

For example data and extraction scripts see: https://github.com/alpoktem/ted_preprocess

## Example Run
Data directory (path `$datadir`) should look like the output folder (sample_ted_punc_data) in https://github.com/alpoktem/ted_preprocess. Word vocabulary list and sampled training/testing/development sets are stored here.

* Training:
Training is done on sequenced data stored in `train_samples` under `$datadir`. 

- Parameters used in example run below: 
	- Prosodic features pause + two semitone feature (f0 range, intensity range), 
	- output: reduced punctuation set (comma, period, question_mark)
	- hidden layer/word embeddings size:100
	- param hidden layer/embeddings size: 10
	- learning rate:0.05
	- maximum sample size (words in one sample) : 50 
	- Batch size: 128

`modelId="mod_rangeF0-rangeI0"`

`python main.py -m $modelId -n 100 -o 10 -l 0.05 -d $datadir -p -f range_f0 -f range_i0 -r -s 50 -b 128`

* Testing:
Testing is done on proscript data using `punctuator.py`. Either single `<input-file>` or `<input-directory>` is given as input using `-i` or `-d` respectively. Even if there's punctuation information on this data, it is ignored. Predictions for each file in the `$test_samples` directory are put into `$out_preditions` directory.

`model_name="Model_single-stage_""$modelId""_h100_lr0.05.pcl"`

`python punctuator.py -m Model_single-stage_mod_rangeF0-rangeI0_h100_lr0.05.pcl -v <word-vocabulary-file> -d $test_samples -o $out_predictions -p -f range_f0 -f range_i0`

* Scoring testing output:
Predictions are compared with groundtruth data using `error_calculator.py`. It either takes two files to compare or two directories containing groundtruth/prediction files. Use `-r` for reducing punctuation marks. 

`python error_calculator.py -g $groundtruthData -p $out_predictions -r`

## Citing

This work is published as:

	@inproceedings{punkProse,
		author = {Alp Oktem and Mireia Farrus and Leo Wanner},
		title = {Attentional Parallel RNNs for Generating Punctuation in Transcribed Speech},
		booktitle = {5th International Conference on Statistical Language and Speech Processing SLSP 2017},
		year = {2017},
		address = {Le Mans, France}
	}
