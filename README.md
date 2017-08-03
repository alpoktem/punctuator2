# punkProse

Punctuation generation for speech transcripts using lexical and prosodic features. 

Modification on forked repository (by reducing training to one stage and addition of more word-level prosodic features. 

For obtaining training data see: https://github.com/TalnUPF/ted_preprocess

## Example Run
Data directory (path $datadir) should look like the output folder in https://github.com/TalnUPF/ted_preprocess

* Training:
Training is done on sequenced data stored in train.pickle under $datadir. 
prosodic features: pause + two semitone feature (f0 range, intensity range), output: reduced punctuation set (comma, period, question_mark)
hidden layer/word embeddings size:100, param hidden layer/embeddings size: 10, learning rate:0.05
`modelId="mod_rangeF0-rangeI0"`
`python main.py -m $modelId -n 100 -o 10 -l 0.05 -d $datadir -p -f range.f0.id -f range.i0.id -r`

* Testing:
Testing is done on continuous data. 
`model_name="Model_single-stage_""$modelId""_h100_lr0.05.pcl"`
`out_predictions="predicted_""$modelId"".txt"`
`python punctuator.py -m Model_single-stage_mod_rangeF0-rangeI0_h100_lr0.05.pcl -i $datadir -o $out_predictions -p -f range.f0.id -f range.i0.id`

* Scoring testing output:
`python error_calculator.py $datadir/groundtruth/data1.txt $out_predictions`

## Citing

This work will be published in:

	@inproceedings{punkProse,
		author = {Alp Oktem and Mireia Farrus and Leo Wanner},
		title = {Attentional Parallel RNNs for Generating Punctuation in Transcribed Speech},
		booktitle = {5th International Conference on Statistical Language and Speech Processing SLSP 2017},
		year = {2017}}