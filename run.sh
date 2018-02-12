dataset='/Users/alp/Desktop/ted_data/data_260118'

modelid="wPOSpmf" 
python main.py -m $modelid -f word -f pos -f pause_before -f f0_mean -p parameters.yaml
python punctuator.py -m Model_single-stage_${modelid}.pcl -d $dataset/test_samples/ -o $dataset/predictions/$modelid -p parameters.yaml
python error_calculator.py -g $dataset/test_groundtruth/ -p $dataset/predictions/$modelid