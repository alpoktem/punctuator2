dataset_mac='/Users/alp/Desktop/ted_data/data_260118'
dataset_10k='/Users/alp/Desktop/ted_data/ted_data_10k'

hiddensize=100
lr=0.05

#Mac
model="w"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wp"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wpmf"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pause_before -f f0_mean -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
echo "============continuous value=============="
echo "=========================================="
model="wPOS"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmfsr"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f f0_mean -f speech_rate_norm -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSp"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmf" 
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f f0_mean -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSprf"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f f0_range -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmi"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f i0_mean -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpri"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f i0_range -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpsr"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f speech_rate_norm -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmfmi" 
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f f0_mean -f i0_mean -f pause_before -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmfbrf"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -f f0_birange -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmfsrbrf"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -f speech_rate_norm  -f f0_birange -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmfsr2"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -f speech_rate_norm -p parameters1.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "==============all prosodic==============="
model="pmfmi"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f pause_before -f f0_mean -f i0_mean -p parameters1.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSprfri"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f i0_range -f f0_range -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSprfmi"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f i0_mean -f f0_range -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmfri"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -f i0_range -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpsrmfmi"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -f i0_mean -f speech_rate_norm -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "============TWO STAGE================"
model="wp2stage"
predicted_dir="2-stage-wp"
python main.py -m $model -d $dataset_mac -f word -f pause_before -p parameters.yaml -t Model_single-stage_w_h100_lr0.05.pcl 
python punctuator.py -m Model_stage-2_wp2stage_h100_lr0.05.pcl -v /Users/alp/Desktop/ted_data/data_260118/vocabulary.txt -d /Users/alp/Desktop/ted_data/data_260118/test_samples/ -o /Users/alp/Desktop/ted_data/data_260118/predictions/2-stage-wp -t Model_single-stage_w_h100_lr0.05.pcl
python error_calculator.py -g /Users/alp/Desktop/ted_data/data_260118/test_groundtruth/ -p /Users/alp/Desktop/ted_data/data_260118/predictions/2-stage-wp
echo "=========================================="
model="wPOSpsrmfmiri"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f speech_rate_norm -f f0_mean -f i0_mean -f i0_range -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wPOSpmfmiri"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -f i0_mean -f i0_range -p parameters.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "=========================================="
model="wp_leveled"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pause_before -p parameters_leveled.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters_leveled.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "==================degis======================"
model="wpmf_leveled"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pause_before -f f0_mean -p parameters_leveled.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters_leveled.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "==================level======================"
model="wPOSpmf_leveled"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -p parameters_leveled.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters_leveled.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir
echo "==================level======================"
model="wPOSpmf_leveledp3"
predicted_dir=${model}_h${hiddensize}_lr${lr}
python main.py -m $model -d $dataset_mac -f word -f pos -f pause_before -f f0_mean -p parameters_leveled_3.yaml
python punctuator.py -m Model_single-stage_${predicted_dir}.pcl -d $dataset_mac/test_samples/ -o $dataset_mac/predictions/$predicted_dir -p parameters_leveled_3.yaml
python error_calculator.py -g $dataset_mac/test_groundtruth/ -p $dataset_mac/predictions/$predicted_dir


