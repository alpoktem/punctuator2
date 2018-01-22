dataset="/Users/alp/Desktop/ted_data/ted_data_16118"

python themain.py -m the-w -d $dataset -f word -l 0.02
python themain.py -m the-wP -d $dataset -f word -f pos -l 0.02
python themain.py -m the-wp -d $dataset -f word -f pause_before -l 0.02
python themain.py -m the-wPp -d $dataset -f word -f pause_before -f pos -l 0.02
python themain.py -m the-wmf -d $dataset -f word -f f0_mean -l 0.02
python themain.py -m the-wpmf -d $dataset -f word -f pause_before -f f0_mean -l 0.02
python themain.py -m the-wprf -d $dataset -f word -f pause_before -f f0_range -l 0.02
python themain.py -m the-wpmi -d $dataset -f word -f pause_before -f i0_mean -l 0.02
python themain.py -m the-wpri -d $dataset -f word -f pause_before -f i0_range -l 0.02

python themain.py -m the-wPpmf -d $dataset -f word -f pos -f pause_before -f f0_mean -l 0.02

