# coding: utf-8

"""
Computes and prints the overall classification error and precision, recall, F-score over punctuations.
"""

from numpy import nan
import codecs
import sys
import os
from utilities import reducePunc, END, SPACE
from optparse import OptionParser

PUNCTUATION_VOCABULARY = {SPACE, ",", ".", "?", "!", "-", ";", ":"}

def compute_error(target_paths, predicted_paths, reduce_punctuation):
    counter = 0
    total_correct = 0

    correct = 0.
    substitutions = 0.
    deletions = 0.
    insertions = 0.

    true_positives = {}
    false_positives = {}
    false_negatives = {}

    for target_path, predicted_path in zip(target_paths, predicted_paths):

        target_punctuation = SPACE
        predicted_punctuation = SPACE

        t_i = 0
        p_i = 0

        with codecs.open(target_path, 'r', 'utf-8') as target, codecs.open(predicted_path, 'r', 'utf-8') as predicted:

            target_stream = target.read().split() + [END]
            predicted_stream = predicted.read().split() + [END]
            # print("TARGET")
            # print(target_stream)
            # print("PREDICTED")
            # print(predicted_stream)
            
            while True:
                # print("%s - %s"%(target_stream[t_i], predicted_stream[p_i]))
                if target_stream[t_i] in PUNCTUATION_VOCABULARY:
                    while target_stream[t_i] in PUNCTUATION_VOCABULARY: # skip multiple consecutive punctuations
                        target_punctuation = target_stream[t_i]
                        target_punctuation = target_punctuation
                        t_i += 1
                else:
                    target_punctuation = SPACE
                   

                if predicted_stream[p_i] in PUNCTUATION_VOCABULARY:
                    predicted_punctuation = predicted_stream[p_i]
                    p_i += 1
                else:
                    predicted_punctuation = SPACE

                if reduce_punctuation:
                    reduced_punctuation = reducePunc(target_punctuation)
                    #print("was %s now %s"%(target_punctuation, reduced_punctuation))
                    target_punctuation = reduced_punctuation

                # print ("target:|%s|\tpredicted:|%s|"%(target_punctuation, predicted_punctuation))                
                is_correct = target_punctuation == predicted_punctuation
                # print(is_correct)

                counter += 1 
                total_correct += is_correct

                if predicted_punctuation == SPACE and target_punctuation != SPACE:
                    deletions += 1
                elif predicted_punctuation != SPACE and target_punctuation == SPACE:
                    insertions += 1
                elif predicted_punctuation != SPACE and target_punctuation != SPACE and predicted_punctuation == target_punctuation:
                    correct += 1
                elif predicted_punctuation != SPACE and target_punctuation != SPACE and predicted_punctuation != target_punctuation:
                    substitutions += 1

                true_positives[target_punctuation] = true_positives.get(target_punctuation, 0.) + float(is_correct)
                false_positives[predicted_punctuation] = false_positives.get(predicted_punctuation, 0.) + float(not is_correct)
                false_negatives[target_punctuation] = false_negatives.get(target_punctuation, 0.) + float(not is_correct)

                assert target_stream[t_i] == predicted_stream[p_i] or predicted_stream[p_i] == "<unk>", \
                       ("File: %s \n" + \
                       "Error: %s (%s) != %s (%s) \n" + \
                       "Target context: %s \n" + \
                       "Predicted context: %s") % \
                       (target_path,
                        target_stream[t_i], t_i, predicted_stream[p_i], p_i,
                        " ".join(target_stream[t_i-2:t_i+2]),
                        " ".join(predicted_stream[p_i-2:p_i+2]))

                t_i += 1
                p_i += 1

                if t_i >= len(target_stream)-1 and p_i >= len(predicted_stream)-1:
                    break

    overall_tp = 0.0
    overall_fp = 0.0
    overall_fn = 0.0

    print "-"*46
    print "{:<16} {:<9} {:<9} {:<9}".format('PUNCTUATION','PRECISION','RECALL','F-SCORE')
    for p in PUNCTUATION_VOCABULARY:

        if p == SPACE:
            continue

        overall_tp += true_positives.get(p,0.)
        overall_fp += false_positives.get(p,0.)
        overall_fn += false_negatives.get(p,0.)

        punctuation = p
        precision = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_positives[p])) if p in false_positives else nan
        recall = (true_positives.get(p,0.) / (true_positives.get(p,0.) + false_negatives[p])) if p in false_negatives else nan
        f_score = (2. * precision * recall / (precision + recall)) if (precision + recall) > 0 else nan        
        print "{:<16} {:<9} {:<9} {:<9}".format(punctuation, round(precision,3)*100, round(recall,3)*100, round(f_score,3)*100)
    print "-"*46
    pre = overall_tp/(overall_tp+overall_fp) if overall_fp else nan
    rec = overall_tp/(overall_tp+overall_fn) if overall_fn else nan
    f1 = (2.*pre*rec)/(pre+rec) if (pre + rec) else nan
    print "{:<16} {:<9} {:<9} {:<9}".format("Overall", round(pre,3)*100, round(rec,3)*100, round(f1,3)*100)
    print "Err: %s%%" % round((100.0 - float(total_correct) / float(counter-1) * 100.0), 2)
    print "SER: %s%%" % round((substitutions + deletions + insertions) / (correct + substitutions + deletions) * 100, 1)

def main(options):

    if options.groundtruth_path:
        target_path = options.groundtruth_path
    else:
        sys.exit("Ground truth file path argument missing")

    if options.predictions_path:
        predicted_path = options.predictions_path
    else:
        sys.exit("Model predictions file path argument missing")

    if os.path.isdir(target_path) and os.path.isdir(predicted_path):
        target_paths = [os.path.join(target_path, f) for f in os.listdir(target_path)]
        predicted_paths = [os.path.join(predicted_path, f) for f in os.listdir(predicted_path)]
    else:
        target_paths = [target_path]
        predicted_paths = [predicted_path]

    compute_error(target_paths, predicted_paths, options.reduced_punctuation) 

if __name__ == "__main__": 
	usage = "usage: %prog [-s infile] [option]"
	parser = OptionParser(usage=usage)
	parser.add_option("-g", "--groundtruth", dest="groundtruth_path", default=None, help="Groundtruth file or directory", type="string")
	parser.add_option("-p", "--predictions", dest="predictions_path", default=None, help="Predicted file or directory", type="string")
	parser.add_option("-r", "--reduced_punctuation", dest="reduced_punctuation", default=True, help="Use reduced punctuation vocabulary", action="store_true")

	(options, args) = parser.parse_args()
	main(options)

    