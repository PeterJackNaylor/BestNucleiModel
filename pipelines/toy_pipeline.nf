
TFRECORD = file('Src/Records.py')
DATA1 = file('Data/UNNormalized/')

TEST_DATA = file('Data/UNNormalized_test/')

process Create_Record_Mean {
    input:
    file data1 from DATA1
    file test from TEST_DATA
    output:
    set file("train.tfrecord"), file("test.tfrecord"), file("mean_array.npy") into TRAIN_TEST_MEAN
    """
    python $TFRECORD --data1 $data1 --test $test \\
                     --output_train train.tfrecord --output_test test.tfrecord \\
                     --output_mean_array mean_array.npy
    """
}

DISTANCE_TRAIN = file("Src/UNetDistCust.py")
BS = 8
EPOCHS = 20
LEARNING_RATE = [0.01]//, 0.001, 0.0001, 0.00001]
WEGIHT_DECAYS = [5]//, 0.5, 0.05, 0.005, 0.0005, 0.00005, 0.000005]
NFEATURES = [2]

process Training {
    tag { "Training ${lr}__${wd}__${nf}" }
    input:
    set file(train), file(test), file(mean) from TRAIN_TEST_MEAN
    each lr from LEARNING_RATE
    each wd from WEGIHT_DECAYS
    each nf from NFEATURES
    output:
    file "${lr}__${wd}__${nf}" into LOGS
    """
    python $DISTANCE_TRAIN --log ${lr}__${wd}__${nf} --learning_rate $lr --weight_decay $wd \\
                           --n_features $nf --epochs $EPOCHS --batch_size $BS --train_record $train \\
                           --test_record $test --mean_file $mean
    """
}

FINAL_SCRIPT = file("Src/final_script.py")

process GiveBest {
    if (params.normalize == 0){
        publishDir "./dist_best_model", copy:true, replace:true
    } else {
        publishDir "./dist_best_model_normalized", copy:true, replace:true
    }
    input:
    file _ from LOGS .collect()
    output:
    file "best"
    file "recap.csv"
    """
    python $FINAL_SCRIPT
    """
}
