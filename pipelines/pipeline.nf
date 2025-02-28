
TFRECORD = file('Src/Records.py')
DATA1 = file('Data/TNBC_NucleiSegmentation')
DATA2 = file('Data/ForDataGenTrainTestVal')
DATA3 = file('Data/DataCPM')

TEST_DATA = file('Data/PickedForTest')

params.normalize = 0

if (params.normalize == 0){
    
    NDATA1 = DATA1
    NDATA2 = DATA2
    NDATA3 = DATA3
    NDATA = TEST_DATA

} else {

    NTN = file("Src/NormalToNormalize.py")

    process Normalize {
        queue "gpu-cbio"
        input:
        file tnbc from DATA1
        file neeraj from DATA2
        file cpm from DATA3
        file test from TEST_DATA
        output:
        file "tnbc_norm" into NDATA1
        file "neeraj_norm" into NDATA2
        file "cpm_norm" into NDATA3
        file "test_norm" into NDATA
        """
        python $NTN $tnbc tnbc_norm $tnbc/Slide_08/08_1.png
        python $NTN $neeraj neeraj_norm $tnbc/Slide_08/08_1.png
        python $NTN $cpm cpm_norm $tnbc/Slide_08/08_1.png
        python $NTN $test test_norm $tnbc/Slide_08/08_1.png
        """
    }
}

process Create_Record_Mean {
    publishDir "./dist_best_model", pattern:"mean_array.npy", copy:true, replace:true
    memory '5GB'
    queue "gpu-cbio"
    input:
    file tnbc from NDATA1
    file neeraj from NDATA2
    file cpm from NDATA3
    file test from NDATA
    output:
    set file("train.tfrecord"), file("test.tfrecord"), file("mean_array.npy") into TRAIN_TEST_MEAN
    """
    module load cuda90
    python $TFRECORD --data1 $tnbc --data2 $neeraj --data3 $cpm --test $test \\
                     --output_train train.tfrecord --output_test test.tfrecord \\
                     --output_mean_array mean_array.npy
    """
}

DISTANCE_TRAIN = file("Src/UNetDistCust.py")
BS = 16
EPOCHS = 160

LEARNING_RATE = [0.01, 0.001, 0.0001, 0.00001]
//LEARNING_RATE = [0.001, 0.0001]
WEGIHT_DECAYS = [0.0005, 0.00005, 0.000005, 0]
// WEGIHT_DECAYS = [0.5, 0.0005, 0]
// NFEATURES = [16, 32, 64]
NFEATURES = [32]

process Training {
    memory '15GB'
    tag { "Training ${lr}__${wd}__${nf}" }
    clusterOptions "--gres=gpu:1"
    queue "gpu-cbio"
    maxForks 10
	input:
	set file(train), file(test), file(mean) from TRAIN_TEST_MEAN
	each lr from LEARNING_RATE
	each wd from WEGIHT_DECAYS
	each nf from NFEATURES
	output:
	file "${lr}__${wd}__${nf}" into LOGS
	"""
    module load cuda90
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
