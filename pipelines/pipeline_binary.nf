
TFRECORD = file('Src/RecordsBinary.py')
DATA1 = file('Data/TNBC_NucleiSegmentation')
DATA2 = file('Data/ForDataGenTrainTestVal')

TEST_DATA = file('Data/PickedForTest')

params.normalize = 0

if (params.normalize == 0){
    
    NDATA1 = DATA1
    NDATA2 = DATA2
    NDATA = TEST_DATA

} else {

    NTN = file("Src/NormalToNormalize.py")

    process Normalize {
        queue "gpu-cbio"
        input:
        file tnbc from DATA1
        file neeraj from DATA2
        file test from TEST_DATA
        output:
        file "tnbc_norm" into NDATA1
        file "neeraj_norm" into NDATA2
        file "test_norm" into NDATA
        """
        python $NTN $tnbc tnbc_norm $tnbc/Slide_08/08_1.png
        python $NTN $neeraj neeraj_norm $tnbc/Slide_08/08_1.png
        python $NTN $test test_norm $tnbc/Slide_08/08_1.png
        """
    }
}

process Create_Record_Mean {
    memory '5GB'
    queue "gpu-cbio"
    input:
    file tnbc from NDATA1
    file neeraj from NDATA2
    file test from NDATA
    output:
    set file("train.tfrecord"), file("test.tfrecord"), file("mean_array.npy") into TRAIN_TEST_MEAN
    """
    source $HOME/init_gpu
    python $TFRECORD --data1 $tnbc --data2 $neeraj --test $test \\
                     --output_train train.tfrecord --output_test test.tfrecord \\
                     --output_mean_array mean_array.npy
    """
}

BINARY_TRAIN = file("Src/UNetTrain.py")
BS = 16
EPOCHS = 80
LEARNING_RATE = [0.001, 0.0001]
WEGIHT_DECAYS = [0.0005, 0.00005, 0.000005]
NFEATURES = [16, 32, 64]

process Training {
    memory '2GB'
    tag { "Training ${lr}__${wd}__${nf}" }
    clusterOptions "--gres=gpu:1"
    queue "gpu-cbio"
    input:
    set file(train), file(test), file(mean) from TRAIN_TEST_MEAN
    each lr from LEARNING_RATE
    each wd from WEGIHT_DECAYS
    each nf from NFEATURES
    output:
    file "${lr}__${wd}__${nf}" into LOGS
    """
    source $HOME/init_gpu
    python $BINARY_TRAIN --log ${lr}__${wd}__${nf} --learning_rate $lr --weight_decay $wd \\
                         --n_features $nf --epochs $EPOCHS --batch_size $BS --train_record $train \\
                         --test_record $test --mean_file $mean
    """
}

FINAL_SCRIPT = file("Src/final_script.py")

process GiveBest {
    if (params.normalize == 0){
        publishDir "./unet_best_model", copy:true, replace:true
    } else {
        publishDir "./unet_best_model_normalized", copy:true, replace:true
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
