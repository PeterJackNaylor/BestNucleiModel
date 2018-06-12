
TFRECORD = file('Src/Records.py')
DATA1 = file('../Data/TNBC_NucleiSegmentation')
DATA2 = file('../Data/ForDataGenTrainTestVal')



process CreateRecord {
    queue "gpu-cbio"
    input:
    file tnbc from DATA1
    file neeraj from DATA2
    output:
    set file("train.tfrecord"), file("test.tfrecord") into RECORD
    """
    python $TFRECORD --test_size 10 --data1 $tnbc --data2 $neeraj --output_train train.tfrecord --output_test test.tfrecord
    """
}


COMPUTE_MEAN = file("Src/ComputeMean.py")

process MeanFile {
        queue "gpu-cbio"
	input:
	set file(train), file(test) from RECORD
	output:
	set file(train), file(test), file("mean_file.npy") into TRAIN_TEST_MEAN
	"""
	python $COMPUTE_MEAN $train $test --output mean_file.npy
	"""
}

DISTANCE_TRAIN = file("Src/UNetDistCust.py")
BS = 16
EPOCHS = 80
LEARNING_RATE = [0.01, 0.001, 0.001]
WEGIHT_DECAYS = [0.0005, 0.00005, 0.000005]
NFEATURES = [16, 32, 64]

process Training {
        tag { "Training ${lr}__${wd}__${nf}" }
        queue "gpu-cbio"
	input:
	set file(train), file(test), file(mean) from TRAIN_TEST_MEAN
	each lr from LEARNING_RATE
	each wd from WEGIHT_DECAYS
	each nf from NFEATURES
	output:
	file "${lr}__${wd}__${nf}" into LOGS
	"""
        export CUDA_VISIBLE_DEVICES=0
        source $HOME/init_gpu
	python $DISTANCE_TRAIN --log ${lr}__${wd}__${nf} --learning_rate $lr --weight_decay $wd --n_features $nf --epochs $EPOCHS --batch_size $BS --train_record $train --test_record $test --mean_file $mean
	"""

}
