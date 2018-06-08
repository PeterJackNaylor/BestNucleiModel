
TFRECORD = file('Src/Records.py')
DATA1 = file('../Data/TNBC_NucleiSegmentation')
DATA2 = file('../Data/ForDataGenTrainTestVal')



process CreateRecord {
    input:
    file tnbc from DATA1
    file neeraj from DATA2
    output:
    file "train.tfrecord" into RECORD
    file "test.tfrecord" into RECORDTEST
    """
    python $TFRECORD --test_size 10 --data1 $tnbc --data2 $neeraj --output_train train.tfrecord --output_test test.tfrecord
    """
}
