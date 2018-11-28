
from numpy import load
from segmentation_net import DistanceUnet, PangNet

def GetOptions():
    import argparse
    parser = argparse.ArgumentParser(
        description='Training Distance')
    parser.add_argument('--train_record', required=True,
                        metavar="str", type=str,
                        help='train_record')
    parser.add_argument('--test_record', required=True,
                        metavar="str", type=str,
                        help='test_record')
    parser.add_argument('--learning_rate', required=True,
                        metavar="float", type=float,
                        help='learning rate')
    parser.add_argument('--epochs', required=True,
                        metavar="int", type=int,
                        help='Number of epochs to perform')  
    parser.add_argument('--batch_size', required=True,
                        metavar="int", type=int,
                        help='batch size')
    parser.add_argument('--log', required=True,
                        metavar="folder", type=str,
                        help='Log folder to dump weights and other things')
    parser.add_argument('--weight_decay', required=True,
                        metavar="float", type=float,
                        help='Weight decay value to be applied')
    parser.add_argument('--mean_file', required=False,
                        metavar="/path/to/mean_file/",
                        help='Path to the mean file to use')
    parser.add_argument('--n_features', required=True,
                        metavar="int", type=int,
                        help='Complexity of the architecture in terms of filters')
    args = parser.parse_args()
    return args


def main():

    args = GetOptions()

    variables_model = {
        ## Model basics

        "image_size": (212, 212),
        "log": args.log, 
        "num_channels": 3,
#        "num_labels": 2, #remove from distance
        'mean_array': load(args.mean_file),
        "seed": None, 
        "verbose": 1,
        "fake_batch": args.batch_size,
        "n_features": args.n_features
    }

    # model = PangNet(**variables_model)
    model = DistanceUnet(**variables_model)

    variables_training = {
        ## training:

        'learning_rate' : args.learning_rate,
        'lr_procedure' : "10epoch", # the decay rate will be reduced every 5 epochs
        'weight_decay': args.weight_decay,
        'batch_size' : args.batch_size, # batch size for the
        'decay_ema' : None, #0.9999, #
        'k' : 0.96, # exponential decay factor
        'n_epochs': args.epochs, # number of epochs
        'early_stopping' : 3, # when to stop training, 20 epochs of non progression
        'save_weights' : True, # if to store as final weights the best thanks to early stopping
        'num_parallele_batch' : 8, # number batch to run in parallel (number of cpu usually)
        'restore' : False, # allows the model to be restored at training phase (or re-initialized)
        "tensorboard": True,
        "track_variable": "f1_score"
    }

    _ = model.train(args.train_record, args.test_record,  **variables_training) #,  **variables_training) #
    model.sess.close() 


if __name__ == '__main__':
    main()
