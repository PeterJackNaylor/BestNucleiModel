#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pdb
import numpy as np
import pandas as pd
import os

class EarlyStopper(object):
    def __init__(self, track_variable, output, saver, session, log,
                       maximum=10, operation='positive', 
                       eps=10**-4):
        self.DataCollector = pd.DataFrame()
        self.track = track_variable
        self.output_name = output
        self.eps = eps
        self.early_stopping_max = maximum
        self.EARLY_STOP = False
        self.log = log
        self.saver = saver
        self.sess = session
        assert operation in ['positive', 'negative']
    def EarlyStop(self, data_res, variable_name, eps):
        """
        Checks on a pandas table if to early stop with respect to the variable_name
        """
        vec_values = np.array(data_res[variable_name])
        if len(vec_values) > self.early_stopping_max:
            val_to_beat = vec_values[-(self.early_stopping_max + 1)]
            values_to_check = vec_values[-self.early_stopping_max:]
            is_one_better = ( values_to_check -  val_to_beat > eps ).any()
            return not is_one_better
        else:
            return False

    def DataCollectorStopper(self, values, names, step, path_var="wgt_path"):
        """
        Collects and fills a table for two given list (names and values) 
        and keeps track of one variable in the list.
        Do be used in the for loop training
        """

        for val, name in zip(values, names):
            self.DataCollector.loc[step, name] = val
        if self.EarlyStop(self.DataCollector, self.track, self.eps):
            best_wgt = np.array(self.DataCollector[path_var])[-(self.early_stopping_max + 1)]
            self.saver.restore(self.sess, best_wgt)
            self.saver.save(self.sess, self.log + '/' + "model.ckpt", step + 1000)
            self.EARLY_STOP = True
            self.DataCollector.to_csv(self.output_name)
        return self.EARLY_STOP

