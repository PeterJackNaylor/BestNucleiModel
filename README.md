# BestNucleiModel
Pipeline for the best nuclei model for histopathology that I can do with Nextflow by baby-sitting the model.

# Running
After installing the necessary dependencies, from the root folder, launch ```nextflow run pipelines/pipeline.nf``` for the models for DIST and ```nextflow run pipelines/pipeline_binary.nf``` for the normal U-Net.

You can add the option ```--normalize 1``` to the nextflow command to perform Vahadane normalisation [1].


[1] Vahadane, Abhishek, et al. "Structure-preserving color normalization and sparse stain separation for histological images." IEEE transactions on medical imaging 35.8 (2016): 1962-1971.