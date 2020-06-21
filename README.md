# Introduction
This project examples the use of cnvrg with labs package to perform:

1. Grid search
2. Ranadom search
3. Bayesian search with scikit-optimize package

# Execution
- input: python3 research/experimenter_example.py
- git: branch - master | commit - latest
- output folder/dir: research/artifacts

# General info
The current script runs guassian process bayesian search with 4 workers and experiments batch size of 4.

__Experiments batch size__ - each bayesian search iteration will sample 4 experiments which will be executed by 4 workers. These experiments batch will update the bayesian search estimator (gaussian process) and will lead to another experiments batch sampling untill score threshold  or n_experiments limit is reached.

__Experiments artifacts__ - each experiment artifacts is pulled to main process (Experimenter which is the orchestrator).
All the experiments artifacts can be deleted by tune_config param: delete_experiments=True. 

__Artifacts path__ - will be user_defined_artifacts_path/experiment_name/

__Experiments artifacts path__ - user_defined_artifacts_path/experiment_name/experiment_ix/

__Best experiment artifacts path__ - user_defined_artifacts_path/experiment_name/final_model/

__final_model artifacts path__ - include the experiment_ix artifacts, experiment_name_metadata.json which is the experimenter metadata and experiment_name_report.json which is all the experimnets scores report.
