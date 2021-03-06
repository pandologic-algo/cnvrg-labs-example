import numpy as np
import json
import os
from cnvrg import Experiment


def lgbm_reg_cnvrg_api(experiment, artifacts_path, metrics):
    # type handling when saving json (numpy types)
    def default(o):
        if isinstance(o, np.int) or isinstance(o, np.int16) or isinstance(o, np.int32) or isinstance(o, np.int64):
            return int(o)
        if isinstance(o, np.float) or isinstance(o, np.float16) or isinstance(o, np.float32) or \
                isinstance(o, np.float64):
            return float(o)
        raise TypeError
    
    experiment_ix = experiment.get('ix')
    hyperparams_dumped = json.dumps(experiment.get('hyperparams'), default=default)
    metrics_dumped = json.dumps(metrics, default=default)

    # cmd = "python3 research/lgbm_reg/train.py --experiment '{}' --artifacts_path '{}' --metrics '{}'".format(experiment_dumped,
    #                                                                                                          artifacts_path,
    #                                                                                                          metrics_dumped)

    cmd = "python3 research/lgbm_reg/train.py"

    # os.system(cmd)
    e = Experiment.run(cmd, 
                       title='lgbm_reg_experiment-{}'.format(experiment.get('ix')),
                       arguments={'experiment_ix': experiment_ix,
                                  'hyperparams': "'{}'".format(hyperparams_dumped), 
                                  'artifacts_path': artifacts_path, 
                                  'metrics': "'{}'".format(metrics_dumped)},
                       compute='medium',
                       output_dir='research/artifacts',
                       sync_before=False)

    e.pull_artifacts(wait_until_success=True)