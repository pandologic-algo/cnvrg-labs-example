from cnvrg import Experiment, Library

# labs
from labs.experimenting import LocalExperimenter

# system paths
cwd = os.getcwd()
sys.path.append(cwd)

# internal
from research.lgbm_reg import lgbm_reg_cnvrg_api
from research.utils.secrets import slack_token


def run_library(path=''):
    e = Experiment.run(command='python3 research/experimenter_example.py', 
        title='lgbm_reg_dummy_bs', 
        compute=None,
        output_dir='research/artifacts')

    # experimnter_config, lab_config = load_yaml(path=path)

    # experiment = LocalExperimenter(**experimnter_config, **lab_config)

    # experiment.run_experiments(lgbm_reg_cnvrg_api)


def load_yaml(path):
    # load config from yaml

    return {}, {}

def lgbm_reg_cnvrg_api(experiment, artifacts_path, metrics):
    global experiment_file_path

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

    cmd = "python3 {}".format(experiment_file_path)

    # os.system(cmd)
    e = Experiment.run(cmd, 
                       title='lgbm_reg_experiment-{}'.format(experiment.get('ix')),
                       arguments={'experiment_ix': experiment_ix,
                                  'hyperparams': hyperparams_dumped, 
                                  'artifacts_path': artifacts_path, 
                                  'metrics': metrics_dumped},
                       compute='medium',
                       output_dir='research/artifacts',
                       sync_before=False)

  
if __name__ == "__main__":
    experiment_file_path = "research/lgbm_reg/train.py

    run_library()