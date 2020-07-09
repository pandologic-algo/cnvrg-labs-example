from cnvrg import Experiment, Library


def run_library():
    e = Experiment.run(command='python3 research/experimenter_example.py', 
        title='lgbm_reg_dummy_bs', 
        compute=None,
        output_dir='research/artifacts')

if __name__ == "__main__":
    run_library()