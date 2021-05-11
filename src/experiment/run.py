#TODO:  File created to read a config and run an experiment


class Experiment():
    def __init__(self,experiment_params = None):
        
        # if experiment_params is None: read json and load params
        self.experiment_params = experiment_params
        
        self.results = None
        self.best_validation_results = None
        self.best_train_results = None
        self.test_results = None

    def run(self):
        pass
    