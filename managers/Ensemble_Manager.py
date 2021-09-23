from managers.BaseManager import BaseManager
from models import Ensemble
import pathlib


class EnsembleManager(BaseManager):
    def load_model(self):
        """Loads the ensemble into self.model and load ckpts for each member"""
        model_class = globals()[self.config['graph']['model']]
        self.model = model_class(config=self.config['graph'], experiment=self.experiment)
        self.model = self.model.to(self.device)
        self.model.load_pretrained(pathlib.Path(self.config['log_path']), self.config['gpu_device'])
        # place all members to device and set eval mode
        for i, model in enumerate(self.model.members):
            self.model.members[i] = model.to(self.device)
            self.model.members[i].eval()
