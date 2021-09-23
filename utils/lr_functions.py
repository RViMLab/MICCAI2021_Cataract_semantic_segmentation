import numpy as np
from collections import OrderedDict


class LRFcts:
    def __init__(self, config: dict, lr_restart_steps: list, lr_total_steps: int):
        self.lr_fct = config['lr_fct']
        self.batchwise = config['lr_batchwise']

        # Restart epochs, and base values
        self.lr_restarts = lr_restart_steps
        restart_vals = config['lr_restart_vals']
        if 0 not in self.lr_restarts:
            self.lr_restarts.insert(0, 0)
        self.lr_restart_vals = [1]
        if isinstance(restart_vals, float) or isinstance(restart_vals, int):
            # Base LR value reduced to fraction every restart, end set to 0
            for i in range(1, len(self.lr_restarts)):
                self.lr_restart_vals.append(self.lr_restart_vals[i - 1] * restart_vals)
        elif isinstance(restart_vals, list):
            assert len(restart_vals) == len(config['lr_restarts']) - 1, \
                "Value Error: lr_restart_vals is list, but not the same length as lr_restarts"
            self.lr_restart_vals.extend(restart_vals)
        if lr_total_steps not in self.lr_restarts:
            self.lr_restarts.append(lr_total_steps)
            self.lr_restart_vals.append(0)
        self.lr_restarts = np.array(self.lr_restarts)
        self.lr_restart_vals = np.array(self.lr_restart_vals)

        # Length of each restart
        self.restart_lengths = np.ones_like(self.lr_restarts)
        self.restart_lengths[:-1] = self.lr_restarts[1:] - self.lr_restarts[:-1]

        # Current restart position
        self.curr_restart = len(self.lr_restarts) - \
            np.argmax((np.arange(lr_total_steps + 1)[:, np.newaxis] >= self.lr_restarts)[:, ::-1], axis=1) - 1
        self.lr_params = config['lr_params']

        self.epochs_ulab = config['ulab_epochs'] if 'ulab_epochs' in config else None
        self.epochs_lab = config['lab_epochs'] if 'lab_epochs' in config else None

        if self.lr_fct == 'piecewise_static':
            #  example entry in config['train']["piecewise_static_schedule"]: [[40,1],[50,0.1]]
            # if s<=40 ==> lr = learning_rate * 1 elif s<=50 ==> lr = learning_rate * 0.1
            assert(len(self.lr_restarts) == 2), 'with piecewise_static lr schedule lr_restarts must be empty list' \
                                              ' instead got {}'.format(self.lr_restarts)
            assert 'piecewise_static_schedule' in self.lr_params
            assert isinstance(self.lr_params['piecewise_static_schedule'], list)
            assert self.lr_params['piecewise_static_schedule'][-1][0] == config['epochs'], \
                "piecewise_static_schedule's last phase must have first element equal to number of epochs " \
                "instead got: {} and {} respectively".format(config['piecewise_static_schedule'][-1][0], config['epochs'])

            piecewise_static_schedule = self.lr_params['piecewise_static_schedule']
            self.piecewise_static_schedule = OrderedDict() # this is essential, it has to be an ordered dict
            phase_prev = 0
            for phase in piecewise_static_schedule: # get ordered dict from list
                assert phase_prev < phase[0], ' piecewise_static_schedule must have increasing first elements per phase' \
                                              ' instead got phase_prev {} and phase {}'.format(phase_prev, phase[0])
                self.piecewise_static_schedule[phase[0]] = phase[1]

    def __call__(self, step: int):
        steps_since_restart = step - self.lr_restarts[self.curr_restart[step]]
        base_val = self.lr_restart_vals[self.curr_restart[step]]
        if self.lr_fct == 'static':
            return base_val
        elif self.lr_fct == 'piecewise_static':
            return self.piecewise_static(step)
        elif self.lr_fct == 'exponential':
            return self.lr_exponential(base_val, steps_since_restart)
        elif self.lr_fct == 'polynomial':
            steps_in_restart = self.restart_lengths[self.curr_restart[step]]
            return self.lr_polynomial(base_val, steps_since_restart, steps_in_restart)
        elif self.lr_fct == 'cosine':
            steps_in_restart = self.restart_lengths[self.curr_restart[step]]
            return self.lr_cosine(base_val, steps_since_restart, steps_in_restart)
        else:
            ValueError("Learning rate schedule '{}' not recognised.".format(self.lr_fct))

    def piecewise_static(self, step):
        # important this only works if self.piecewise_static_schedule is an ordered dict!
        for phase_end in self.piecewise_static_schedule.keys():
            lr = self.piecewise_static_schedule[phase_end]
            if step <= phase_end:
                return lr

    def lr_exponential(self, base_val: float, steps_since_restart: int):
        gamma = .98 if self.lr_params is None else self.lr_params
        lr = base_val * gamma ** steps_since_restart
        return lr

    def lr_polynomial(self, base_val: float, steps_since_restart: int, steps_in_restart: int):
        """Based on formula (2) in https://ieeexplore.ieee.org/abstract/document/8929465"""
        power = .9 if self.lr_params is None else self.lr_params
        lr = base_val * (1 - steps_since_restart / steps_in_restart) ** power
        return lr

    def lr_cosine(self, base_val, steps_since_restart, steps_in_restart):
        lr = base_val * 0.5 * (1. + np.cos(np.pi * steps_since_restart / steps_in_restart))
        return lr


if __name__ == '__main__':
    def lr_exponential(base_val: float, steps_since_restart: int, steps_in_restart=None, gamma: int = .98):
        lr = base_val * gamma ** steps_since_restart
        return lr

    def lr_cosine(base_val, steps_since_restart, steps_in_restart):
        lr = base_val * 0.5 * (1. + np.cos(np.pi * steps_since_restart / steps_in_restart))
        return lr

    lr_start = 0.0001
    T = 100
    lrs = [lr_cosine(lr_start, step, T) for step in range(T)]
    lrs_exp = [lr_exponential(lr_start, step % (T//4), T//4) for step in range(T)]
    import matplotlib.pyplot as plt
    plt.plot(lrs)
    plt.plot(lrs_exp)
    plt.show()
    a = 1