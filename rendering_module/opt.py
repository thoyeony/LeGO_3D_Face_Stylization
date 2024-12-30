import configargparse
class ClipOpt:
    def __init__(self):
        self.opt_dict = {}
        self.opt_dict['run'] = 'branch'
        self.opt_dict['sigma'] = 5.0
        self.opt_dict['clamp'] = 'tanh'
        self.opt_dict['n_augs'] = 1
        self.opt_dict['normmincrop'] = 0.1
        self.opt_dict['normmaxcrop'] = 0.1
        self.opt_dict['geoloss'] = True
        self.opt_dict['normdepth'] = 2
        self.opt_dict['frontview'] = True
        self.opt_dict['frontview_std'] = 4
        self.opt_dict['clipavg'] = 'view'
        self.opt_dict['lr_decay'] = 0.9
        self.opt_dict['normclamp'] = 'tanh'
        self.opt_dict['maxcrop'] = 1.0
        
        self.opt_dict['save_render'] = True
        self.opt_dict['seed'] = 41
        self.opt_dict['output_dir'] = './text2mesh/save_result'
        self.opt_dict['n_iter'] = 1500
        self.opt_dict['learning_rate'] = 0.0005
        self.opt_dict['normal_learning_rate'] = 0.0005
        
        self.opt_dict['n_normaugs'] = 4
        self.opt_dict['background'] = [1, 1, 1]
        self.opt_dict['frontview_center'] = [0, 0]
        self.opt_dict['n_views'] = 2
        
    def get_opt_dict(self):
        return self.opt_dict
