import yaml

class EasyDict(dict):
    """Helper to allow accessing dict keys with dots: d.key instead of d['key']"""
    def __init__(self, d=None, **kwargs):
        if d is None: d = {}
        if kwargs: d.update(kwargs)
        for k, v in d.items():
            setattr(self, k, v)
            if isinstance(v, dict):
                setattr(self, k, EasyDict(v))
    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        setattr(self, name, value)
    def __getitem__(self, name):
        return getattr(self, name)

def load_configs(file_path):
    with open(file_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return EasyDict(config_dict)
