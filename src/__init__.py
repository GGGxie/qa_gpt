import yaml,os
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上层目录的路径
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, "conf"))
config_path = os.path.join(parent_dir, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
print(config)

os.environ["ZHIPUAI_API_KEY"] = config['zhipu_api_key']

__all__ = ['config']
