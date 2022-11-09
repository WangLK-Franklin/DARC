from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic


env = create_env(env_id='pyth_lq', lq_config='s4a2')

check_dynamic(env, traj_num=3, 
                log_policy_dir='./results/INFADP/s4a2',
                policy_iteration='100000')