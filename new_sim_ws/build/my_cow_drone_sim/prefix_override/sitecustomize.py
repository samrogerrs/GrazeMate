import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/sam/GrazeMate/new_sim_ws/install/my_cow_drone_sim'
