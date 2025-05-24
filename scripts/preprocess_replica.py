import os
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
from pathlib import Path

def to_se3_vec(pose):
    quat = R.from_matrix(pose[:3, :3]).as_quat()
    return np.hstack((pose[:3, 3], quat))


parent = os.path.dirname(__file__)
seqs = sorted(glob('data/Replica/room*')) + sorted(glob('data/Replica/office*'))
seqs = ["/home/farsee2/YZX_code/datasets/Replica/office0/"]
result_data = "./results"
for seq in seqs:
    print(seq)
    base_name = Path(seq).name
    result_path = f'{result_data}/{base_name}'
    os.makedirs(result_path, exist_ok=True)
    os.system(f'rm -rf {result_path}/colors')
    os.system(f'rm -rf {result_path}/depths')
    os.makedirs(f'{result_path}/colors', exist_ok=True)
    os.makedirs(f'{result_path}/depths', exist_ok=True)
    
    for color, depth in zip(glob(f'{seq}/results/frame*'), glob(f'{seq}/results/depth*')):
        color_name = os.path.basename(color)
        depth_name = os.path.basename(depth)

        color_link = os.path.join(result_path, 'colors', color_name)
        depth_link = os.path.join(result_path, 'depths', depth_name)

        color_src = os.path.abspath(color)
        depth_src = os.path.abspath(depth)

        os.symlink(color_src, color_link)
        os.symlink(depth_src, depth_link)

    traj = np.loadtxt(f"{seq}/traj.txt").reshape(-1, 4, 4)
    traj_tum = [np.hstack(([i], to_se3_vec(p))) for i, p in enumerate(traj)]
    np.savetxt(f'{result_path}/traj_tum.txt', traj_tum)
