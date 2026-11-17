# The scoring program compute scores from:
# - The ground truth
# - The predictions made by the candidate model

###########
# IMPORTS #
###########

import json
import os
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
import shutil
import glob
from sys import argv


## EVO IMPORTS
from evo.core import metrics
import evo.core.sync as sync
import evo.core.trajectory as evotraj
# Quaternion multiplication
from evo.core.transformations import quaternion_multiply as quatmult
# Conversion between quat and rotm
from evo.core.transformations import quaternion_matrix as quat2rotm_
from evo.core.transformations import quaternion_from_matrix
# Conversion between eul angles and rotm
from evo.core.transformations import euler_matrix as eul2rotm_
from evo.core.transformations import euler_from_matrix as rotm2eul_





####################
# HELPER FUNCTIONS #
####################

def makeSpline(x):
    from ceva import Ceva
    with open(x) as file:
        header = file.readline()
        # Split the string by comma
        fields = header.split(", ")
        configs = {}
        for field in fields:
            kv = field.split(": ")
            configs[kv[0]] = float(kv[1])
        return Ceva(int(configs["Order"]), configs["Dt"], configs["MinTime"], x)

def load_gtr_dataframe(gt_dir):
    gtr_logs = glob.glob(f"{gt_dir}/*/spline_log.csv", recursive=True)
    df_gtr = pd.DataFrame(gtr_logs, columns=["log"])

    # Decode the sequence name
    df_gtr["seq"] = df_gtr["log"].apply(lambda x: os.path.basename(os.path.dirname(x)))
    df_gtr["spline"] = df_gtr["log"].apply(lambda x: makeSpline(x))
    print(df_gtr["seq"])
    return df_gtr

def calculate_metric(traj_gtr, traj_est):
    metric = metrics.APE(pose_relation=metrics.PoseRelation.translation_part)
    metric.process_data((traj_gtr, traj_est))
    return float(
        metric.get_result(ref_name="reference", est_name="estimate").stats["rmse"]
    )

def write_file(file, content):
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)


# Coversion among rotation formulisms
def eul2rotm(e0, e1, e2):
    return eul2rotm_(e0, e1, e2, 'rzyx')[0:3, 0:3]

def eul2quat(e0, e1, e2):
    return rotm2quat(eul2rotm(e0, e1, e2))

def quat2rotm(q):
    return quat2rotm_(q)[0:3, 0:3]

def quat2eul(q):
    return rotm2eul(quat2rotm(q))

def rotm2eul(M):
    return rotm2eul_(M, 'rzyx')

def rotm2quat(M):
    return quaternion_from_matrix(M)

# Converting (x y z qx qy qz qw) into T
def pose2Tf(pose):
    q = pose[0, [6, 3, 4, 5]]
    t = np.reshape(pose[0, [0, 1, 2]], (3, 1))
    return Qt2Tf(q, t)

def pose2Rt(pose):
    pose_ = np.reshape(pose, (1, 7))
    R = quat2rotm(pose_[0, [6, 3, 4, 5]])
    t = np.reshape(pose_[0, [0, 1, 2]], (3, 1))
    return R, t

def Tf2Qt(T):
    q = rotm2quat(T[0:3, 0:3])
    t = T[0:3, 3:4]
    return q, t

def Qt2Tf(q, t):
    T = np.identity(4)
    T[0:3, 0:3] = quat2rotm(q)
    T[0:3, 3:4] = t
    return T

def Rt2Tf(R, t):
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3:4] = t
    return T

def Tf2Rt(T):
    return (T[0:3, 0:3], T[0:3, 3:4])

def tfinv(T):
    R = T[0:3, 0:3].copy()
    t = T[0:3, 3:4].copy()
    R = R.T
    t = np.dot(R, t)*(-1)
    return Rt2Tf(R, t)

def tfmult(T1, T2):
    
    R1 = T1[0:3, 0:3].copy()
    t1 = T1[0:3, 3:4].copy()

    R2 = T2[0:3, 0:3].copy()
    t2 = T2[0:3, 3:4].copy()

    R = np.dot(R1, R2)
    t = np.dot(R1, t2) + t1
    return Rt2Tf(R, t)

def resetdir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=False)


def print_bar():
    print('-' * 10)


# NOT USED CURRENTLY
def make_figure(scores):
    x = None
    y = [scores[dataset] for dataset in x]
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_ylabel('accuracy')
    ax.set_title('Submission results')
    return fig

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    fig_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return fig_b64


##############
# EVALUATION #
##############
def evaluate_seq(test, p_West_Best, q_West_Best, T_B_Best, spline):
    R_B_Best = T_B_Best[0:3, 0:3]
    q_Best_B = rotm2quat(R_B_Best.T)
    p_B_Best = T_B_Best[0:3, 3]

    # Find index of the data that are not in the interval
    inidx = (spline.minTime() <= test) & (test <= spline.maxTime())

    # Extract the valid SE3 trajetory
    test = test[inidx]
    p_West_Best = p_West_Best[inidx, :]
    q_West_Best = q_West_Best[inidx, :]

    # Convert the Best to grountruth body
    p_West_B = np.empty_like(p_West_Best)
    q_West_B = np.empty_like(q_West_Best)
    for idx in range(len(test)):
        q_West_B[idx, :] = quatmult(q_West_Best[idx, :], q_Best_B)
        p_West_B[idx, :] = (
            p_West_Best[idx, :].T - quat2rotm(q_West_B[idx, :]) @ p_B_Best
        ).T

    # Sample the gtr
    gtr_pose_stamped = spline.getPose(test)
    tgtr = gtr_pose_stamped[:, 0]
    p_W_B = gtr_pose_stamped[:, 1:4]
    q_W_B = gtr_pose_stamped[:, [7, 4, 5, 6]]

    # Calculate the error
    
    traj_est = evotraj.PoseTrajectory3D(
        positions_xyz=p_West_B, orientations_quat_wxyz=q_West_B, timestamps=test
    )

    # Get the traj gtr
    traj_gtr = evotraj.PoseTrajectory3D(
        positions_xyz=p_W_B, orientations_quat_wxyz=q_W_B, timestamps=tgtr
    )

    # Associate (formality, we have already sampled the gtr using the measurement time)
    traj_est, traj_gtr = sync.associate_trajectories(traj_est, traj_gtr, max_diff=0.1)

    # Calculate the error
    rmse = calculate_metric(traj_est, traj_gtr)

    return (traj_est, traj_gtr, rmse)

def evaluate_all(input_dir, gt_dir) -> tuple[dict, dict]:    
    # load gt as dataframe
    df_gtr = load_gtr_dataframe(gt_dir)
    
    # enumerate sequences for which gt is available
    rmse = dict()
    results = dict()
    for seq in df_gtr["seq"].to_list():
        
        path_to_traj_data = glob.glob(f"{input_dir}/est_{seq}.csv")
        # path_to_traj_data = glob.glob(f"{gt_dir}/*/sampled_10Hz.csv", recursive=True)
        xtrz_tf = np.eye(4)
        traj_data = np.loadtxt(
            path_to_traj_data[0], 
            delimiter=",",
            skiprows=1 # first row containts column names
        )
        ts = traj_data[:, 0]
        pose = traj_data[:, 1:4]
        quat = traj_data[:, [7, 4, 5, 6]]

        est_traj, gt_traj, err = evaluate_seq(
            test=ts,                                                # time stamps from SLAM
            p_West_Best=pose,                                       # pose estimate from SLAM
            q_West_Best=quat,                                       # quaternion estimate from SLAM
            T_B_Best=xtrz_tf,                                       # frame rotation
            spline=df_gtr[df_gtr["seq"] == seq]["spline"].iloc[0]   # turn gt into spline
        )
        rmse[seq] = err
        results[seq] = (est_traj, gt_traj)
    return rmse, results


def main():

    input_dir = argv[1]
    output_dir = argv[2]

    #########
    # PATHS #
    #########
    input_dir = os.path.join('/app/input', 'res')    # Input from ingestion program
    gt_dir = os.path.join('/app/input', 'ref')
    output_dir = '/app/output/' # To write the scores
    score_file = os.path.join(output_dir, 'scores.json')          # Scores
    html_file = os.path.join(output_dir, 'detailed_results.html') # Detailed feedback

    print_bar()
    print('Evaluation of SLAM.')
    write_file(html_file, '<h1>Detailed results</h1>') # Create the file to give real-time feedback
    

    # Evaluate
    scores, results = evaluate_all(input_dir, gt_dir)
    print_bar()
    print('Scoring program finished. Writing scores.')
    print(scores)
    write_file(score_file, json.dumps(scores))

    # Create a figure for detailed results
    # figure = fig_to_b64(make_figure(scores))
    # write_file(html_file, f'<img src="data:image/png;base64,{figure}">')

if __name__ == '__main__':
    main()
