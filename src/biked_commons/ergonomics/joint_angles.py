import numpy as np
import torch
from scipy.stats import norm

USE_DICT = {
    "road": {
        "opt_knee_angle": (37.5, 10),
        "opt_back_angle": (45, 5),
        "opt_awrist_angle": (90, 5),
        "opt_ankle_angle": (100.0, 5.0),
    },
    "mtb": {
        "opt_knee_angle": (37.5, 10),
        "opt_back_angle": (50, 5),
        "opt_awrist_angle": (90, 5),
        "opt_ankle_angle": (100.0, 5.0),
    },
    "commute": {
        "opt_knee_angle": (37.5, 10),
        "opt_back_angle": (52, 5),
        "opt_awrist_angle": (85, 5),
        "opt_ankle_angle": (100.0, 5.0),
    },
}


# #####
# Bike Ergonomic Angle Fit Calculator
# Calculates body angles given bike and body measurements
# Calculates ergonomic score/probability of fit given use case
######


#############################################
# Masking Functions for Triangle Inequality #
#############################################

def deg_to_r(deg: torch.Tensor) -> torch.Tensor:
    """
    Converts degrees to radians (works with tensors or scalars).
    """
    return deg.to(dtype=torch.float32) * (torch.pi / 180)


def rad_to_d(rad: torch.Tensor) -> torch.Tensor:
    """
    Converts radians to degrees (works with tensors or scalars).
    """
    return rad.to(dtype=torch.float32) * (180 / torch.pi)



def validity_mask(bikes, bodies, arm_angle):
    """
    Input: bikes, bodies, arm_angles matricies n x 5, n x 8, nx1
    Output: n x 1 True/False mask for valid/invalid
    TRUE = Violation
    FALSE = NO violation
    Checks for triangle inequality for:
        Leg
        Arm/torso
    """
    ### Mask invalid arm/torso combos ###
    # Straight line distance between saddle position and handlebar position
    bike_reach = np.sqrt(np.square(-bikes[:, 0:1] - bikes[:, 3:4]) + np.square(bikes[:, 1:2] - bikes[:, 4:5]))
    # Armlength considering arm bent at elbow
    functional_arms = np.sqrt(
        np.square(bodies[:, 3:4] / 2) + np.square(bodies[:, 3:4] / 2) - 2 * np.square(bodies[:, 3:4] / 2) * np.cos(
            deg_to_r(arm_angle)))
    # Mask where sum of any 2 sides are greater than 3rd side for all side combos
    mask_upper = (functional_arms + bodies[:, 2:3] > bike_reach) & (functional_arms + bike_reach > bodies[:, 2:3]) & (
                bodies[:, 2:3] + bike_reach > functional_arms)

    ### Mask invalid leg combos ###
    # Functional low leg length (considering ankle angle and foot length)
    a_2 = np.square(bodies[:, 4:5])
    b_2 = np.square(bodies[:, 0:1])
    ab_cos = 2 * bodies[:, 4:5] * bodies[:, 0:1] * np.cos(bodies[:, 5:6] * (np.pi / 180))
    funcitonal_low_leg = np.sqrt(a_2 + b_2 - ab_cos)

    # Straightline distance from seat to pedal at longest part of pedal stroke
    straightline_seat = np.sqrt(np.square(bikes[:, 0:1]) + np.square(bikes[:, 1:2])) + bikes[:, 4:5]

    # Mask where sum of any 2 sides are greater than 3rd side for all side combos
    mask_lower = (funcitonal_low_leg + bodies[:, 1:2] > straightline_seat) & (
                funcitonal_low_leg + straightline_seat > bodies[:, 1:2]) & (
                             bodies[:, 1:2] + straightline_seat > funcitonal_low_leg)

    # Combine and return masks
    return ~(np.logical_and(mask_upper, mask_lower))


###################################
# FUNCTIONS FOR CALCUATING ANGLES #
###################################
def knee_extension_angle(bike_vectors, body_vectors, CA, ret_a2=False):
    """
    Input:
        bike vector, body vector, crank angle, OPTIONAL return a2
        np array bike vector:
            [SX, SY, HX, HY, CL]^T
            (seat_x, seat_y, hbar_x, hbar_y, crank len)
            Origin is bottom bracket
        np array body vector:
            [LL, UL, TL, AL, FL, AA]
            (lowleg, upleg, torso len, arm len, foot len, ankle angle)
        CA = crank angle:
            crank angle fom horizontal in DEGREES
        Optional return a2:
            returns a2 for use in kneeoverpedal check

    Output:
        Knee extension angle in radians
        OR
        None if not valid coords (i.e. NaA appeared)

    """
    CA = deg_to_r(CA)

    # Body parts
    UL = body_vectors[:, 0:1]
    LL = body_vectors[:, 1:2]
    AL = body_vectors[:, 2:3]
    TL = body_vectors[:, 3:4]
    FL = body_vectors[:, 4:5]
    AA = deg_to_r(body_vectors[:, 6:7])
    EA = deg_to_r(body_vectors[:, 7:8])

    # Precomputed squares
    LL_s = LL ** 2
    UL_s = UL ** 2
    TL_s = TL ** 2
    AL_s = AL ** 2
    FL_s = FL ** 2

    HX = bike_vectors[:, 0:1]  # Hand x
    HY = bike_vectors[:, 1:2]  # Hand y
    SX = bike_vectors[:, 2:3] * -1  # Hip x (flip because convension here is positive x is forward on the bike)
    SY = bike_vectors[:, 3:4]  # Hip y
    CL = bike_vectors[:, 4:5]  # Crank length

    # Law of cosines for ankle
    x_1 = torch.sqrt(LL_s + FL_s - 2 * LL * FL * torch.cos(AA))

    LX = CL * torch.cos(CA) - SX
    LY = SY - CL * torch.sin(CA)
    x_2 = torch.sqrt(LX ** 2 + LY ** 2)

    alpha_1 = torch.arccos(torch.clamp((x_1 ** 2 - UL_s - x_2 ** 2) / (-2 * UL * x_2), -1.0, 1.0))
    alpha_2 = torch.atan2(LY, LX) - alpha_1
    if ret_a2:
        return alpha_2

    LLY = LY - UL * torch.sin(alpha_2)
    LLX = LX - UL * torch.cos(alpha_2)

    alpha_3 = torch.atan2(LLY, LLX) - alpha_2
    alpha_4 = torch.arccos(torch.clamp((FL_s - LL_s - x_1 ** 2) / (-2 * LL * x_1), -1.0, 1.0))

    return (alpha_3 + alpha_4) * (180 / torch.pi)


def back_armpit_angles(bike_vectors, body_vectors):
    """
    Input: bike_vector, body_vector, elbow_angle in degrees
    Output: back angle, armpit to elbow angle, armpit to wrist angle in degrees

    np array bike vector:
            [SX, SY, HX, HY, CL]^T
    np array body vector:
            [LL, UL, TL, AL, FL, AA]
    """
    UL = body_vectors[:, 0:1]
    LL = body_vectors[:, 1:2]
    AL = body_vectors[:, 2:3]
    TL = body_vectors[:, 3:4]
    FL = body_vectors[:, 4:5]
    AA = body_vectors[:, 6:7] * (torch.pi / 180)
    EA = body_vectors[:, 7:8] * (torch.pi / 180)

    HX = bike_vectors[:, 0:1]  # Hand x
    HY = bike_vectors[:, 1:2]  # Hand y
    SX = bike_vectors[:, 2:3] * -1  # Hip x (flip because convension here is positive x is forward on the bike)
    SY = bike_vectors[:, 3:4]  # Hip y
    CL = bike_vectors[:, 4:5]  # Crank length

    sth_dx = HX - SX
    sth_dy = HY - SY
    sth_dist = torch.sqrt(sth_dx**2 + sth_dy**2)
    sth_ang = torch.atan2(sth_dy, sth_dx)

    # Law of cosines to simulate elbow bend
    x_1 = (AL / 2) ** 2 + (AL / 2) ** 2 - 2 * (AL / 2) * (AL / 2) * torch.cos(EA)

    tors_ang = torch.arccos(torch.clamp((TL ** 2 + sth_dist ** 2 - x_1) / (2 * TL * sth_dist), -1.0, 1.0))
    back_angle = tors_ang + sth_ang

    armpit_to_wrist = torch.arccos(torch.clamp((TL ** 2 + x_1 - sth_dist ** 2) / (2 * TL * torch.sqrt(x_1)), -1.0, 1.0))

    return back_angle * (180 / torch.pi), armpit_to_wrist * (180 / torch.pi)


def all_angles(bike_vectors, body_vectors):
    """
    Input: bike, body, arm angle (at elbow) in degrees
    Output: tuple (min_ke angle, back angle, awrist angle) in degrees
    """
    N = body_vectors.shape[0]
    device = body_vectors.device

    # Append default ankle (90 deg) and elbow (20 deg)
    body_vectors = torch.cat((
        body_vectors,
        torch.full((N, 1), 90.0, device=device),
        torch.full((N, 1), 20.0, device=device)
    ), dim=1)

    cur_min = knee_extension_angle(bike_vectors, body_vectors, torch.tensor(90.0))

    for test_ca in range(180, 360): #TODO revisit this and try to eliminate if possible
        cur_test = knee_extension_angle(bike_vectors, body_vectors, torch.tensor(test_ca))
        cur_min = torch.minimum(cur_min, cur_test)

    ke_ang = cur_min
    b_angs, aw_angs = back_armpit_angles(bike_vectors, body_vectors)

    return torch.cat((ke_ang, b_angs, aw_angs), dim=1)


############################
# FUNCTIONS FOR SCORING #
############################

def adjusted_nll(bike_vectors: torch.Tensor, body_vectors: torch.Tensor, use_vec: list[str]) -> torch.Tensor:
    """
    Adjusted NLL = log(pdf at optimal) - log(pdf at observed).
    This ensures that the minimum is zero at the optimal angle.
    Output: (N, 3) torch tensor: [knee_nll, back_nll, awrist_nll]
    """
    angles = all_angles(bike_vectors, body_vectors)  # (N, 3)

    means = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][0], USE_DICT[u]["opt_back_angle"][0], USE_DICT[u]["opt_awrist_angle"][0]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    stds = torch.tensor([
        [USE_DICT[u]["opt_knee_angle"][1], USE_DICT[u]["opt_back_angle"][1], USE_DICT[u]["opt_awrist_angle"][1]]
        for u in use_vec
    ], dtype=angles.dtype, device=angles.device)

    # Log-PDF at observed values
    log_pdf_actual = -((angles - means) ** 2) / (2 * stds ** 2) - torch.log(stds * torch.sqrt(torch.tensor(2 * torch.pi, device=angles.device)))

    # Log-PDF at the optimal (i.e., mean), which is the peak of the Gaussian
    log_pdf_optimal = - torch.log(stds * torch.sqrt(torch.tensor(2 * torch.pi, device=angles.device)))

    # Difference gives adjusted NLL
    adjusted_nll = log_pdf_optimal - log_pdf_actual  # = 0 at the peak
    return adjusted_nll


