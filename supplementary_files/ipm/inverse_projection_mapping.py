import numpy as np
from math import sin, cos
import cv2


def do_nothing(x):
    pass


def main():
    capture = cv2.VideoCapture(0)
    PI = 3.1415926

    CALIBRATION_MODE = True
    frameWidth = 1200  # 640
    frameHeight = 720  # 480

    waitkey_delay = 300 if CALIBRATION_MODE else 1
    alpha_ = 90
    beta_ = 90
    gamma_ = 90
    f_ = 500
    dist_ = 500

    cv2.namedWindow("Result")

    cv2.createTrackbar("Alpha", "Result", alpha_, 180, do_nothing)
    cv2.createTrackbar("Beta", "Result", beta_, 180, do_nothing)
    cv2.createTrackbar("Gamma", "Result", gamma_, 180, do_nothing)
    cv2.createTrackbar("f", "Result", f_, 2000, do_nothing)
    cv2.createTrackbar("Distance", "Result", dist_, 2000, do_nothing)

    while(True):
        # Capture frame-by-frame
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (frameWidth, frameHeight))

        alpha_ = cv2.getTrackbarPos('Alpha', 'Result')
        beta_ = cv2.getTrackbarPos('Beta', 'Result')
        gamma_ = cv2.getTrackbarPos('Gamma', 'Result')
        f_ = cv2.getTrackbarPos('f', 'Result')
        dist_ = cv2.getTrackbarPos('Distance', 'Result')

        w, h = frame.shape[1], frame.shape[0]
        alpha = (alpha_ - 90) * PI / 180
        beta = (beta_ - 90) * PI / 180
        gamma = (gamma_ - 90) * PI / 180
        focalLength = f_
        dist = dist_

        print(alpha, beta, gamma, focalLength, dist)

        # Projecion matrix 2D -> 3D
        A1 = np.array(
            [[1, 0, -w / 2],
             [0, 1, -h / 2],
             [0, 0, 0],
             [0, 0, 1]])

        # Rotation matrices Rx, Ry, Rz
        RX = np.array(
            [[1, 0, 0, 0],
             [0, cos(alpha), -sin(alpha), 0],
             [0, sin(alpha), cos(alpha), 0],
             [0, 0, 0, 1]])

        RY = np.array(
            [[cos(beta), 0, -sin(beta), 0],
             [0, 1, 0, 0],
             [sin(beta), 0, cos(beta), 0],
             [0, 0, 0, 1]])

        RZ = np.array(
            [[cos(gamma), -sin(gamma), 0, 0],
             [sin(gamma), cos(gamma), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])

        # R - rotation matrix
        R = RX @ RY @ RZ

        # T - translation matrix
        T = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, dist],
             [0, 0, 0, 1]])
        #
        # # K - intrinsic matrix
        K = np.array(
            [[focalLength, 0, w / 2, 0],
             [0, focalLength, h / 2, 0],
             [0, 0, 1, 0]])

        transformationMat = K @ (T @ (R @ A1))

        frame = cv2.warpPerspective(frame,
                                    transformationMat,
                                    (frameWidth, frameHeight),
                                    cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(waitkey_delay) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


# Running the Program
if __name__ == "__main__":
    main()
