import cv2
import numpy as np
import matplotlib.pyplot as plt

def track_and_mask_green(frame):
    # 1) Convert to HSV and make a wide “green” mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([50,  90,  90])
    upper_hsv = np.array([95, 255, 255])
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 2) Also pick up pixels where G channel is significantly brighter than R and B
    b, g, r = cv2.split(frame)
    mask_bgr = ((g > r + 50) & (g > b + 50)).astype(np.uint8) * 255

    # 3) Combine both masks
    mask = cv2.bitwise_or(mask_hsv, mask_bgr)

    # 4) Clean up with morphology
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5) Apply mask to get the isolated‑green result
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return mask, result

def main(video_source='video_green.mp4',
         scale=0.6,
         save_outputs=True):
    # Open video / camera
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: cannot open video source '{video_source}'")
        return

    # Get FPS & frame delay
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_delay = int(1000 / fps)

    # Read one frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read first frame.")
        cap.release()
        return
    h, w = frame.shape[:2]
    small_w, small_h = int(w * scale), int(h * scale)

    # Prepare VideoWriters if saving
    if save_outputs:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_orig   = cv2.VideoWriter('output_original.mp4',  fourcc, fps, (small_w, small_h))
        out_mask   = cv2.VideoWriter('output_mask.mp4',      fourcc, fps, (small_w, small_h), isColor=False)
        out_res    = cv2.VideoWriter('output_masked.mp4',    fourcc, fps, (small_w, small_h))
        out_marked = cv2.VideoWriter('output_marked.mp4',    fourcc, fps, (small_w, small_h))

    # Reset to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # USER‑DEFINED PIVOT (in scaled pixels)
    pivot_px = (320, 50)    # set hinge pixel here

    com_data = []  # will hold (cx, cy, t_sec)
    print(f"Capturing '{video_source}' @ {fps:.1f} FPS, scale={scale}")

    # Capture loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream.")
            break

        small, result = None, None
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        mask, result = track_and_mask_green(small)

        # draw contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marked = small.copy()
        cv2.drawContours(marked, contours, -1, (0,255,0), 2)

        # compute centroid
        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]
            t  = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            com_data.append((cx, cy, t))
            cv2.circle(small, (int(cx), int(cy)), 5, (0,0,255), -1)

        # show windows
        cv2.imshow('Original', small)
        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)
        cv2.imshow('Outlined', marked)

        # save video outputs
        if save_outputs:
            out_orig.write(small)
            out_mask.write(mask)
            out_res.write(result)
            out_marked.write(marked)

        # quit on 'q'
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    if save_outputs:
        out_orig.release()
        out_mask.release()
        out_res.release()
        out_marked.release()
        print("Saved: output_original.mp4, output_mask.mp4, output_masked.mp4, output_marked.mp4")
    cv2.destroyAllWindows()

    # POST‑PROCESSING: compute θ(t), T and g
    data = np.array(com_data)
    cx, cy, t = data[:,0], data[:,1], data[:,2]

    # angle time series
    dx = cx - pivot_px[0]
    dy = pivot_px[1] - cy
    theta = np.arctan2(dx, dy)

    print("Full Cover Marked Physical Pendulum Results")

    # zero-crossings method
    sgn = np.sign(theta - theta.mean())
    idx = np.where((sgn[:-1] < 0) & (sgn[1:] > 0))[0]
    t_cross = t[idx]
    half_periods = np.diff(t_cross)
    T_zc = 2 * np.mean(half_periods)
    print(f"Period (zero‑cross) = {T_zc:.4f} s")

    # FFT method
    theta_dt = theta - np.mean(theta)
    N = len(theta_dt)
    dt = np.mean(np.diff(t))
    freqs = np.fft.rfftfreq(N, dt)
    spectrum = np.abs(np.fft.rfft(theta_dt))
    peak = np.argmax(spectrum[1:]) + 1
    f0 = freqs[peak]
    T_fft = 1.0 / f0
    print(f"Period (FFT) = {T_fft:.4f} s")

    # compute g from simple pendulum: g = 4π² d / T²
    # if you know d in meters, replace `d_m` below
    d_m = np.mean(np.sqrt((cx - pivot_px[0])**2 + (pivot_px[1] - cy)**2))  # in pixels
    # you must convert pixels→meters via your calibration s = L_real/L_px
    # here we'll assume you calibrated: s = real_length_m / L_px
    # then d = d_m * s
    # For demonstration, set:
    s_cal = 0.75 / 300.0  # real 0.75 m corresponds to 300 px
    d = d_m * s_cal
    g_zc  = 4 * np.pi**2 * d / T_zc**2
    g_fft = 4 * np.pi**2 * d / T_fft**2
    print(f"g (zero‑cross) = {g_zc:.4f} m/s²")
    print(f"g (FFT)        = {g_fft:.4f} m/s²")

    # PLOTTING & SAVING FIGURES
    plt.figure()
    plt.plot(t, theta, '-')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle θ (rad)')
    plt.title('Pendulum Angle vs Time')
    plt.grid(True)
    plt.savefig('pendulum_angle_vs_time_full.png')

    plt.figure()
    plt.plot(freqs, spectrum, '-')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT of θ(t)')
    plt.grid(True)
    plt.savefig('pendulum_fft_spectrum_full.png')

    plt.show()


if __name__ == '__main__':
    main('video_green.mp4', scale=0.6, save_outputs=True)
