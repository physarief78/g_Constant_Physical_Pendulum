import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─── PHYSICAL PARAMETERS ─────────────────────────────────────────────
m_rod = 0.46925    # masa batang (kg)
m_bob = 0.00926    # masa beban   (kg)
L     = 0.75       # panjang batang (m)
d     = 0.46       # pivot → COM  (m)

# total mass & moment of inertia about pivot
m_tot   = m_rod + m_bob
I_rod   = (m_rod * L**2) / 3.0
I_bob   = m_bob * L**2
I_tot   = I_rod + I_bob

def track_and_mask_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([50,  90,  90])
    upper_hsv = np.array([95, 255, 255])
    mask_hsv  = cv2.inRange(hsv, lower_hsv, upper_hsv)

    b,g,r = cv2.split(frame)
    mask_bgr = ((g > r + 50) & (g > b + 50)).astype(np.uint8)*255

    mask = cv2.bitwise_or(mask_hsv, mask_bgr)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def main(video_source='video_green_middle.mp4',
         scale=0.6,
         save_outputs=True):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: cannot open '{video_source}'"); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000/fps)

    # grab first frame for dims
    ret, frame = cap.read()
    if not ret:
        print("Error reading first frame."); cap.release(); return
    h,w = frame.shape[:2]
    small_w, small_h = int(w*scale), int(h*scale)

    # prepare optional writers
    if save_outputs:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_orig   = cv2.VideoWriter('output_original_middle.mp4',  fourcc, fps, (small_w, small_h))
        out_mask   = cv2.VideoWriter('output_mask_middle.mp4',      fourcc, fps, (small_w, small_h), isColor=False)
        out_res    = cv2.VideoWriter('output_masked_middle.mp4',    fourcc, fps, (small_w, small_h))
        out_marked = cv2.VideoWriter('output_marked_middle.mp4',    fourcc, fps, (small_w, small_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pivot_px = (320, 50)  # set your hinge pixel here

    com_data = []
    print(f"Recording @ {fps:.1f} FPS, scale={scale}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        mask = track_and_mask_green(small)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marked = small.copy()
        cv2.drawContours(marked, contours, -1, (0,255,0), 2)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = M["m10"]/M["m00"]
            cy = M["m01"]/M["m00"]
            t  = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            com_data.append((cx, cy, t))
            cv2.circle(small, (int(cx),int(cy)), 4, (0,0,255), -1)

        # show & save
        cv2.imshow('orig', small)
        cv2.imshow('mask', mask)
        cv2.imshow('marked', marked)
        if save_outputs:
            out_orig.write(small)
            out_mask.write(mask)
            out_res.write(marked)
            out_marked.write(marked)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    if save_outputs:
        out_orig.release(); out_mask.release()
        out_res.release();  out_marked.release()
        print("Saved videos.")
    cv2.destroyAllWindows()

    # ─── POST‑PROCESSING ───────────────────────────────────────────────
    data = np.array(com_data)
    cx, cy, t = data[:,0], data[:,1], data[:,2]

    # compute θ(t)
    dx = cx - pivot_px[0]
    dy = pivot_px[1] - cy
    theta = np.arctan2(dx, dy)

    print("Middle Marked Center of Mass of Physical Pendulum Results")

    # zero‑cross approach
    sgn = np.sign(theta - theta.mean())
    idx = np.where((sgn[:-1] < 0) & (sgn[1:] > 0))[0]
    t_cross = t[idx]
    halfP = np.diff(t_cross)
    T_zc = 2*np.mean(halfP)
    print(f"Period (zero‑cross) = {T_zc:.4f} s")

    # FFT approach
    theta_dt = theta - np.mean(theta)
    N = len(theta_dt)
    dt = np.mean(np.diff(t))
    freqs = np.fft.rfftfreq(N, dt)
    spectrum = np.abs(np.fft.rfft(theta_dt))
    peak = np.argmax(spectrum[1:]) + 1
    f0   = freqs[peak]
    T_fft = 1.0 / f0
    print(f"Period (FFT peak)    = {T_fft:.4f} s")

    # compute g
    g_zc  = 4*np.pi**2 * I_tot / (m_tot * d * T_zc**2)
    g_fft = 4*np.pi**2 * I_tot / (m_tot * d * T_fft**2)
    print(f"g (zero‑cross) = {g_zc:.4f} m/s²")
    print(f"g (FFT)        = {g_fft:.4f} m/s²")

    # ─── PLOTTING ───────────────────────────────────────────────────────
    plt.figure()
    plt.plot(t, theta, '-')
    plt.xlabel('Time (s)');    plt.ylabel('Angle θ (rad)')
    plt.title('Pendulum Angle vs Time')
    plt.grid(True)
    plt.savefig('pendulum_angle_vs_time.png')

    plt.figure()
    plt.plot(freqs, spectrum, '-')
    plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude')
    plt.title('FFT of θ(t)')
    plt.grid(True)
    plt.savefig('pendulum_fft_spectrum.png')

    plt.show()


if __name__ == '__main__':
    main('video_green_middle.mp4', scale=0.6, save_outputs=True)
