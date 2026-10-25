"""Streamlit dashboard for the Driver Drowsiness Detection System.
left, right = st.columns([3, 1.25])
            frame_slot, badge_slot, met_slot = left.empty(), right.empty(), right.empty()
            bar = st.progress(0.0)
            rows, i = [], 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out, res = pipe.process_frame(frame, i / fps)
                rows.append(asdict(res))
                if i % 3 == 0:
                    frame_slot.image(out, channels="BGR", use_container_width=True)
                    badge_slot.markdown(badge(res.state), unsafe_allow_html=True)
                    render_metrics(met_slot, res)
                i += 1
                if i % 10 == 0:
                    bar.progress(min(i / total, 1.0))
            bar.progress(1.0)
        finally:
            try:
                tmp.close()
                if cap is not None:
                    cap.release()
                if pipe is not None:
                    pipe.close()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        if not rows:
            st.warning("No frames were read from the video.")
            st.stop()
        st.session_state["log"] = pd.DataFrame(rows)

    df = st.session_state.get("log")
    if df is not None and len(df):
        st.subheader("Results")
        counts = df["state"].value_counts(normalize=True).mul(100).round(1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duration", f"{df['t'].iloc[-1]:.1f} s")
        c2.metric("Time in SLEEPING", f"{counts.get(SLEEPING, 0):.1f} %")
        c3.metric("Time in DROWSY", f"{counts.get(DROWSY, 0):.1f} %")
        c4.metric("Blinks / yawns / nods", f"{df['blinks'].iloc[-1]} / {df['yawns'].iloc[-1]} / {df['nods'].iloc[-1]}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["t"], y=df["ear_smooth"], name="EAR", line=dict(color="#F2A900", width=2)))
        fig.add_hline(y=ear_thresh, line_dash="dash", line_color="#8F9AAB", annotation_text="threshold")
        sleeping = df["state"] == SLEEPING
        if sleeping.any():
            grp = (sleeping != sleeping.shift()).cumsum()
            for _, g in df[sleeping].groupby(grp[sleeping]):
                fig.add_vrect(x0=g["t"].iloc[0], x1=g["t"].iloc[-1], fillcolor="#D2493F", opacity=0.25, line_width=0)
        fig.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10),
                          title="Eye aspect ratio over time (red = SLEEPING)", xaxis_title="seconds")
        st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        dist = df["state"].value_counts().reset_index()
        dist.columns = ["state", "frames"]
        bar_fig = go.Figure(go.Bar(x=dist["state"], y=dist["frames"], marker_color=[STATE_HEX.get(s, "#6B7380") for s in dist["state"]]))
        bar_fig.update_layout(template="plotly_dark", height=280, title="State distribution", margin=dict(l=10, r=10, t=40, b=10))
        a.plotly_chart(bar_fig, use_container_width=True)

        change = df["state"] != df["state"].shift()
        events = df.loc[change, ["t", "state"]].rename(columns={"t": "time (s)"})
        events["time (s)"] = events["time (s)"].round(2)
        b.markdown("**State changes**")
        b.dataframe(events, hide_index=True, height=240, use_container_width=True)

        st.download_button("Download frame-by-frame CSV", df.to_csv(index=False).encode(),
                           file_name="drowsiness_log.csv", mime="text/csv")

# ---------------------------------------------------------------- live mode
elif mode == "Live webcam":
    try:
        import av
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
    except Exception:
        st.error("Live mode needs `streamlit-webrtc` and `av`. Install with: pip install streamlit-webrtc av")
        st.stop()

    class Processor(VideoProcessorBase):
        def __init__(self):
            self.pipe = DrowsinessPipeline(cfg, eye_model, calibrate)
            self.t0 = time.time()
            self.lock = threading.Lock()
            self.latest = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            out, res = self.pipe.process_frame(img, time.time() - self.t0)
            with self.lock:
                self.latest = res
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    st.info("Allow camera access, then press START. Changing settings in the sidebar needs a stop and start.")
    ctx = webrtc_streamer(
        key="drowsiness-live",
        video_processor_factory=Processor,
        rtc_configuration=get_rtc_configuration(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    badge_slot, met_slot, audio_slot = st.empty(), st.empty(), st.empty()
    alarm = Alarm(cfg)
    while ctx.state.playing:
        proc = ctx.video_processor
        res = None
        if proc is not None:
            with proc.lock:
                res = proc.latest
        if res is not None:
            badge_slot.markdown(badge(res.state), unsafe_allow_html=True)
            render_metrics(met_slot, res)
            if alarm_on and alarm.should_sound(res.state, time.time()):
                with audio_slot:
                    components.html(beep_html(), height=0)
        time.sleep(0.25)

# ---------------------------------------------------------------- about
else:
    st.subheader("How it works")

    st.markdown(f"""
1. **MediaPipe Face Mesh** finds 478 facial landmarks on every frame.
2. **Geometry:** Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pitch are calculated from landmarks.
3. **Eye CNN:** Each eye is cropped and classified as open or closed by the CNN model.
4. **Fusion:** An eye counts as closed when EAR and CNN agree, or when CNN confidence is above {cfg.CNN_CONF:.2f} and EAR is borderline.
5. **State machine:** The system transitions from `ALERT` to `DROWSY` and then `SLEEPING`, based on eye closure, PERCLOS, yawns, and head nods.

**Privacy:** Live webcam frames are processed in memory. Uploaded videos are stored temporarily for analysis.
""")

    st.caption(
        "Limitations: Low light, sunglasses, and extreme head angles "
        "can reduce accuracy. This is not a certified safety device."
    )
