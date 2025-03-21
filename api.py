from flask import Flask, render_template, Response, request, session
import cv2
import numpy as np
import random
from mvp import ExerciseTracker  # Import your MVP code

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management

tracker = None  # Exercise tracker instance
filename = ["push-up_3.mp4", "plank_5.mp4", "pull up_1.mp4", "hammer curl_8.mp4", "tricep dips_11.mp4", "tricep pushdown_40.mp4"]
exercises = None
latest_frame = None  # Store the latest received frame


frame_counter = 0  # Initialize the counter globally

@app.route('/receive_frame', methods=['POST'])
def receive_frame():
    global tracker, latest_frame, frame_counter  # Use global to access frame_counter

    if tracker is None:
        return Response("Tracker not initialized", status=500)  # Prevents crashes

    try:
        if not request.data:
            return Response("No frame data received", status=400)

        # Convert raw binary data to NumPy array
        nparr = np.frombuffer(request.data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return Response("Failed to decode image", status=400)

        latest_frame = frame.copy()  # Store the modified frame so it’s used in generate_frames()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = tracker.pose.process(rgb_frame)
        frame = cv2.resize(frame, (640, 480))  # Resize frame to smaller resolution

        if result and result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            angles, keypoints = tracker.get_angles_from_landmarks(landmarks)
            tracker.count_reps(frame, angles, result, landmarks, keypoints)

        # Skip frames by checking the frame_counter
        if frame_counter % 2 == 0:  # Skip every second frame
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Lower quality to speed up encoding
            return Response(buffer.tobytes(), mimetype="image/jpeg")

        # Increment the frame counter for every received frame
        frame_counter += 1

        # Call the generator function to stream frames
        return Response(generate(frame), mimetype='multipart/x-mixed-replace; boundary=frame')

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return Response(f"Server Error: {str(e)}", status=500)


def generate(frame):
    global frame_counter  # Use global here to modify frame_counter
    try:
        # Skip frames by checking the frame_counter
        if frame_counter % 2 == 0:  # Skip every second frame
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Lower quality to speed up encoding
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

        # Increment the frame counter for every received frame
        frame_counter += 1

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n' + str(e).encode() + b'\r\n\r\n')
        
def generate_frames():
    global latest_frame, tracker
    while True:
        if latest_frame is None:
            continue  # Wait until a frame is received

        frame = latest_frame.copy()

        # Convert frame to RGB for pose processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = tracker.pose.process(frame) if tracker else None

        if result and result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            angles = tracker.get_angles_from_landmarks(landmarks)  # Extract angles
            tracker.count_reps(frame, angles, result)  # Process exercise tracking

        # Encode processed frame back to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


@app.route('/')
def index():
    global exercises
    exercises = ["Push-up", "Plank", "Pull-up", "Hammer Curl", "Tricep Dip", "Tricep Pull-down"]
    return render_template('index.html', exercises=exercises)


@app.route('/start_exercise', methods=['GET'])
def start_exercise():
    global tracker
    exercise_id = int(request.args.get('exercise', 0))  # Get from URL params
    session['exercise_id'] = exercise_id
    tracker = ExerciseTracker(exercise_id=exercise_id)  # Initialize exercise tracker
    return render_template('exercise.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_exercise', methods=['POST'])
def stop_exercise():
    global tracker
    reps = tracker.rep_count if tracker else 0
    calories = round(tracker.calories_burned if tracker else 0, 2)
    duration = round(tracker.exercise_duration if tracker else 0, 2)
    return render_template('results.html', reps=reps, calories=calories, duration=duration,
                           exercise_type=tracker.exercise_type, sets_completed=random.randint(0, 6), heart_rate=98)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    print("Video upload attempted")
    return Response("Video upload not implemented yet", status=501)


if __name__ == '__main__':
    app.run(debug=True) #uncomment to use flask development server  
