import cv2
import mediapipe as mp
import numpy as np
import base64

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

# Initialize Mediapipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def process_frame(image_data: str) -> str:
    """
    Process a Base64-encoded image to detect facial landmarks and determine if the person is focused.
    """
    try:
        if image_data.startswith("data:"):
            header, image_data = image_data.split(",", 1)
        img_bytes = base64.b64decode(image_data)
    except Exception as e:
        return f"Error: Decoding Base64 failed - {e}"

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return "Error: Invalid frame"

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return "Pose Not Detected"
    
    for face_landmarks in results.multi_face_landmarks:
        image_points = np.array([
            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),
            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h),
            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h)
        ], dtype="double")
        
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1)
        ])
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            # Convert rotation vector to rotation matrix.
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            # Decompose the rotation matrix to get Euler angles.
            retval, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
            # retval contains the Euler angles (pitch, yaw, roll) in degrees.
            euler_angles = np.array(retval).flatten() # Flatten to ensure a 1D array.
            pitch = float(euler_angles[0])
            yaw   = float(euler_angles[1])
            roll  = float(euler_angles[2])
            
            # Iris tracking
            left_iris_x = face_landmarks.landmark[468].x * w
            right_iris_x = face_landmarks.landmark[473].x * w
            nose_x = face_landmarks.landmark[1].x * w
            
            iris_position = (left_iris_x + right_iris_x) / 2 - nose_x
            
            return pitch, yaw, roll, iris_position

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that accepts Base64-encoded frames from a client,
    processes them, and sends back a focus status.
    """
    
    await websocket.accept()
    try:
        calibration_frames = 10
        calibration_angles = []
        while True:
            # Receive a Base64-encoded image from the client.
            image_data = await websocket.receive_text()
            result_base = process_frame(image_data)

            if(type(result_base)==type('abc')):
                status = result_base

            else:
                if calibration_frames != 0:
                    calibration_frames -= 1
                    calibration_angles.append(result_base)

                    if calibration_frames == 0:
                        # Compute average baseline angles.
                        baseline_pitch = np.mean([angle[0] for angle in calibration_angles])
                        baseline_yaw = np.mean([angle[1] for angle in calibration_angles])
                        baseline_roll = np.mean([angle[2] for angle in calibration_angles])
                    status = "Calculating Face Angles"
                    
                else:
                    pitch, yaw, roll, iris_position = result_base
                    # Now you can compute differences relative to a baseline.
                    relative_pitch = abs(pitch - baseline_pitch)
                    relative_yaw = abs(yaw - baseline_yaw)
                    relative_roll = abs(roll - baseline_roll)
                    
                    if relative_pitch < 20 and relative_yaw < 20 and relative_roll < 20:
                        if abs(iris_position) < 2 or abs(iris_position)>10:
                            status="Not Focused"
                        else:
                            status = "Focused"
                    else:
                        status = "Not Focused"
            
            await websocket.send_text(status)
    except WebSocketDisconnect:
        print("Client disconnected")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_certfile="/home/b123127/ssl/cert.pem",
        ssl_keyfile="/home/b123127/ssl/key.pem"
    )
