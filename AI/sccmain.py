from flask import Flask, Response, jsonify, request
import random
import pymysql
import cv2
from flask_cors import CORS
from scc import CameraManager, InferenceEngine

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

camera = CameraManager()
inference = InferenceEngine()

@app.route('/get_question', methods=['GET'])
def game1_get_question():
    question = random.choice(inference.actions)
    inference.set_question(question)
    return jsonify({"question": question})

@app.route('/get_game_info', methods=['GET'])
def game1_get_game_info():
    question, result, confidence = inference.get_state()

    # numpy.float32 → Python float 변환
    if confidence is not None:
        confidence = float(confidence)

    print("👉 게임 상태:", question, result, confidence)

    return jsonify({
        "question": question,
        "game_result": result,
        "confidence": confidence
    })

@app.route('/save_incorrect', methods=['POST'])
def save_result():
    data = request.json
    user_id = data.get('user_id')
    user_lesson_id = data.get('userLesson_id')
    confidence = data.get('confidence')
    result = data.get('result')

    check_answer = True if "정답" in result else False #수정
    check_accuracy = int(confidence * 100)

    try:
        conn = pymysql.connect(
            host='database-sonsu.c3gig4u4qamm.ap-northeast-2.rds.amazonaws.com',
            user='user',
            password='useruser',
            db='db_sonsu',
            charset='utf8'
        )
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO games (userLesson_id, user_id, check_answer, check_accuracy)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(sql, (user_lesson_id, user_id, check_answer, check_accuracy))
        conn.commit()
        return jsonify({"status": "success"})
    except Exception as e:
        print("DB 저장 실패:", e)
        return jsonify({"status": "error", "message": str(e)})
    finally:
        conn.close()

import mediapipe as mp

def generate_frames_game1(target_width=1080, target_height=607):  # 16:9
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = inference.mp_holistic

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        # 동작 인식
        predicted_action, confidence = inference.process_frame(frame)

        # Mediapipe Holistic 처리
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = inference.holistic.process(img_rgb)

        # 관절 그리기 (손 + 팔 + 몸)
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2)
            )

        if result.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,128,0), thickness=2, circle_radius=2)
            )

        if result.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,0,128), thickness=2, circle_radius=2)
            )

        # 영상 크기 조정
        frame = cv2.resize(frame, (target_width, target_height))

        # JPEG 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def game1_video_feed():
    return Response(generate_frames_game1(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_confidence')
def get_confidence_route():
    return jsonify({'confidence': inference.confidence_value})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
