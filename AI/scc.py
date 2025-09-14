import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

class CameraManager:
    def __init__(self, width=1200, height=None):
        self.width = width
        self.height = height if height else round(self.width * 9 / 16)  # 16:9 계산
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)

class InferenceEngine:
    def __init__(self, model_path="final.tflite", seq_length=30, min_conf=0.8):
        self.actions = [
            '안녕하세요', '감사합니다', '사랑합니다', '어머니', '아버지', '동생', '잘', '못', '간다', '나',
            '이름', '만나다', '반갑다', '부탁', '학교', '생일', '월', '일', '나이', '복습', '학습', '눈치', '오다', '말', '곱다',
        ]
        self.seq_length = seq_length
        self.min_conf = min_conf

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        self.seq = []
        self.last_prediction_time = 0
        self.prediction_cooldown = 1.0
        self.current_question = None
        self.game_result = None
        self.confidence_value = None

    def set_question(self, question):
        self.current_question = question.strip() if question else None
        self.game_result = None

    def get_state(self):
        return self.current_question, self.game_result, self.confidence_value

    def process_frame(self, frame):
        if frame is None:
            return None, None, None 

        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.holistic.process(img_rgb)

        joint_list = []

        if result.left_hand_landmarks:
            joint_list.extend([[lm.x, lm.y, lm.z] for lm in result.left_hand_landmarks.landmark])
        else:
            joint_list.extend([[0,0,0]]*21)

        if result.right_hand_landmarks:
            joint_list.extend([[lm.x, lm.y, lm.z] for lm in result.right_hand_landmarks.landmark])
        else:
            joint_list.extend([[0,0,0]]*21)

        if result.pose_landmarks:
            joint_list.extend([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark])
        else:
            joint_list.extend([[0,0,0]]*33)

        joint_list = np.array(joint_list).flatten()
        self.seq.append(joint_list)
        if len(self.seq) > self.seq_length:
            self.seq.pop(0)

        current_time = time.time()
        if len(self.seq) == self.seq_length and (current_time - self.last_prediction_time >= self.prediction_cooldown):
            
            input_data = np.expand_dims(np.array(self.seq), axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            predicted_action = self.actions[np.argmax(prediction)]
            confidence = float(np.max(prediction)) 
            self.confidence_value = confidence

            
            self.game_result = None

            
            if confidence >= self.min_conf:
                if self.current_question:
                    q = self.current_question.strip().lower()
                    p = predicted_action.strip().lower()
                    if p == q:
                        self.game_result = "정답입니다!"
                    else:
                        self.game_result = "틀렸습니다!"
                else:
                    self.game_result = "문제가 출제되지 않았습니다."

                self.last_prediction_time = current_time

            return predicted_action, confidence, self.game_result

        return None, None, None
