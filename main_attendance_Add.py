import threading
from functools import partial
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
import numpy as np
import os
import sys
import subprocess
from datetime import datetime
from PIL import Image
from kivy.core.window import Window
import pandas as pd
from time import sleep

Window.clearcolor = (.8, .8, .8, 1)

# Define screens
class AttendenceWindow(Screen):
    pass

class DatasetWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass

# Explicitly register the classes with the Factory
from kivy.factory import Factory
Factory.register('AttendenceWindow', cls=AttendenceWindow)
Factory.register('DatasetWindow', cls=DatasetWindow)

# Now load the KV file
kv = Builder.load_file("my.kv")

class MainApp(App):
    running = False
    Dir = os.path.dirname(os.path.realpath(__file__))
    msg_thread = None
    att_thread = None
    data_thread = None
    train_thread = None
    msg_clear = True
    msg_timer = 0

    def message_cleaner(self):
        while True:
            if not self.msg_clear:
                while self.msg_timer > 0:
                    sleep(0.25)
                    self.msg_timer -= 0.25
                # Clear message on both screens
                kv.get_screen('main').ids.info.text = ""
                kv.get_screen('second').ids.info.text = ""
                self.msg_clear = True

    def show_message(self, message, screen="both", duration=5):
        if (self.msg_thread is None) or (not self.msg_thread.is_alive()):
            self.msg_thread = threading.Thread(target=self.message_cleaner, daemon=True)
            self.msg_thread.start()
        if screen == "both":
            kv.get_screen('main').ids.info.text = message
            kv.get_screen('second').ids.info.text = message
        elif screen == "main":
            kv.get_screen('main').ids.info.text = message
        elif screen == "second":
            kv.get_screen('second').ids.info.text = message
        self.msg_timer = duration
        self.msg_clear = False

    def build(self):
        self.icon = os.path.join(self.Dir, 'webcam.ico')
        self.title = 'Face Detection Attendance System'
        return kv

    def break_loop(self):
        self.running = False

    def startAttendance(self):
        if self.att_thread is None or not self.att_thread.is_alive():
            self.att_thread = threading.Thread(target=self.Attendance, daemon=True)
            self.att_thread.start()

    def startTrain(self):
        if self.train_thread is None or not self.train_thread.is_alive():
            self.train_thread = threading.Thread(target=self.train, daemon=True)
            self.train_thread.start()

    def startDataset(self):
        if self.data_thread is None or not self.data_thread.is_alive():
            self.data_thread = threading.Thread(target=self.dataset, daemon=True)
            self.data_thread.start()

    def UserList(self):
        users_file = os.path.join(self.Dir, 'list', 'users.csv')
        if not os.path.exists(users_file):
            self.show_message("Users file not found.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(users_file)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, users_file])
        except Exception as e:
            print(e)

    def AttendanceList(self):
        attendance_file = os.path.join(self.Dir, 'Attendance', 'Attendance.csv')
        if not os.path.exists(attendance_file):
            self.show_message("Attendance file not found.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(attendance_file)
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, attendance_file])
        except Exception as e:
            print(e)

    def Attendance(self):
        self.running = True
        dataset_path = os.path.join(self.Dir, 'dataset')
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)
        try:
            try:
                user_id = int(kv.get_screen('main').ids.user_id.text)
            except ValueError:
                self.show_message("Invalid User ID.", "main")
                return

            now = datetime.now()
            date_str = now.strftime("%d/%m/%Y")
            # Load Haar cascades for face and eye detection
            eye_cascade = cv2.CascadeClassifier(os.path.join(self.Dir, 'haarcascade_eye.xml'))
            face_cascade = cv2.CascadeClassifier(os.path.join(self.Dir, 'haarcascade_frontalface_default.xml'))

            recog = cv2.face.LBPHFaceRecognizer_create()
            trainer_file = os.path.join(self.Dir, 'trainer', 'trainer.yml')
            if not os.path.exists(trainer_file):
                self.show_message("Training file not found. Please train the model first.", "main")
                return
            recog.read(trainer_file)
            font = cv2.FONT_HERSHEY_DUPLEX

            rec = 0
            face_numbers = 5
            blink = 0
            no_face_frame_count = 0   # Count consecutive frames with no face
            camera = cv2.VideoCapture(0)
            camera.set(3, 1920)
            camera.set(4, 1080)
            minWidth = int(0.001 * camera.get(3))
            minHeight = int(0.001 * camera.get(4))

            while self.running:
                ret, frame = camera.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=face_numbers,
                    minSize=(minWidth, minHeight)
                )
                eyes = eye_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5)

                if len(faces) > 0:
                    no_face_frame_count = 0  # Reset counter if a face is detected
                    if len(eyes) < 2:
                        blink += 1
                else:
                    no_face_frame_count += 1
                    if no_face_frame_count > 30:
                        blink = 0

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 1)

                cv2.putText(frame, "Blink(16+) : {}".format(blink), (1020, 50), font, 1, (0, 0, 255), 2)

                for (x, y, w, h) in faces:
                    label, confidence = recog.predict(gray[y:y+h, x:x+w])
                    if label == user_id and confidence < 35:
                        rec = 1
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        status = "Attendance Recorded"
                        cv2.putText(frame, status, (x, y+h+25), font, 1, (0, 255, 0), 1)
                        try:
                            users_csv = os.path.join(self.Dir, 'list', 'users.csv')
                            df_users = pd.read_csv(users_csv)
                            name = df_users.loc[df_users['id'] == user_id, 'name'].iloc[0]
                        except Exception:
                            name = "Unknown"
                        conf_text = "  {0}%".format(round(100 - confidence))
                    else:
                        rec = 0
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        status = "Attendance Not Recorded"
                        cv2.putText(frame, status, (x, y+h+25), font, 1, (0, 0, 255), 1)
                        name = "unknown"
                        conf_text = "  {0}%".format(round(100 - confidence))
                    cv2.putText(frame, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
                    cv2.putText(frame, conf_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)

                Clock.schedule_once(partial(self.display_frame, frame))
                if cv2.waitKey(1) == 27:  # ESC key to exit loop
                    break

            # Update attendance CSV: add today's date column (default "Absent") and update user status
            attendance_file = os.path.join(self.Dir, 'Attendance', 'Attendance.csv')
            try:
                df_att = pd.read_csv(attendance_file)
            except FileNotFoundError:
                df_att = pd.DataFrame(columns=['id', 'name'])
            try:
                users_csv = os.path.join(self.Dir, 'list', 'users.csv')
                df_users = pd.read_csv(users_csv)
                name = df_users.loc[df_users['id'] == user_id, 'name'].iloc[0]
            except Exception:
                name = "Unknown"
            if date_str not in df_att.columns:
                df_att[date_str] = "Absent"
            if user_id not in df_att['id'].values:
                new_row = pd.DataFrame([{'id': user_id, 'name': name, date_str: "Absent"}])
                df_att = pd.concat([df_att, new_row], ignore_index=True)
            if rec == 1 and blink > 15:
                df_att.loc[df_att['id'] == user_id, date_str] = "Present"
                self.show_message("Attendance Recorded Successfully.")
            else:
                df_att.loc[df_att['id'] == user_id, date_str] = "Absent"
                self.show_message("Attendance not recorded; marked as absent.")
            df_att.to_csv(attendance_file, index=False)
            print("Attendance file updated at:", attendance_file)
            camera.release()
            cv2.destroyAllWindows()
        except Exception as e:
            self.show_message("Some error occurred. Try again!", "main")
            print(e)
            return

    def display_frame(self, frame, dt):
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        texture.flip_vertical()
        kv.get_screen('main').ids.vid.texture = texture

    def dataset(self):
        dataset_path = os.path.join(self.Dir, 'dataset')
        list_path = os.path.join(self.Dir, 'list')
        attendance_path = os.path.join(self.Dir, 'Attendance')
        for path in [dataset_path, list_path, attendance_path]:
            if not os.path.isdir(path):
                os.mkdir(path)
        try:
            name = kv.get_screen('second').ids.user_name.text.strip()
            face_id_text = kv.get_screen('second').ids.user_id.text.strip()
            snap_text = kv.get_screen('second').ids.snap.text.strip()
            if not name or not face_id_text or not snap_text:
                kv.get_screen('second').ids.info.text = "All fields are required."
                return
            try:
                face_id = int(face_id_text)
                snap_amount = int(snap_text)
            except ValueError:
                kv.get_screen('second').ids.info.text = "User ID and Snap Amount must be numbers."
                return
            camera = cv2.VideoCapture(0)
            camera.set(3, 1920)
            camera.set(4, 1080)
            face_cascade = cv2.CascadeClassifier(os.path.join(self.Dir, 'haarcascade_frontalface_default.xml'))
            count = 0
            while True:
                ret, frame = camera.read()
                if not ret:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    count += 1
                    image_path = os.path.join(dataset_path, f"{name}_{face_id}_{count}.jpg")
                    cv2.imwrite(image_path, gray[y:y+h, x:x+w])
                    cv2.imshow('Dataset Capture', frame)
                key = cv2.waitKey(10) & 0xff
                if key == 27 or count >= snap_amount:
                    break
            camera.release()
            cv2.destroyAllWindows()
            # Update users.csv file
            users_csv = os.path.join(list_path, 'users.csv')
            try:
                df_users = pd.read_csv(users_csv)
            except FileNotFoundError:
                df_users = pd.DataFrame(columns=['id', 'name'])
            if face_id not in df_users['id'].values:
                new_row = pd.DataFrame([{'id': face_id, 'name': name}])
                df_users = pd.concat([df_users, new_row], ignore_index=True)
                df_users.to_csv(users_csv, index=False)
                print("Updated users.csv at", users_csv)
            # Update Attendance.csv file
            attendance_csv = os.path.join(attendance_path, 'Attendance.csv')
            try:
                df_att = pd.read_csv(attendance_csv)
            except FileNotFoundError:
                df_att = pd.DataFrame(columns=['id', 'name'])
            if face_id not in df_att['id'].values:
                new_row = pd.DataFrame([{'id': face_id, 'name': name}])
                for col in df_att.columns:
                    if col not in ['id', 'name']:
                        new_row[col] = "Absent"
                df_att = pd.concat([df_att, new_row], ignore_index=True)
                df_att.to_csv(attendance_csv, index=False)
                print("Attendance file updated at:", attendance_csv)
            self.show_message("Dataset Created Successfully. Training model...", "second")
            # Automatically train the model after dataset creation
            self.startTrain()
        except Exception as e:
            self.show_message("Some error occurred. Please try again.", "second")
            print(e)
            return

    def getImage_Labels(self, dataset, face_cascade):
        imagesPath = [os.path.join(dataset, f) for f in os.listdir(dataset)]
        faceSamples = []
        ids = []
        if not imagesPath:
            return None, None
        for imagePath in imagesPath:
            try:
                PIL_img = Image.open(imagePath).convert('L')
            except Exception:
                continue
            img_numpy = np.array(PIL_img, 'uint8')
            try:
                face_id = int(os.path.split(imagePath)[-1].split("_")[1])
            except (IndexError, ValueError):
                continue
            faces = face_cascade.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(face_id)
        return faceSamples, ids

    def train(self):
        dataset_path = os.path.join(self.Dir, 'dataset')
        trainer_path = os.path.join(self.Dir, 'trainer')
        if not os.path.isdir(dataset_path):
            self.show_message("No dataset available.")
            return
        if not os.path.isdir(trainer_path):
            os.mkdir(trainer_path)
        kv.get_screen('main').ids.info.text = "Training Faces..."
        kv.get_screen('second').ids.info.text = "Training Faces..."
        sleep(3)
        kv.get_screen('main').ids.info.text = ""
        kv.get_screen('second').ids.info.text = ""
        try:
            recog = cv2.face.LBPHFaceRecognizer_create()
            face_cascade = cv2.CascadeClassifier(os.path.join(self.Dir, 'haarcascade_frontalface_default.xml'))
            faces, ids = self.getImage_Labels(dataset_path, face_cascade)
            if faces is None or ids is None or len(faces) == 0:
                self.show_message("No dataset available for training.")
                return
            recog.train(faces, np.array(ids))
            recog.write(os.path.join(trainer_path, 'trainer.yml'))
            unique_faces = len(np.unique(ids))
            self.show_message(f"{unique_faces} face(s) trained.")
        except Exception as e:
            self.show_message("Some error occurred during training. Try again!")
            print(e)
            return

if __name__ == "__main__":
    MainApp().run()
