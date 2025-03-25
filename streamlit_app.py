import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import date, datetime
import joblib
import base64
from sklearn.neighbors import KNeighborsClassifier

# Set paths for data storage
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)
datetoday = date.today().strftime("%m_%d_%y")
csv_path = f'Attendance/Attendance-{datetoday}.csv'

# Initialize CSV if not present
if not os.path.exists(csv_path):
    pd.DataFrame(columns=["Name", "Roll", "Time", "Date", "Branch"]).to_csv(csv_path, index=False)

# Load face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract faces
def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    return faces

# Train face recognition model
def train_model():
    faces, labels = [], []
    
    print("Training on the following students:")
    
    for user in os.listdir('static/faces'):
        user_path = f'static/faces/{user}'
        if not os.path.isdir(user_path):
            continue

        print(f"➡ {user}")  # Print each student being trained
        
        for imgname in os.listdir(user_path):
            img = cv2.imread(f"{user_path}/{imgname}")
            if img is None:
                print(f"❌ Skipping corrupted image: {imgname}")
                continue

            faces.append(cv2.resize(img, (50, 50)).ravel())
            labels.append(user)

    if len(faces) < 2:
        print("❌ Not enough faces to train. Add more images!")
        return

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(np.array(faces), labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')
    
    print("✅ Model training complete!")

# Mark attendance
def add_attendance(name):
    parts = name.split('_')
    if len(parts) < 2:
        return  

    username, userid = parts[0], parts[1]
    userbranch = parts[2] if len(parts) > 2 else "UNKNOWN"
    current_time = datetime.now().strftime("%H:%M:%S")
    today_date = date.today().strftime("%d-%B-%Y")

    df = pd.read_csv(csv_path)
    if str(userid) not in df['Roll'].astype(str).values:
        pd.DataFrame([[username, userid, current_time, today_date, userbranch]], 
                     columns=['Name', 'Roll', 'Time', 'Date', 'Branch']).to_csv(csv_path, mode='a', header=False, index=False)
        st.success(f"✅ Attendance Added: {username} ({userid}) at {current_time}")

# Calculate attendance percentages
def calculate_attendance_percentages(start_date_str):
    start_date = pd.to_datetime(start_date_str, format='%d-%B-%Y')
    all_files = [f for f in os.listdir('Attendance') if f.endswith('.csv')]

    attendance_data = pd.DataFrame()
    for file in all_files:
        file_path = os.path.join('Attendance', file)
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%B-%Y')
        attendance_data = pd.concat([attendance_data, df])

    df_filtered = attendance_data[attendance_data['Date'] >= start_date]
    total_days = len(df_filtered['Date'].unique())
    
    if total_days == 0:
        return pd.DataFrame(columns=['Roll', 'Name', 'Percentage'])

    # Group by Roll and Name to calculate attendance percentage
    attendance_percentage = df_filtered.groupby(['Roll', 'Name']).size() / total_days * 100
    attendance_percentage = attendance_percentage.reset_index().rename(columns={0: 'Percentage'})
    
    return attendance_percentage

# Custom CSS for styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit UI
def main():
    # Load custom CSS
    local_css("style.css")

    # Page title and description
    st.markdown(
        """
        <div class="header">
            <h1>Face Recognition Attendance System</h1>
           
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar menu
    menu = ["Home", "Register", "Mark Attendance", "View Attendance"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Page
    if choice == "Home":
        st.markdown(
            """
            <div class="home">
                <h2>Welcome to the Attendance System</h2>
                <p>This system allows you to:</p>
                <ul>
                    <li>Register new students using facial recognition.</li>
                    <li>Mark attendance automatically using a webcam.</li>
                    <li>View and download attendance reports.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Register New Student
    elif choice == "Register":
        st.subheader("Register New Student")
        newusername = st.text_input("Enter Name")
        newuserid = st.text_input("Enter Roll Number")
        newusermail=st.text_input("Enter Email")
        newuserbranch = st.selectbox("Select Branch", ["CSEA", "CSEB", "IT"])
        
        if st.button("Register"):
            if newusername and newuserid and newuserbranch:
                user_folder = f'static/faces/{newusername}_{newuserid}_{newuserbranch}_{newusermail}'
                os.makedirs(user_folder, exist_ok=True)

                # Capture 30 images using the webcam
                cap = cv2.VideoCapture(0)
                count = 0

                stframe = st.empty()
                while count < 30:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    faces = extract_faces(frame)
                    for (x, y, w, h) in faces:
                        face = frame[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, (50, 50))
                        cv2.imwrite(f"{user_folder}/{count}.jpg", face_resized)
                        count += 1

                    stframe.image(frame, channels="BGR", caption=f"Capturing... {count}/30")
                
                cap.release()
                stframe.empty()
                st.success(f"✅ {newusername} has been registered!")
                train_model()

    # Mark Attendance
    elif choice == "Mark Attendance":
        st.subheader("Mark Attendance")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = extract_faces(frame)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
                face = face.reshape(1, -1)
                model = joblib.load('static/face_recognition_model.pkl')
                identified_person = model.predict(face)[0]
                add_attendance(identified_person)

                # Extract just name and roll
                parts = identified_person.split('_')
                display_name = f"{parts[0]} ({parts[1]})" if len(parts) >= 2 else identified_person

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                stframe.image(frame, channels="BGR", caption=f"Attendance Marked for {display_name}")

                break

            stframe.image(frame, channels="BGR", caption="Scanning for faces...")

        cap.release()

    # View Attendance + Percentage with Download Option
    elif choice == "View Attendance":
        st.subheader("Today's Attendance")
        df = pd.read_csv(csv_path)
        st.dataframe(df)

        st.subheader("Attendance Percentage")
        start_date = st.date_input("Select Start Date", date(2025, 3, 9))
        
        if st.button("Calculate Percentage"):
            attendance_percentages = calculate_attendance_percentages(start_date.strftime('%d-%B-%Y'))
            st.dataframe(attendance_percentages)
  

            percentage_file = 'Attendance/attendance_percentages.xlsx'
            attendance_percentages.to_excel(percentage_file, index=False)

            with open(percentage_file, "rb") as f:
                st.download_button(
                    label="Download Attendance Percentages",
                    data=f,
                    file_name="attendance_percentages.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()