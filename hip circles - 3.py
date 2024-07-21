import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize mediapipe pose class and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Constants
total_rounds = 3
total_time = 60  # Total time in seconds for each round
rest_time = 5  # Rest time between rounds

# Initialize variables to store scores
round_scores = []
total_score = 0
total = [0,0]

# Initialize OpenCV window
cv2.namedWindow('Mediapipe', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mediapipe', 800, 600)  # Resize window for better visualization

for round_num in range(total_rounds):
    print(f"\nRound {round_num + 1}")

    flagLiftUp = False
    flagRightUp = False
    flagLiftSide = False
    flagRightSide = False
    flagLiftBack = False
    flagRightBack = False
    count = [0,0]

    start_time = time.time()  # Get current time when the round starts

    cap = cv2.VideoCapture(0)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor from BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
          
            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            remaining_time = max(total_time - int(elapsed_time), 0)
            
            #Make a rectangle
            cv2.rectangle(image, (0,0), (220, 200), (128, 128, 128), -1)

            # Display timer on the screen
            cv2.putText(image, f"Round {round_num + 1} - Time left: {remaining_time}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            # Display current round score
            cv2.putText(image, f"Round {round_num + 1} Score R: {count[1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.putText(image, f"Round {round_num + 1} Score L: {count[0]}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            # Display total score so far
            cv2.putText(image, f"Total Score: {total_score}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Check if time is up
            if elapsed_time > total_time:
                break

            # Extract landmarks
            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates for left knee
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z]

                    # Get coordinates for right knee
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z]
                    
                    # Get coordinates for left ankle
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z]

                    # Get coordinates for right ankle
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z]


                    if left_knee[1] < right_knee[1] * 0.9 and left_ankle[1] < right_ankle[1] * 0.9 and left_knee[2] < -0.3:
                        flagLiftUp = True
                    elif left_knee[1] > right_knee[1] * 0.9 and left_ankle[1] > right_ankle[1] * 0.9:
                        flagLiftUp = False


                    if flagLiftUp and -0.3 < left_knee[2] < 0.3:
                        flagLiftSide = True
                    elif left_knee[2] < -0.3:
                        flagLiftSide = False


                    if flagLiftUp and flagLiftSide and left_ankle[2] > 0.3:
                        #cv2.putText(image, 'Nice Work', (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        flagLiftBack = True
                    elif flagLiftBack and left_knee[1] > right_knee[1] * 0.9 and left_ankle[1] > right_ankle[1] * 0.9 and left_ankle[2] < 0.3:
                        flagLiftBack = False
                        count[0] += 1


                    if right_knee[1] < left_knee[1] * 0.9 and right_ankle[1] < left_ankle[1] * 0.9 and right_knee[2] < -0.3:
                        flagRightUp = True
                    elif right_knee[1] > left_knee[1] * 0.9 and right_ankle[1] > left_ankle[1] * 0.9:
                        flagRightUp = False


                    if flagRightUp and -0.3 < right_knee[2] < 0.3:
                        flagRightSide = True
                    elif right_knee[2] < -0.3:
                        flagRightSide = False


                    if flagRightUp and flagRightSide and right_ankle[2] > 0.3:
                        #cv2.putText(image, 'Nice Work', (350, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        flagRightBack = True
                    elif flagRightBack and right_knee[1] > left_knee[1] * 0.9 and right_ankle[1] > left_ankle[1] * 0.9 and right_ankle[2] < 0.3:
                        flagRightBack = False
                        count[1] += 1


                    cv2.putText(image, 'Count: ' + str(sum(count)) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


                except Exception as e:
                    print(f"Error processing landmarks: {e}")

           
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               
            
            cv2.imshow('Mediapipe', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    # Calculate score for the round
    round_score = sum(count)
    round_scores.append(round_score)
    print(f"Score for Round {round_num + 1}: {round_score}")

    # Add round score to total score
    total_score += round_score

    total[0] += count[0]
    total[1] += count[1]

    # Rest between rounds
    if round_num < total_rounds - 1:
        print(f"Resting for {rest_time} seconds before next round...")
        time.sleep(rest_time)

# Print total score after all rounds are completed
print(f"\nTotal Score across all rounds: {total_score}")
print(f"\nTotal Score for Left Hip: {total[0]}")
print(f"\nTotal Score for Right Hip: {total[1]}")


while True:
    gray_color = (128, 128, 128)  # RGB values for gray color
    blank_image = np.full((480, 640, 3), gray_color, dtype=np.uint8)

    # Display current round score
    cv2.putText(blank_image, f"Total Score R: {total[1]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(blank_image, f"Total Score L: {total[0]}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Display total score so far
    cv2.putText(blank_image, f"Total Score: {total_score}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    # Display the image in a window
    cv2.imshow('Statistics', blank_image)
    



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

