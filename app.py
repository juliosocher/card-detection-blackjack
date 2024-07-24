from flask import Flask, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Load YOLOv8 model
model = YOLO('model_50_epochs.pt') 

# Define a dictionary to map class IDs to card labels
class_labels = {
    0: ('10C', 10),
    1: ('10D', 10),
    2: ('10H', 10),
    3: ('10S', 10),
    4: ('2C', 2), 
    5: ('2D', 2),
    6: ('2H', 2), 
    7: ('2S', 2),
    8: ('3C', 3),
    9: ('3D', 3),
    10: ('3H', 3),
    11: ('3S', 3),
    12: ('4C', 4),
    13: ('4D', 4),
    14: ('4H', 4),
    15: ('4S', 4),
    16: ('5C', 5),
    17: ('5D', 5),
    18: ('5H', 5),
    19: ('5S', 5),
    20: ('6C', 6),
    21: ('6D', 6),
    22: ('6H', 6),
    23: ('6S', 6),
    24: ('7C', 7),
    25: ('7D', 7),
    26: ('7H', 7),
    27: ('7S', 7),
    28: ('8C', 8),
    29: ('8D', 8),
    30: ('8H', 8),
    31: ('8S', 8),
    32: ('9C', 9),
    33: ('9D', 9),
    34: ('9H', 9),
    35: ('9S', 9),
    36: ('AC', 11), # Ace can also be considered 1 and it needs to be coded accordingly
    37: ('AD', 11),
    38: ('AH', 11),
    39: ('AS', 11),
    40: ('JC', 10),
    41: ('JD', 10),
    42: ('JH', 10),
    43: ('JS', 10),
    44: ('KC', 10),
    45: ('KD', 10),
    46: ('KH', 10),
    47: ('KS', 10),
    48: ('QC', 10),
    49: ('QD', 10),
    50: ('QH', 10),
    51: ('QS', 10)
}

def generate_frames():
     # Access the webcam
    cap = cv2.VideoCapture(0)  

    # Initialize dictionaries to store detected cards and their frame counts
    detected_cards_count = {}
    detected_cards = set()
    sum_detected_cards = 0
    consecutive_frame_threshold = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        results = model(frame)

        # Get the annotated frame with detections
        annotated_frame = results[0].plot()
        
        # Temporarily store the current frame's detected cards
        current_frame_cards = set()

        # Loop through the detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = class_labels.get(class_id, 'Unknown')
            current_frame_cards.add(label)
        
        # Update the detected cards count
        for card in current_frame_cards:
            if card in detected_cards_count:
                detected_cards_count[card] += 1
            else:
                detected_cards_count[card] = 1
        
        # Remove cards that were not detected in the current frame
        cards_to_remove = [card for card in detected_cards_count if card not in current_frame_cards]
        for card in cards_to_remove:
            detected_cards_count[card] -= 1
            if detected_cards_count[card] <= 0:
                detected_cards_count.pop(card)

        # Add cards to the detected cards set if they appear for 3 consecutive frames
        for card, count in detected_cards_count.items():
            if count >= consecutive_frame_threshold:
                detected_cards.add(card)
            else:
                detected_cards.discard(card)

        # Display the detected cards at the top right corner
        y_offset = 20
        for card_label in detected_cards:
            cv2.putText(annotated_frame, f'Card: {card_label[0]}', (annotated_frame.shape[1] - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 20
        
        # Summing the displayed cards
        sum_detected_cards = sum(value for _, value in detected_cards)

        num_of_aces = 0

        for _, value in detected_cards:
             if value == 11:
                  num_of_aces += 1

        if sum_detected_cards > 21 and num_of_aces > 0:
             sum_detected_cards = sum_detected_cards - 10 * num_of_aces

        cv2.putText(annotated_frame, f'Total Value: {sum_detected_cards}', (annotated_frame.shape[1] - 300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Displaying the message about BlackJack or Explosion
        if sum_detected_cards > 21:
                message = 'Exploded!'
        elif sum_detected_cards == 21:
                message = 'BLACKJACK!'
        else:
                message = ''

        if message:
            # Get the size of the message text
            screen_height = annotated_frame.shape[0]
            screen_width = annotated_frame.shape[1]
            font_scale = screen_height / 5 / 20  # Adjust font scale based on 20% of screen height
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)[0]

            # Calculate the center position for the text
            text_x = (screen_width - text_size[0]) // 2
            text_y = (screen_height + text_size[1]) // 2

            # Draw the message in the middle of the screen
            cv2.putText(annotated_frame, message, (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 255) if message == 'Exploded!' else (0, 255, 255), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
