import time
import datetime
from cv2 import cv2
import numpy as np
from telegram.ext import (Updater, CommandHandler, CallbackContext)
import test
def timingFile():
    try:
        file = open('FallTimmings.txt', 'r')
        file.close()
    except:
        print("File not found!")
        file = open('FallTimmings.txt', 'w')
        file.close()

def start(update, context):
    update.message.reply_text("Heello")

def sendMessage(updater):
    updater.bot.sendMessage(chat_id='727907688', text='Your patient has fallen!')
def sendPhoto(updater):
    filename = 'D:/09 Projects and Blogs/EIG Project/Edited-XMan Tan/person_falling.jpg'
    updater.bot.send_photo(chat_id='727907688', photo=open(filename, 'rb'))

def main(update: Updater, context: CallbackContext):
    # Slowdown variables
    slowDown = 0
    allSlowDown = 0
    vidName = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorKNN()  # Edit
    fall_frames = 0
    fall_count = 0  # Times the program detected a "fall"
    fall_check = True
    MHI_DURATION = 30
    DEFAULT_THRESHOLD = 32
    ret, frames = vidName.read()
    mhi_h, mhi_w = frames.shape[:2]
    prev_frame = frames.copy()
    motion_history = np.zeros((mhi_h, mhi_w), np.float32)
    timestamp = 0


    first_frame = None
    run = True
    # Telegram bot functions

    token = "1646126067:AAEivXirCdUyTZuxGboeXvMenIdIWN2-bqo"
    updater = Updater(token=token, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    updater.start_polling()

    filename = 'D:/09 Projects and Blogs/EIG Project/Edited-XMan Tan/person_falling.jpg'
    updater.bot.sendMessage(chat_id='727907688', text='Application Started!')

    # Looks like updater.idle() loop and does not loop in while(run)
    while vidName.isOpened():
        ret, frames = vidName.read()

        try:
            status = "Empty"
            frame_diff = cv2.absdiff(frames, prev_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

            timestamp += 1
            prev_frame = frames.copy()

            # update motion history
            # cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION) # changed
            # normalize motion history
           # mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)

            # Convert each frames to gray scale and subtract the background
            gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray, (5, 5))

            # First frame - Changes
            if first_frame is None:
                first_frame = gray

            # Edges
            frame_delta = cv2.absdiff(first_frame, gray)
            edges = cv2.Canny(gray, 50, 50)

            # fgmask processing
            fgmask = fgbg.apply(gray)
            fgmask = cv2.erode(fgmask, (3, 3))
            fgmask = cv2.dilate(fgmask, (5, 5))

            # Find contours
            contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            edge_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Changes

            # Do static background subtraction followed by canny edging of the the subtracted background

            if contours:
                # List to hold all areas
                areas = []
                edge_areas = []
                for contour in contours:
                    ar = cv2.contourArea(contour)
                    areas.append(ar)

                max_area = max(areas, default=1)
                max_area_index = areas.index(max_area)

                for contour in edge_contours:  # Changes
                    ar = cv2.contourArea(contour)
                    edge_areas.append(ar)

                edge_max_area = max(edge_areas, default=1)
                edge_max_area_index = edge_areas.index(edge_max_area)

                # Contours
                cnt = contours[max_area_index]
                x, y, w, h = cv2.boundingRect(cnt)
                ed_cnt = edge_contours[edge_max_area_index]
                if edge_max_area > 3:  # Changes
                    cv2.drawContours(frames, ed_cnt, -1, (0, 255, 0), 3)

                    # Weed out specks and shakes
                if max_area > 1000:
                    if w > h:
                        fall_frames += 1

                        if fall_frames > 15:
                            if not fall_check:
                                # Save The Falls
                                fall_count += 1
                                file = open('FallTimmings.txt', 'a')
                                file.write("{}\n".format(datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")))
                                file.close()
                                file = open('Fall.txt', 'w')
                                file.write("Fall: {}\n".format(fall_count))
                                file.close()
                                fall_check = True
                                cv2.imwrite('person_falling.jpg', frames)


                            status = "Fell"
                            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            sendMessage(updater)
                            updater.bot.send_photo(chat_id='727907688', photo=open(filename, 'rb'))
                            time.sleep(slowDown)

                        else:
                            # Ensure there is an rectangle on the contour
                            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if h >= w:
                        status = "Occupied"
                        fall_frames = 0
                        fall_check = False
                        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Text written no video
            cv2.putText(frames, "Room Status: {0}".format(status), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        2)
            cv2.putText(frames, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frames.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # Show each frame
            cv2.imshow('video', frames)
            cv2.imshow('FGmask', fgmask)
            cv2.imshow('Edges', edges)
            # cv2.imshow('motion-history', mh)
            time.sleep(allSlowDown)

            # Stop code using "q"
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            break
    cv2.destroyAllWindows()



if __name__ == "__main__":
    #test.test()
    main(context=CallbackContext, update=Updater)
