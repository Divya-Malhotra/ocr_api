import cv2
import numpy as np
import sys
import pytesseract
import tkinter as tk
from tkinter import ttk
import operator
from collections import Counter
np.set_printoptions(threshold=np.inf)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\divya.malhotra\\AppData\\Local\\Tesseract-OCR\\tesseract'
TESSDATA_PREFIX = 'C:\\Users\\divya.malhotra\\AppData\\Local\\Tesseract-OCR\\tessdata'
config_amount = ("-l eng --oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.,")
config = ("-l eng --oem 1 --psm 7")
config_full = ("-l eng --oem 01--psm 11")
NORM_FONT= ("Verdana", 10)

def popupmsg():
    msg = "Please click at right area to get text"
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

def find_remove_inner_contours(mask,img_orig,img_thresh,y_top_val,y_bottom_val,x_left_val,x_right_val):
    # i = img_orig.copy()
    # print("shape is ", mask.shape)
    # contours, hierarchy = cv2.findContours(mask[y_top_val:y_bottom_val,x_left_val:x_right_val], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,offset=(x_left_val,y_top_val))
    # for ind,evry_cnt in enumerate(contours):
    #     # if hierarchy[0, ind, 3] == -1:
    #     x, y, w, h = cv2.boundingRect(evry_cnt)
    #     cv2.rectangle(img_orig, (x, y), (x + w, y + h), (255, 0, 255), 2)
    #     cv2.imshow("mmask_rect",img_orig)
    #     if ((h > (mask.shape[0]/3)) & (h != mask.shape[0] & w != mask.shape[1])):
    #         cv2.rectangle(mask, (x, y), (x + w, y + h), (255,255, 255), -1)
    # cv2.imshow("masked_im", mask)
    # np.savetxt('masked.csv',mask,delimiter=',')

    for cl in range(img_thresh.shape[1]):
         start = 0
         end = img_thresh.shape[0]
         first_black_pixel_mask = 0
         cnt = 1
         while(start<end):
             colm = np.where(mask[start:end, cl] == 0)[0]
             if len(colm) > 0:
                 first_black_pixel_mask= colm[0] + start
                 if mask[first_black_pixel_mask,cl] == mask[first_black_pixel_mask-1,cl]:
                     cnt=cnt+1
                     start = first_black_pixel_mask+1
                 else:
                     start= first_black_pixel_mask+1
                     cnt=1
                 if cnt>end/20:
                     mask[:,cl] =255
                     break
             else:
                 break
         continue
    # compute_connected_components(mask)
    return mask

def find_bill_content(edges,r,c,img_thresh,img):
    img_orig = img.copy()
    final_contours = []
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for ind, cn in enumerate(contours):
        if hierarchy[0,ind,3] == -1:
            x, y, w, h = cv2.boundingRect(cn)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            final_contours.append(cn)
    # cv2.imshow("all_contours", img)
    x_left = []
    y_top = []
    x_right = []
    y_bottom = []
    contours_tobe_removed = []
    for evry_cnt in final_contours:
        x,y,w,h= cv2.boundingRect(evry_cnt)
        if ((h>r/2) and (w>=.4*c)):
            print("dimension::",x,y,w,h)
            print("area::", cv2.contourArea(evry_cnt))
            cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
            x_left.append(x)
            y_top.append(y)
            x_right.append(x+w)
            y_bottom.append(y+h)
        elif ((h>r/13) and (w<=.08*c)):
            img_thresh = cv2.rectangle(img_thresh,(x,y),(x+w,y+h),(255,255,255),-1)

    isBillLargestContour = True if ((len(x_left)>0) & (len(x_right)>0) & (len(y_top)>0) & (len(y_bottom)>0))  else False
    # cv2.imshow("image_bill_contour", img)
    if isBillLargestContour:
        x_left_val = min(x_left)
        x_right_val = max(x_right)
        y_top_val = min(y_top)
        y_bottom_val = max(y_bottom)
        mask = np.zeros(img_thresh.shape, dtype='uint8')
        mask.fill(255)
        mask[y_top_val:y_bottom_val, x_left_val:x_right_val] = img_thresh[y_top_val:y_bottom_val,x_left_val:x_right_val]
        # cv2.imshow("masked image", mask)
        img_thresh = find_remove_inner_contours(mask,img_orig,img_thresh,y_top_val,y_bottom_val,x_left_val,x_right_val)
    # cv2.imshow("final_im", img_thresh)
    return img_thresh

def check_if_amount(img_thresh, mouseY, first_black_pixel, last_black_pixel):
    index_zero = np.where(img_thresh[mouseY,first_black_pixel:last_black_pixel])[0]
    index_zero[:] = index_zero[::-1]
    cnt = 0
    start_pt = 0
    for indx, elm in enumerate(index_zero[start_pt:]):
        start_pt_backup = start_pt
        if elm == index_zero[indx-1]-1:
            cnt = cnt + 1
            if cnt >= .1 * img_thresh.shape[1]:
                break
        else:
            cnt = 0
            start_pt = indx+1
    return index_zero[start_pt_backup]+first_black_pixel-10

def pre_process_image(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    # Convert image to gray scale
    img_gray = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
    # Apply normal thresholding to get only black and white pixels in image
    thresh, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    thresh, img_thresh = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    # additional changes
    blur_img = cv2.GaussianBlur(img_thresh, (5, 5), 0)
    edges = cv2.Canny(blur_img, 70, 110, -1)
    return img_thresh, img_gray, edges, result_norm

def check_retailer(img_thresh,mouseY,retailer_x):
    index_left_x = np.where(img_thresh[mouseY, :mouseX])[0]
    index_right_x = np.where(img_thresh[mouseY, mouseX:])[0]
    index_right_x = [elm+retailer_x for elm in index_right_x]
    index_left_x[:] = index_left_x[::-1]
    cnt = 0
    start_pt = 0
    for indx, pt in enumerate(index_left_x[start_pt:]):
        if index_left_x[indx]> -1:
            if pt == index_left_x[indx - 1] - 1:
                cnt = cnt + 1
                if cnt >= .1 * img_thresh.shape[1]:
                    break
            else:
                cnt = 0
                start_pt = indx + 1
                start_pt_backup = start_pt
    x_left = index_left_x[start_pt_backup]  - 10

    cnt = 0
    start_pt = 0
    for indx, pt in enumerate(index_right_x[start_pt:]):
        if index_right_x[indx] < img_thresh.shape[1]-1:
            if pt == index_right_x[indx + 1] - 1:
                cnt = cnt + 1
                if cnt >= .1 * img_thresh.shape[1]:
                    break
            else:
                cnt = 0
                start_pt = indx + 1
                start_pt_backup = start_pt
    x_right = index_right_x[start_pt_backup] + 10
    return x_left, x_right

def check_amount(img_thresh,mouseY,amount_x):
    index_left_x = np.where(img_thresh[mouseY, :mouseX])[0]
    index_right_x = np.where(img_thresh[mouseY, mouseX:])[0]
    index_right_x = [elm+amount_x for elm in index_right_x]
    index_left_x[:] = index_left_x[::-1]
    cnt = 0
    start_pt = 0
    for indx, pt in enumerate(index_left_x[start_pt:]):
        if index_left_x[indx]> -1:
            if pt == index_left_x[indx - 1] - 1:
                cnt = cnt + 1
                if cnt >= .05 * img_thresh.shape[1]:
                    break
            else:
                cnt = 0
                start_pt = indx + 1
                start_pt_backup = start_pt
    x_left = index_left_x[start_pt_backup]  - 10

    cnt = 0
    start_pt = 0
    for indx, pt in enumerate(index_right_x[start_pt:]):
        if index_right_x[indx] < img_thresh.shape[1]-1:
            if pt == index_right_x[indx + 1] - 1:
                cnt = cnt + 1
                if cnt >= .05 * img_thresh.shape[1]:
                    break
            else:
                cnt = 0
                start_pt = indx + 1
                start_pt_backup = start_pt
    x_right = index_right_x[start_pt_backup] + 10
    return x_left, x_right



def cal_text_height(img_thresh,img,mouseY, bottom_y,retailer_x,amount_x, top_y=0):
    mouseY_backup = mouseY
    row,col,color= img.shape
    # np.savetxt('thresholded.csv', img_thresh, delimiter=',')
    # np.savetxt('mousePointer.csv',img_thresh[mouseY,:], delimiter=",")
    only_white_char = np.where(img_thresh[mouseY, :] == 255)
    # print(only_white_char)
    first_white_pixel = only_white_char[0][0]
    last_white_pixel = only_white_char[0][-1]
    only_black_chars = np.where(img_thresh[mouseY,first_white_pixel:last_white_pixel]==0)
    if len(only_black_chars[0])==0:
        popupmsg()
        sys.exit()
    first_black_pixel = only_black_chars[0][0] + first_white_pixel
    last_black_pixel = only_black_chars[0][-1] + first_white_pixel
    if is_amount:
        # first_black_pixel = check_if_amount(img_thresh, mouseY, first_black_pixel, last_black_pixel)
        first_black_pixel, last_black_pixel = check_amount(img_thresh, mouseY, amount_x)
    else:
        first_black_pixel, last_black_pixel = check_retailer(img_thresh,mouseY, retailer_x)
    list_above_min = []
    list_below_max = []
    list_right_max = []
    list_left_min = []

    for elm in range(first_black_pixel,last_black_pixel+1):
        mouseY = mouseY - 1
        while (mouseY>top_y):
            if is_amount:
                # first_pixel = check_if_amount(img_thresh, mouseY, first_black_pixel, last_black_pixel)
                # last_pixel = last_black_pixel
                first_pixel, last_pixel = check_amount(img_thresh, mouseY,amount_x)
            else:
                first_pixel, last_pixel = check_retailer(img_thresh, mouseY,retailer_x)
            list_left_min.append(first_pixel)
            list_right_max.append(last_pixel)
            if ((img_thresh[mouseY, elm] == 255)):
                list_above_min.append(mouseY)
                break
            else:
                mouseY = mouseY - 1
        mouseY = mouseY_backup
    mouseY = mouseY_backup

    for elm in range(first_black_pixel,last_black_pixel+1):
        mouseY = mouseY+1
        while (mouseY<bottom_y):
            if is_amount:
                # first_pixel = check_if_amount(img_thresh, mouseY, first_black_pixel, last_black_pixel)
                # last_pixel = last_black_pixel
                first_pixel, last_pixel = check_amount(img_thresh, mouseY,amount_x)
            else:
                first_pixel, last_pixel = check_retailer(img_thresh, mouseY, retailer_x)
            list_left_min.append(first_pixel)
            list_right_max.append(last_pixel)
            if ((img_thresh[mouseY, elm] == 255)):
                list_below_max.append(mouseY)
                break
            else:
                mouseY = mouseY + 1
        mouseY = mouseY_backup

    if (len(list_above_min) ==0 | len(list_below_max)==0):
        popupmsg()
        sys.exit()
    top_y = min(list_above_min)
    bottom_y = max(list_below_max)
    right_x = max(list_right_max)
    left_x = min(list_left_min)
    return (top_y,bottom_y,right_x,left_x)

def ocr_func(img_location,retailer_x,retailer_y,amount_x,amount_y):
    global is_amount, mouseY, mouseX
    img_path = img_location
    print(img_location)
    i = 2
    is_amount = False
    for val in range(2):
        if val == 0:
            mouseX = retailer_x
            mouseY = retailer_y
        else:
            is_amount = True
            mouseX = amount_x
            mouseY = amount_y
        print(mouseY, mouseX)
        # Read input image
        img = cv2.imread(img_path)
        img_input = img.copy()
        # Pre process input image
        cleaned_image, grayed_img, edges, normalized_image = pre_process_image(img)
        row, col = grayed_img.shape
        # find largest contour coordinates
        clean_bill_image = find_bill_content(edges, row, col, cleaned_image, img)
        cv2.imshow("clean Image", clean_bill_image)
        y1, y2, x2, x1 = cal_text_height(clean_bill_image, img, mouseY,img.shape[0], retailer_x,amount_x)
        cropped_image = grayed_img[y1 - 4:y2 + 4, x1 - 4:x2 + 4]
        # cv2.imshow("cropped", cropped_image)
        # cv2.waitKey(0)
        if not is_amount:
            text = pytesseract.image_to_string(cropped_image, config=config)
            print("text is ", text)
        if is_amount:
            amount = pytesseract.image_to_string(cropped_image, config=config_amount)
            print("amount is ", amount)
    return text, amount

if __name__ == '__main__':
    print("Welcome")
    global row, column
    global mouseX, mouseY, is_amount, retailer_x, amount_x
    img_location= sys.argv[1]
    retailer_x = int(sys.argv[2])
    retailer_y = int(sys.argv[3])
    amount_x = int(sys.argv[4])
    amount_y = int(sys.argv[5])
    text, amount = ocr_func(img_location,retailer_x,retailer_y,amount_x,amount_y)


