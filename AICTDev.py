# Copyright @ Mark S. Hong

import pydicom
import numpy as np
import cv2
import copy

def making_img(file_name):
    f = pydicom.dcmread(file_name)
    ds = f.pixel_array
    cvt_ds = (((ds-np.min(ds))/np.max(ds-np.min(ds)))*255).astype(np.uint8)
    img = cv2.merge([cvt_ds,cvt_ds,cvt_ds])
    # image shape = (512,512,3)
    return img

# def making_video_from_imgs(imgs_array):
#     out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'mp4v'),15,(512,512))
#     for i in range(len(imgs_array)):
#         out.write(imgs_array[i])
#     out.release()

def investigate_img(img, fn):
    # fn is used in out definition line
    # Total images list with contour and text
    # Because threshold 0~255
    sudo_cont_img = []
    # Initial Video
    out = cv2.VideoWriter('project_%s.mp4' %fn, cv2.VideoWriter_fourcc(*'mp4v'),15,(512,512))
    for i in range(255):
        new_img = copy.copy(img)

        cont_img = []
        ret, img_result = cv2.threshold(new_img, i, 255, cv2.THRESH_BINARY)
        img_result_gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
        # image_result_gray shape = (512,512)
        contours, hierarchy = cv2.findContours(img_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for j in range(len(contours)):
            new_cont_img = copy.copy(new_img)
            if cv2.contourArea(contours[j]) > 500:
                cv2.drawContours(new_cont_img, contours, j, (0,255,0),2)
                # Moment function
                mmt = cv2.moments(contours[j])
                cx = int(mmt['m10']/mmt['m00'])
                cy = int(mmt['m01']/mmt['m00'])
                # Moment function

                cv2.putText(new_cont_img, "%d / %d / %d" %(j, len(contours),i), (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                cv2.putText(new_cont_img, "IMG/Cont/THRESH", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                cv2.putText(new_cont_img, "AREA:"+str(cv2.contourArea(contours[j])),(20, 110), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                # Display the moment
                cv2.putText(new_cont_img, "%d / %d" %(cx, cy), (20, 480), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                cont_img.append(new_cont_img)
            else:
                pass
        sudo_cont_img.extend(cont_img)
        # for k in range(len(cont_img)):
        #     cv2.imshow("img", cont_img[k])
        #     cv2.waitKey(0)
    for l in range(len(sudo_cont_img)):
        out.write(sudo_cont_img[l])
    out.release()
    cv2.destroyAllWindows()
    return
def single_img(img):
    cont_img = []
    ret, img_result = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    img_result_gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(img_result_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for j in range(len(contours)):
        new_img = copy.copy(img)
        if cv2.contourArea(contours[j]) > 500:
            cv2.drawContours(new_img, contours, j, (0,255,0),2)
            # Moment dictionary
            mmt = cv2.moments(contours[j])
            # print(mmt)
            cx = int(mmt['m10']/mmt['m00'])
            cy = int(mmt['m01']/mmt['m00'])

            # Moments
            # 1. Spatial Moments
            #   m00, m10, m01, m20, m11, m02, m30, m21, m12, m03 (10)
            # 2. Central Moments
            #   mu20, mu11, mu02, mu30, mu21, mu12, mu03 (7)
            # 3. Central Normalized Moments
            #   nu20, nu11, nu02, nu30, nu21, nu03 (6)
        
            cv2.putText(new_img, "%d / %d" %(j, len(contours)), (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(new_img, "IMG/Cont/THRESH", (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(new_img, "AREA:"+str(cv2.contourArea(contours[j])),(20, 110), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            # Display the moment
            cv2.putText(new_img, "Moment %d / %d" %(cx, cy), (20, 480), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cont_img.append(new_img)
        else:
            pass
    for i in range(len(cont_img)):
        cv2.imshow("test", cont_img[i])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def main_serial():
    ### appendix start
    file_name_ds = []
    for i in range(275):
        fnd = "IM-0001-0%03d.dcm" %(i+1)
        file_name_ds.append(fnd)
    ### appendix end
    for j in range(275):
        file_name = file_name_ds[j]
        data = making_img(file_name)
        investigate_img(data, file_name[9:12])

def main():
    file_name = input("Type the file name >>>")
    data = making_img(file_name)
    investigate_img(data, file_name)

if __name__ == '__main__':
    main()
