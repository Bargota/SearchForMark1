import cv2
import numpy  as np
import cv2.aruco as aruco
import os

def main():
    #ShowGridBoard()
    Video2()

def Test():
    #img =ShowGridBoard()
    img  = cv2.imread('C:\\v.orlov\\Programm\\python\\SearchForMark1\\Mark\\bsZTs.jpg',0)
    img = RotateImage(img,45)
    #ShowImage(img)
    img = DetectImage(img)
    ShowImage(img)

def RotateImage(img,angle):
    rows,cols = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
    
def DetectImage(img):
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        
        # Make sure all 5 markers were detected before printing them out
    if ids is not None and len(ids) >= 1:
        # Print corners and ids to the console
        for i, corner in zip(ids, corners):
            print('ID: {}; Corners: {}'.format(i, corner))

        # Outline all of the markers detected in our image
        QueryImg = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 255,0))
    return img


def ShowImage(img):
    cv2.imshow('QueryImage', img)
    cv2.waitKey()
    cv2.destroyAllWindows() 

        

def FindMarksInImage(image):
    aruco.detectMarkers(image)

def ShowGridBoard():
    # Create gridboard, which is a set of Aruco markers
    # the following call gets a board of markers 5 wide X 7 tall
    gridboard = aruco.GridBoard_create(
            markersX=1, 
            markersY=1, 
            markerLength=0.09, 
            markerSeparation=0.01, 
            dictionary=aruco.Dictionary_get(aruco.DICT_6X6_1000))

    # Create an image from the gridboard
    #img = gridboard.draw(outSize=(988, 1400))
    img = gridboard.draw(outSize=(800, 800))
    #cv2.imwrite("test_gridboard.jpg", img)


    ## Display the image to us
    ShowImage(img)
    return img


def ShowOneMark():
    '''
    drawMarker(...)
        drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
    '''
 
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
    print(aruco_dict)
    # second parameter is id number
    # last parameter is total image size
    img = aruco.drawMarker(aruco_dict, 3, 700)
    cv2.imwrite("test_marker.jpg", img)
 
    cv2.imshow('frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    
def Video():
    # Constant parameters used in Aruco methods
    ARUCO_PARAMETERS = aruco.DetectorParameters_create()
    ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_1000)

    # Create grid board object we're using in our stream
    board = aruco.GridBoard_create(
            markersX=1,
            markersY=1,
            markerLength=0.09,
            markerSeparation=0.01,
            dictionary=ARUCO_DICT)

    # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None

    cam = cv2.VideoCapture(0)


    #calibrationFile = "C:\\v.orlov\\Programm\python\\SearchForMark1\\SearchForMark\\calibrationFileName.xml"
    calibrationFile = "calibrationFileName.xml"
    calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
    camera_matrix = calibrationParams.getNode("cameraMatrix").mat()
    dist_coeffs = calibrationParams.getNode("distCoeffs").mat()

    while(cam.isOpened()):
        # Capturing each frame of our video stream
        ret, QueryImg = cam.read()
        if ret == True:

            # grayscale image
            gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
            #ShowImage(gray)
            
            # Detect Aruco markers
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
        
            # Make sure all 5 markers were detected before printing them out
            if ids is not None and len(ids) >= 1:
                # Print corners and ids to the console
                for i, corner in zip(ids, corners):
                    os.system('cls')
                    print('ID: {}; Corners: {}'.format(i, corner))

                # Outline all of the markers detected in our image
                #QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))
                #QueryImg=aruco.drawAxis(QueryImg, cam)
                markerLength = 20
                rvec, tvec = aruco.estimatePoseSingleMarkers(corners, markerLength, camera_matrix, dist_coeffs) # For a single marker
                imgWithAruco = aruco.drawDetectedMarkers(QueryImg, corners, ids, (0,255,0))
                QueryImg = aruco.drawAxis(imgWithAruco, camera_matrix, dist_coeffs, rvec, tvec, 100)

                # Wait on this frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Display our image
            cv2.imshow('QueryImage', QueryImg)


        # Exit at the end of the video on the 'q' keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows() 

def Video2():
    # Load Calibrated Parameters
    calibrationFile = "calibrationFileName.xml"
    calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ)
    camera_matrix = calibrationParams.getNode("cameraMatrix").mat()
    dist_coeffs = calibrationParams.getNode("distCoeffs").mat()

    r = calibrationParams.getNode("R").mat()
    new_camera_matrix = calibrationParams.getNode("newCameraMatrix").mat()

    image_size = (1920, 1080)
    #map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, dist_coeffs, r, new_camera_matrix, image_size, cv2.CV_16SC2)



    aruco_dict = aruco.getPredefinedDictionary( aruco.DICT_6X6_1000 )
    markerLength = 20   # Here, our measurement unit is centimetre.
    markerSeparation = 4   # Here, our measurement unit is centimetre.
    board = aruco.GridBoard_create(1, 1, markerLength, markerSeparation, aruco_dict)
    arucoParams = aruco.DetectorParameters_create()
    img = board.draw(outSize=(800, 800))
    #ShowImage(img)


    videoFile = 0
    cap = cv2.VideoCapture(videoFile)

    while(True):
        ret, frame = cap.read() # Capture frame-by-frame
        if ret == True:
            #frame_remapped = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)     # for fisheye remapping
            frame_remapped_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # aruco.detectMarkers() requires gray image

            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame_remapped_gray, aruco_dict, parameters=arucoParams)  # First, detect markers
            aruco.refineDetectedMarkers(frame_remapped_gray, board, corners, ids, rejectedImgPoints)

            if ids != None: # if there is at least one marker detected
                im_with_aruco_board = aruco.drawDetectedMarkers(frame, corners, ids, (0,255,0))
                
                retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs)  # posture estimation from a aruco board
                if retval != 0:
                    x = np.array([[10], [-10],[0]], np.int32)
                    new_tvec = tvec+x
                    im_with_aruco_board = aruco.drawAxis(im_with_aruco_board, camera_matrix, dist_coeffs, rvec, new_tvec, 50)  # axis length 100 can be changed according to your requirement
            else:
                im_with_aruco_board = frame

            cv2.imshow("arucoboard", im_with_aruco_board)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        else:
            break


    cap.release()   # When everything done, release the capture
    cv2.destroyAllWindows()


main() 
