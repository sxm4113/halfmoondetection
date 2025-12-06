import cv2
import numpy as np
import os 
  
def merge_lines(lines, angle_thresh=5):
    merged = []
    used = [False]*len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue
        x1,y1,x2,y2 = lines[i][0]
        angle_i = np.degrees(np.arctan2(y2-y1, x2-x1))
        pts = [(x1,y1),(x2,y2)]
        used[i] = True

        # collect similar lines
        for j in range(i+1, len(lines)):
            if used[j]:
                continue
            xx1,yy1,xx2,yy2 = lines[j][0]
            angle_j = np.degrees(np.arctan2(yy2-yy1, xx2-xx1))
            if abs(angle_i - angle_j) < angle_thresh:
                pts.extend([(xx1,yy1),(xx2,yy2)])
                used[j] = True

        # project points onto line direction
        pts = np.array(pts)
        vx, vy = np.cos(np.radians(angle_i)), np.sin(np.radians(angle_i))
        proj = pts[:,0]*vx + pts[:,1]*vy
        min_idx = np.argmin(proj)
        max_idx = np.argmax(proj)
        merged.append((pts[min_idx][0], pts[min_idx][1],
                       pts[max_idx][0], pts[max_idx][1]))
    return merged

def detection_midpoint(ful_img):
 
    # Load grayscale
    #ful_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if ful_img is None:
        raise ValueError("Image not found")
    
    img = cv2.subtract(ful_img, np.full_like(ful_img, 30))

    # Crop margin
    margin = 3
    cropped = img[margin:img.shape[0]-margin, margin:img.shape[1]-margin]

    # Edge detection
    edges = cv2.Canny(cropped, 20, 60)

    # Hough line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                            minLineLength=50, maxLineGap=15)
    if lines is None:
        raise ValueError("No line detected")
    # Merge similar lines
    merged_lines = merge_lines(lines)

    # Pick longest merged line
    lengths = []
    for (x1,y1,x2,y2) in merged_lines:
        lengths.append((np.hypot(x2-x1, y2-y1), (x1,y1,x2,y2)))
    lengths.sort(key=lambda t: t[0], reverse=True)
    x1,y1,x2,y2 = lengths[0][1]

    # Midpoint
    mid = ((x1+x2)//2, (y1+y2)//2)
    mid_orig = (mid[0]+margin, mid[1]+margin)
    print("Merged line endpoints:", (x1+margin,y1+margin), (x2+margin,y2+margin))
    print("Midpoint of merged line:", mid_orig)

    # Visualization
    out = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    cv2.line(out, (x1,y1), (x2,y2), (0,255,0), 2)   # merged line
    cv2.circle(out, mid, 5, (0,0,255), -1)          # midpoint
    cv2.imshow("Merged Line Midpoint", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
def enhance_contrast(img):
    """
    Simple contrast enhancement using CLAHE (adaptive histogram equalization). 
    """
    if len(img.shape) == 2:  # grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)
    else:  # color image
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l2 = clahe.apply(l)
        lab = cv2.merge((l2,a,b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
file_dir = r"C:\Users\sangy\Documents\test_c\half-moon_cvi\narrow"
files = os.listdir(file_dir)
for file in files:
    if file.lower().endswith(".tif"):
        filename = os.path.join(file_dir, file)

        # Read .tif image
        ful_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # or IMREAD_COLOR if needed
        if ful_img is None:
            print(f"Could not read {filename}")
            continue

        # Contrast enhancement
        enhanced_img = enhance_contrast(ful_img)

        # Call your detection function
        detection_midpoint(enhanced_img)
    