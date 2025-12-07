import cv2
import numpy as np
import os
import click
  
def merge_lines(lines, angle_thresh=5):
    """
    Merge collinear line segments into longer lines.
    lines: list of (x1,y1,x2,y2) tuples
    angle_thresh: maximum angle difference (degrees) to consider lines collinear
    """
    merged = []
    used = [False] * len(lines)

    for i in range(len(lines)):
        if used[i]:
            continue

        x1, y1, x2, y2 = lines[i]
        angle_i = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        pts = [(x1, y1), (x2, y2)]
        used[i] = True

        # collect similar lines
        for j in range(i+1, len(lines)):
            if used[j]:
                continue
            xx1, yy1, xx2, yy2 = lines[j]
            angle_j = np.degrees(np.arctan2(yy2 - yy1, xx2 - xx1))
            if abs(angle_i - angle_j) < angle_thresh:
                pts.extend([(xx1, yy1), (xx2, yy2)])
                used[j] = True

        # project points onto line direction
        pts = np.array(pts)
        vx, vy = np.cos(np.radians(angle_i)), np.sin(np.radians(angle_i))
        proj = pts[:,0] * vx + pts[:,1] * vy
        min_idx = np.argmin(proj)
        max_idx = np.argmax(proj)

        merged.append((pts[min_idx][0], pts[min_idx][1],
                       pts[max_idx][0], pts[max_idx][1]))

    return merged

def extend_line(x1, y1, x2, y2, edges, max_extend=100):
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        return (x1, y1, x2, y2)

    dx, dy = dx / length, dy / length  # unit vector

    # extend backwards
    x_start, y_start = x1, y1
    for i in range(max_extend):
        xi, yi = int(x_start - dx), int(y_start - dy)
        if xi < 0 or yi < 0 or xi >= edges.shape[1] or yi >= edges.shape[0]:
            break
        if edges[yi, xi] == 0:
            break
        x_start, y_start = xi, yi

    # extend forwards
    x_end, y_end = x2, y2
    for i in range(max_extend):
        xi, yi = int(x_end + dx), int(y_end + dy)
        if xi < 0 or yi < 0 or xi >= edges.shape[1] or yi >= edges.shape[0]:
            break
        if edges[yi, xi] == 0:
            break
        x_end, y_end = xi, yi

    return (x_start, y_start, x_end, y_end)


def detection_midpoint(img, debug=False, debug_dir="debug_outputs"):
    if img is None:
        raise ValueError("Image not found")

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[DEBUG] Input image pixel range: min={np.min(img)}, max={np.max(img)}")

    # Crop margin
    margin = 3
    cropped = img[margin:img.shape[0]-margin, margin:img.shape[1]-margin]
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "02_cropped.png"), cropped)

    # Edge detection
    edges = cv2.Canny(cropped, 20, 60)
    if debug:
        print(f"[DEBUG] Edge map pixel range: min={np.min(edges)}, max={np.max(edges)}")
        cv2.imwrite(os.path.join(debug_dir, "03_edges.png"), edges)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                            threshold=15, minLineLength=30, maxLineGap=60)
    if lines is None:
        raise ValueError("No line detected")

    # Filter lines by length (35–280 px)
    filtered_lines = []
    for x1, y1, x2, y2 in lines[:,0]:
        length = np.hypot(x2 - x1, y2 - y1)
        if 35 <= length <= 280:
            filtered_lines.append((x1, y1, x2, y2))

    if not filtered_lines:
        raise ValueError("No lines within specified length range")

    if debug:
        print(f"[DEBUG] Found {len(lines)} raw lines, {len(filtered_lines)} within length range")
        
        out_lines = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        for (x1, y1, x2, y2) in filtered_lines:
            cv2.line(out_lines, (x1, y1), (x2, y2), (255, 0, 0), 1)  # blue lines
        cv2.imwrite(os.path.join(debug_dir, "filtered_lines.png"), out_lines)
        cv2.imshow("Filtered Lines", out_lines)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    merged_lines = merge_lines(filtered_lines)

    # Pick longest line
    lengths = [(np.hypot(x2-x1, y2-y1), (x1,y1,x2,y2)) for (x1,y1,x2,y2) in merged_lines]
    lengths.sort(key=lambda t: t[0], reverse=True)
    x1, y1, x2, y2 = lengths[0][1]

    # Extend the chosen line along edges
    x1, y1, x2, y2 = extend_line(x1, y1, x2, y2, edges, max_extend=300)

    # Midpoint
    mid = ((x1+x2)//2 + margin, (y1+y2)//2 + margin)
 
    # Visualization
    out = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    cv2.line(out, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.circle(out, mid, 5, (0,0,255), -1)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, "04_final.png"), out)
        cv2.imshow("Final", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imshow("Merged Line Midpoint", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mid


def enhance_contrast(img, saturated=0.35):
    """
    Enhance contrast similar to ImageJ's 'Enhance Contrast' with saturated pixels.
    """
    flat = img.flatten()
    low_cut = np.percentile(flat, saturated)
    high_cut = np.percentile(flat, 100 - saturated)

    img_clipped = np.clip(img, low_cut, high_cut)
    img_norm = cv2.normalize(img_clipped, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm.astype(np.uint8)
 
@click.command()
@click.option('--file-dir', required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory containing .tif images")
@click.option('--debug', is_flag=True, default=False,
              help="Enable debug mode for extra logging and visualization")
def main(file_dir, debug):
    files = os.listdir(file_dir)
    for file in files:
        if file.lower().endswith(".tif"):
            filename = os.path.join(file_dir, file)
            full_img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            if full_img is None:
                print(f"Could not read {filename}")
                continue

            if debug:
                print(f"[DEBUG] ORIGINAL IMAGE: {full_img.dtype}, {np.min(full_img)}, {np.max(full_img)}")

            enhanced_img = enhance_contrast(full_img)

            try:
                detection_midpoint(enhanced_img, debug=debug)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    main()