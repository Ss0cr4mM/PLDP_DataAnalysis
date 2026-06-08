"""
find_region.py
--------------
Interactive tool to find the (x, y, width, height) region of a number in an image.

Two modes:
  1. Click-and-drag  → draws a rectangle, prints the region when you release
  2. Hover mode      → prints coordinates as you move the mouse (--hover flag)

Usage:
    python find_region.py <image_path>
    python find_region.py <image_path> --hover

Dependencies:
    pip install opencv-python
"""

import cv2
import sys

# ── State ─────────────────────────────────────────────────────────────────────
drawing = False
start_x = start_y = 0
end_x = end_y = 0
original = None
canvas = None


def on_mouse(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        # Always print current coords in the window title
        cv2.setWindowTitle("find_region", f"Cursor: ({x}, {y})")
        if drawing:
            end_x, end_y = x, y
            # Redraw rectangle live
            canvas = original.copy()
            cv2.rectangle(canvas, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y

        rx = min(start_x, end_x)
        ry = min(start_y, end_y)
        rw = abs(end_x - start_x)
        rh = abs(end_y - start_y)

        if rw > 2 and rh > 2:
            print(f"\n✅  Region found!")
            print(f"    region = ({rx}, {ry}, {rw}, {rh})")
            print(f"\n    Use it like this:")
            print(f"    read_number_from_image('your_image.png', region=({rx}, {ry}, {rw}, {rh}))\n")

            # Draw final rectangle with label
            canvas = original.copy()
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            label = f"({rx}, {ry}, {rw}, {rh})"
            cv2.putText(canvas, label, (rx, ry - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


def main():
    global original, canvas

    if len(sys.argv) < 2:
        print("Usage: python find_region.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    original = cv2.imread(path)

    if original is None:
        print(f"Error: could not load '{path}'")
        sys.exit(1)

    # Resize large images so they fit on screen (doesn't affect coordinates)
    h, w = original.shape[:2]
    max_dim = 1000
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        original = cv2.resize(original, (int(w * scale), int(h * scale)))
        if scale != 1.0:
            print(f"ℹ️  Image scaled to {scale:.2f}x for display.")
            print(f"   Coordinates shown are in the SCALED image.")
            print(f"   To get original coordinates, divide by {scale:.4f}\n")

    canvas = original.copy()

    print("=" * 55)
    print("  REGION FINDER")
    print("=" * 55)
    print("  • Click and drag to draw a rectangle around your number")
    print("  • The region tuple will be printed in the terminal")
    print("  • Press 'r' to reset, 'q' or ESC to quit")
    print("=" * 55)

    cv2.namedWindow("find_region", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("find_region", on_mouse)

    while True:
        cv2.imshow("find_region", canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):   # q or ESC
            break
        elif key == ord('r'):       # reset
            canvas = original.copy()
            print("Canvas reset.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()