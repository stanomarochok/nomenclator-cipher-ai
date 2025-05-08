import os
import glob
import cv2
import argparse
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# ─── CONFIG ──────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = "../../scripts/yolo/runs/detect/train14/weights/best.pt"
DEFAULT_VAL_DIR = "../../materials/dataset/word_symbol_annotations_yolo/yolo_dataset_2/images/val_demo" # _generalization"
DEFAULT_OUT_PATH = "../../scripts/yolo/runs/detect/train14/img66_left.jpg"
DEFAULT_IMAGE_PATH = "../../materials/dataset/word_symbol_annotations_yolo/yolo_dataset_2/images/val_demo/img66_left.jpg"

DEFAULT_GRID_ROWS = 1
DEFAULT_GRID_COLS = 2
DEFAULT_IMG_SIZE = 1280
DEFAULT_DEVICE = "0"
DEFAULT_FONT_SIZE = 20
DEFAULT_LINE_WIDTH = 6
DEFAULT_DRAW_ROWS = True
DEFAULT_DRAW_COLS = True
DEFAULT_DRAW_TABLES = False
DEFAULT_EPS_ROW = 30
DEFAULT_EPS_COL = 50
DEFAULT_MODE = "single"
DEFAULT_ACTION = "show"
DEFAULT_AUTO_OUT_NAME = True


# ────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO and tile predictions into a grid or single image.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="path to model.pt")
    parser.add_argument("--val-dir", default=DEFAULT_VAL_DIR, help="directory of validation images")
    parser.add_argument("--out", default=DEFAULT_OUT_PATH, help="output path for result image")
    parser.add_argument("--default-image", default=DEFAULT_IMAGE_PATH,
                        help="default image path to use if none specified")
    parser.add_argument("--rows", type=int, default=DEFAULT_GRID_ROWS, help="grid rows (only for grid mode)")
    parser.add_argument("--cols", type=int, default=DEFAULT_GRID_COLS, help="grid cols (only for grid mode)")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMG_SIZE, help="inference image size")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="device id or 'cpu'")
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE, help="font size")
    parser.add_argument("--line-width", type=int, default=DEFAULT_LINE_WIDTH, help="box line width")
    parser.add_argument("--mode", choices=["single", "grid"], default=DEFAULT_MODE,
                        help="process a single image or build a grid")
    parser.add_argument("--action", choices=["save", "show"], default=DEFAULT_ACTION,
                        help="whether to save or show the result")
    parser.add_argument("--draw-rows", action='store_true', default=DEFAULT_DRAW_ROWS, help="draw rows")
    parser.add_argument("--draw-cols", action='store_true', default=DEFAULT_DRAW_COLS, help="draw columns")
    parser.add_argument("--draw-tables", action='store_true', default=DEFAULT_DRAW_TABLES, help="draw table structure")
    parser.add_argument("--eps-row", type=int, default=DEFAULT_EPS_ROW, help="row clustering threshold")
    parser.add_argument("--eps-col", type=int, default=DEFAULT_EPS_COL, help="column clustering threshold")
    parser.add_argument("--auto-out-name", action='store_true', default=DEFAULT_AUTO_OUT_NAME,
                        help="auto-generate output filename based on input")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--images", nargs="+", metavar="PATH", help="specific images to use")
    group.add_argument("--indices", nargs="+", type=int, metavar="N", help="indices into val-dir")
    return parser.parse_args()


def select_images(args):
    if args.images:
        return args.images

    if args.indices:
        all_paths = sorted(glob.glob(os.path.join(args.val_dir, "*.*")))
        try:
            return [all_paths[i] for i in args.indices]
        except IndexError as e:
            raise RuntimeError(f"Index error: {e}")

    if args.mode == "single":
        return [args.default_image]
    else:
        all_paths = sorted(glob.glob(os.path.join(args.val_dir, "*.*")))
        return all_paths[:args.rows * args.cols]


def annotate_images(results, font_size, line_width, args):
    annotated = []
    for r in results:
        img = r.orig_img.copy()

        boxes = r.boxes.xyxy.cpu().numpy()[:, :4] if r.boxes else np.empty((0, 4))
        scores = r.boxes.conf.cpu().numpy() if r.boxes else np.empty((0,))
        classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes else np.empty((0,))
        names = r.names if hasattr(r, 'names') else {}

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names.get(cls, cls)} {score:.2f}" if names else f"{cls} {score:.2f}"
            color = (255, 0, 0) if names.get(cls, '') == 'word' else (255, 255, 0)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
            font_scale = font_size / 100.0
            cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

        if args.draw_tables:
            img = draw_table_structure(
                img,
                boxes,
                row_gap=args.eps_row,  # <- use row_gap instead of eps_row
                col_gap=args.eps_col,  # <- use col_gap instead of eps_col
                draw_rows=args.draw_rows,
                draw_cols=args.draw_cols
            )
        annotated.append(img)
    return annotated


def make_grid(imgs, rows, cols):
    heights = [im.shape[0] for im in imgs]
    widths = [im.shape[1] for im in imgs]
    target_h = min(heights)
    target_w = min(widths)

    resized = [cv2.resize(im, (target_w, target_h)) for im in imgs]
    grid = np.zeros((target_h * rows, target_w * cols, 3), dtype=np.uint8)

    for idx, im in enumerate(resized):
        r, c = divmod(idx, cols)
        grid[r * target_h:(r + 1) * target_h, c * target_w:(c + 1) * target_w] = im
    return grid


def draw_table_structure(image, boxes, row_gap=50, col_gap=50, draw_rows=True, draw_cols=True, color=(255, 0, 0), thickness=2):
    if len(boxes) == 0:
        return image

    h, w = image.shape[:2]

    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)

    rows = []
    current_row = []
    last_center_y = None

    for box in boxes:
        center_y = (box[1] + box[3]) / 2
        if last_center_y is None or abs(center_y - last_center_y) < row_gap:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
        last_center_y = center_y
    if current_row:
        rows.append(current_row)

    x_centers = []
    for row in rows:
        row = sorted(row, key=lambda b: (b[0] + b[2]) / 2)
        x_centers.extend([(b[0] + b[2]) / 2 for b in row])

    x_centers = np.array(sorted(x_centers))

    if len(x_centers) > 0:
        clusters = [x_centers[0]]
        for xc in x_centers[1:]:
            if abs(xc - clusters[-1]) > col_gap:
                clusters.append(xc)

        if draw_cols:
            for x in clusters[1:-1]:
                x = int(round(x))
                cv2.line(image, (x, 0), (x, h), color, thickness)

    if draw_rows:
        for row in rows[1:-1]:
            y_mean = int(round(np.mean([(b[1] + b[3]) / 2 for b in row])))
            cv2.line(image, (0, y_mean), (w, y_mean), color, thickness)

    return image


def output_result(image, out_path, action, window_name="Result", window_width=900):
    if action == "save":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, image)
        print(f"Saved output to {out_path}")
    else:
        # Make window resizable
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)

        # Optionally, resize the window to a preferred initial size
        h, w = image.shape[:2]
        if w > window_width:
            scale = window_width / w
            cv2.resizeWindow(window_name, int(w * scale), int(h * scale))

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_inference(model, img_paths, device, imgsz):
    return model(img_paths, device=device, imgsz=imgsz, verbose=False)


def main():
    args = parse_args()
    img_paths = select_images(args)

    if args.mode == "grid" and len(img_paths) != args.rows * args.cols:
        raise RuntimeError(f"Expected {args.rows * args.cols} images, got {len(img_paths)}")

    model = YOLO(args.model)
    results = run_inference(model, img_paths, args.device, args.imgsz)
    annotated_imgs = annotate_images(results, args.font_size, args.line_width, args)

    if args.mode == "grid":
        final_image = make_grid(annotated_imgs, args.rows, args.cols)
    else:
        final_image = annotated_imgs[0]

    output_result(final_image, args.out, args.action)


if __name__ == "__main__":
    main()
