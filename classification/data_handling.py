import os
import random
import numpy as np
import h5py


def get_bounding_box_from_segmentation(raw_images,
                                       cell_instance_segmentation,
                                       infected_cell_mask):
    raw_cells = []
    cell_labels = []
    for img, seg_mask, infect_mask in zip(raw_images,
                                          cell_instance_segmentation,
                                          infected_cell_mask):
        cell_ids = np.unique(seg_mask)[1:]

        # split mask into binary masks
        masks = (seg_mask == cell_ids[:, None, None])

        # get bounding box coordinates for each masks
        num_cells = len(cell_ids)

        for i in range(num_cells):
            pos = np.where(masks[i])
            x0 = np.min(pos[1])
            x1 = np.max(pos[1])
            y0 = np.min(pos[0])
            y1 = np.max(pos[0])

            # crop out raw cell from image
            raw_cell = img[:, y0:y1, x0:x1]

            # extract label from nuclei infection map
            labeled_nucleus = infect_mask[y0:y1, x0:x1].copy()
            labeled_nucleus[~masks[i][y0:y1, x0:x1]] = 0
            labels, counts = np.unique(labeled_nucleus, return_counts=True)

            if len(labels) in {0, 1}:
                continue
            else:
                cell_label = labels[np.argmax(counts[1:]) + 1] - 1

            raw_cells += [raw_cell]
            cell_labels += [cell_label]

    assert len(raw_cells) == len(cell_labels)

    c = list(zip(raw_cells, cell_labels))
    random.shuffle(c)
    cells, labels = zip(*c)

    return (cells, labels)


def load_data_from_dir(root_dir):
    files = [os.path.join(root_dir, f)
             for f in os.listdir(root_dir)]
    raw, cells, infected = [], [], []
    for path_to_file in files:
        with h5py.File(path_to_file) as f:
            raw += [f['raw'][:]]
            cells += [f['cells'][:]]
            infected += [f['infected'][:]]
    return raw, cells, infected


def main():
    # Data handling
    raw, cells, infected = load_data_from_dir(ROOT_DIR)
    raw_cells, cell_labels = get_bounding_box_from_segmentation(raw, cells, infected)


if __name__=='__main__':
    ROOT_DIR = "data/CovidGroundTruth"
    main()
