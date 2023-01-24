from .ssd_model import SSD300
from .ssd_model_utils_entrypoints import nvidia_ssd_processing_utils

import torch
import sys
from matplotlib import pyplot as plt
import matplotlib.patches as patches

precision = 'fp32'

#ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
#torch.load('C:\\Users\\ashcherbakov\\Desktop\\model_ssd.pt', map_location=torch.device('cpu'))    

ssd_model = SSD300() 
ssd_model.load_state_dict(torch.load('C:\\Users\\ashcherbakov\\Desktop\\modelssd_weights.pth', map_location=torch.device('cpu')))
ssd_model.eval()

utils = nvidia_ssd_processing_utils()

def predict(uris, confidence):
    inputs = [utils.prepare_input(uri) for uri in uris]
    tensor = utils.prepare_tensor(inputs)

    with torch.no_grad():
        detections_batch = ssd_model(tensor)

    results_per_input = utils.decode_results(detections_batch)
    best_results_per_input = [utils.pick_best(results, confidence) for results in results_per_input]
    classes_to_labels = utils.get_coco_object_dictionary()

    return inputs, best_results_per_input, classes_to_labels

def draw_results(inputs, results, labels):
    for image_idx in range(len(results)):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        # Show original, denormalized image...
        image = inputs[image_idx] / 2 + 0.5
        ax.imshow(image)
        # ...with detections
        bboxes, classes, confidences = results[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, "{} {:.0f}%".format(labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.3))
        plt.axis('off')
        plt.savefig('C:\\Users\\ashcherbakov\\Desktop\\CVDetection_app\\CVDetection_app\\cv_detection_app\\static\\cv_detection_app\\foo.png', bbox_inches='tight')
        plt.close(fig)
    return results[0]