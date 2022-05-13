from loader.loader import get_loader

def get_dataset_info(DATASET_NAME):
    if "ilab" in DATASET_NAME:
        loader_new = get_loader("multi_attribute_loader_file_list_ilab")
        file_list_root = "dataset_lists/ilab_lists/"
        att_path = "dataset_lists/combined_attributes.p"
        NUM_CLASSES = (6, 6, 6, 6)
    elif "mnist_rotation" in DATASET_NAME:
        loader_new = get_loader('multi_attribute_loader_file_list_mnist_rotation')
        file_list_root = "dataset_lists/mnist_rotation_lists/"
        att_path = "dataset_lists/combined_attributes.p"
        NUM_CLASSES = (10, 10, 10, 10)
    # elif "rotation_model" in DATASET_NAME:
    #     loader_new = get_loader("multi_attribute_loader_file_list")
    #     file_list_root = "dataset_lists/biased_cars_lists/"
    #     att_path = "data/biased_cars/att_dict_simplified.p"
    #     NUM_CLASSES = (5, 5, 5, 5)
    elif "rotation_model" in DATASET_NAME:
        loader_new = get_loader('multi_attribute_loader_file_list_semantic_segmentation')
        file_list_root = "dataset_lists/biased_cars_lists/"
        att_path = "data/biased_cars/att_dict_simplified.p"
        NUM_CLASSES = (5, 5, 5, 5)
    return loader_new, NUM_CLASSES, file_list_root, att_path