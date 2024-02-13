result_dir="kitti_final_results"

python scripts/plot_sequence_results.py \
--r "$result_dir"/eb_dense169_results_all00_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_results_all02_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_results_all03_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_results_all04_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_results_all05_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_results_all08_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_results_all09_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_results_all10_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all00_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all02_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all03_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all04_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all05_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all08_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all09_sub5_gap_10.pkl \
--r "$result_dir"/eb_dense169_nosub_results_all10_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all00_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all02_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all03_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all04_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all05_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all08_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all09_sub5_gap_10.pkl \
--r "$result_dir"/sift_results_all10_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all00_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all02_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all03_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all04_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all05_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all08_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all09_sub5_gap_10.pkl \
--r "$result_dir"/lift_results_all10_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all00_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all02_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all03_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all04_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all05_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all08_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all09_sub5_gap_10.pkl \
--r "$result_dir"/d2net_results_all10_sub5_gap_10.pkl \
--merge --simple --rename

# --r "$result_dir"/eb_dense169_d2sub_results_all00_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_d2sub_results_all02_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_d2sub_results_all03_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_d2sub_results_all04_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_d2sub_results_all05_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_d2sub_results_all08_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_d2sub_results_all09_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_d2sub_results_all10_sub5_gap_10.pkl \
#

# --r "$result_dir"/sift_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/sift_results_all06_sub5_gap_10.pkl \
# --r "$result_dir"/sift_results_all07_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_results_all06_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_results_all07_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_nosub_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_nosub_results_all06_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_nosub_results_all07_sub5_gap_10.pkl \


# --r "$result_dir"/eb_dense169_compressed_results_all00_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all02_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all03_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all04_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all05_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all06_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all07_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all08_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all09_sub5_gap_10.pkl \
# --r "$result_dir"/eb_dense169_compressed_results_all10_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all06_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all07_sub5_gap_10.pkl \

# --r "$result_dir"/eb_grid/eb_dense169_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/eb_grid/eb_dense169_results_all06_sub5_gap_10.pkl \
# --r "$result_dir"/eb_grid/eb_dense169_results_all07_sub5_gap_10.pkl \

# original resnet configs
# --r "$result_dir"/resnet_224_res5c_results_all00_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all02_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all03_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all04_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all05_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all06_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all07_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all08_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all09_sub5_gap_10.pkl \
# --r "$result_dir"/resnet_224_res5c_results_all10_sub5_gap_10.pkl \


# --r "$result_dir"/vgg16_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/vgg16_results_all03_sub5_gap_10.pkl \
# --r "$result_dir"/vgg16_results_all04_sub5_gap_10.pkl \
# --r "$result_dir"/vgg11_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/vgg11_results_all03_sub5_gap_10.pkl \
# --r "$result_dir"/vgg11_results_all04_sub5_gap_10.pkl \
# --r "$result_dir"/alex_results_all01_sub5_gap_10.pkl \
# --r "$result_dir"/alex_results_all03_sub5_gap_10.pkl \
# --r "$result_dir"/alex_results_all04_sub5_gap_10.pkl \
