
rootpath=../VisualSearch
testCollection=msrvtt_test
logger_name=../VisualSearch/msrvtt10ktrain/cvpr_2019/msrvtt10kval/dual_encoding_concate_full_dp_0.2_measure_exp/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_resnet-152-img1k-flatten0_outputos_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048/loss_func_cont_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0001_decay_0.99_grad_clip_2.0_val_metric_recall/runs_0
n_caption=20
overwrite=1

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py $testCollection --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name --n_caption $n_caption

